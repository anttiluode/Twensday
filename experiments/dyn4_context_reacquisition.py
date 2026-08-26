from __future__ import annotations

import argparse
import numpy as np


class SlowModel:
    def __init__(self, dim: int, lr: float = 0.0025) -> None:
        self.w = np.zeros(dim, dtype=float)
        self.lr = lr

    def score(self, z: np.ndarray) -> float:
        return float(self.w @ z)

    def update(self, z: np.ndarray, y: int, fast_state: float) -> None:
        yhat = np.tanh(self.score(z) + fast_state)
        self.w += self.lr * (y - yhat) * z
        n = np.linalg.norm(self.w)
        if n > 8.0:
            self.w *= 8.0 / n


class ContextCache:
    def __init__(self, capacity: int, policy: str, seed: int, n_contexts: int) -> None:
        self.capacity = capacity
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.ctx = np.full(capacity, -1, dtype=int)
        self.state = np.zeros(capacity, dtype=float)
        self.priority = np.zeros(capacity, dtype=float)
        self.age = np.zeros(capacity, dtype=int)
        self.ptr = 0
        self.n_contexts = n_contexts
        if policy == "full_table":
            self.table = np.full(n_contexts, np.nan)
        else:
            self.table = None

    def lookup(self, ctx: int):
        if self.policy == "none":
            return None
        if self.policy == "full_table":
            v = self.table[ctx]
            return None if np.isnan(v) else float(v)
        hits = np.flatnonzero(self.ctx == ctx)
        if len(hits) == 0:
            return None
        return float(self.state[int(hits[-1])])

    def write(self, ctx: int, state: float, utility: float, now: int, oracle_priority: float = 0.0) -> None:
        if self.policy == "none":
            return
        if self.policy == "full_table":
            self.table[ctx] = state
            return

        # Update existing stable address in place.
        hits = np.flatnonzero(self.ctx == ctx)
        if len(hits):
            i = int(hits[-1])
            self.state[i] = state
            self.priority[i] = utility if self.policy != "oracle" else oracle_priority
            self.age[i] = now
            return

        empty = np.flatnonzero(self.ctx < 0)
        if len(empty):
            i = int(empty[0])
        elif self.policy == "fifo":
            i = self.ptr
            self.ptr = (self.ptr + 1) % self.capacity
        elif self.policy == "random":
            i = int(self.rng.integers(0, self.capacity))
        elif self.policy in {"residual", "oracle"}:
            # Weak aging lets stale contexts eventually yield even if they were
            # once useful. Residual priority is measured from actual block value.
            aged = self.priority * np.exp(-(now - self.age) / 6000.0)
            i = int(np.argmin(aged))
            incoming = utility if self.policy == "residual" else oracle_priority
            if incoming <= aged[i]:
                return
        else:
            raise ValueError(self.policy)

        self.ctx[i] = ctx
        self.state[i] = state
        self.priority[i] = utility if self.policy != "oracle" else oracle_priority
        self.age[i] = now

    def scramble_addresses(self) -> None:
        if self.policy in {"none", "full_table"}:
            return
        self.rng.shuffle(self.ctx)


def make_world(
    steps: int,
    seed: int,
    dim: int = 12,
    n_contexts: int = 48,
    block_len: int = 25,
):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=dim)
    w /= np.linalg.norm(w) + 1e-12
    offsets = rng.choice(
        np.array([-1.0, -0.7, -0.4, -0.2, 0.2, 0.4, 0.7, 1.0]),
        size=n_contexts,
        replace=True,
    ).astype(float)

    # Randomize context order once, then recur in that same long cycle. The
    # stable identity is available to the cache but carries no algebraic clue
    # about the random offset.
    order = rng.permutation(n_contexts)
    ctx = np.empty(steps, dtype=int)
    z = rng.normal(size=(steps, dim))
    y = np.empty(steps, dtype=int)

    for t in range(steps):
        block = t // block_len
        c = int(order[block % n_contexts])
        ctx[t] = c
        raw = float(w @ z[t] + offsets[c] + 0.10 * rng.normal())
        y[t] = 1 if raw >= 0 else -1

    return z, y, ctx, offsets


def run_policy(
    seed: int,
    policy: str,
    steps: int = 12000,
    capacity: int = 12,
    n_contexts: int = 48,
    block_len: int = 25,
    scramble: bool = False,
):
    z, y, ctx, offsets = make_world(
        steps, seed, n_contexts=n_contexts, block_len=block_len
    )
    model = SlowModel(z.shape[1])
    cache = ContextCache(capacity, policy, seed + 70000, n_contexts)

    q = 0.0
    eta_q = 0.22
    correct = []
    first1_recur = []
    first5_recur = []
    late_recur = []
    first5_new = []
    high_residual_first1 = []
    cache_hits = []
    seen = np.zeros(n_contexts, dtype=int)

    block_base_loss = []
    block_fast_loss = []
    current_ctx = None
    current_was_recur = False
    current_hit = False

    for t in range(steps):
        pos = t % block_len
        c = int(ctx[t])

        if pos == 0:
            # Finish previous block: the value of its fast state is the memory
            # item; utility is how much that state improved over slow model alone.
            if current_ctx is not None:
                utility = max(
                    float(np.mean(block_base_loss) - np.mean(block_fast_loss)),
                    0.0,
                )
                cache.write(
                    current_ctx,
                    q,
                    utility,
                    now=t,
                    oracle_priority=abs(float(offsets[current_ctx])),
                )

            current_ctx = c
            current_was_recur = seen[c] > 0
            seen[c] += 1
            block_base_loss = []
            block_fast_loss = []

            retrieved = cache.lookup(c)
            current_hit = retrieved is not None
            cache_hits.append(float(current_hit) if current_was_recur else np.nan)
            q = 0.0 if retrieved is None else float(retrieved)

            if scramble and current_was_recur and (t // block_len) % 7 == 0:
                cache.scramble_addresses()

        slow_score = model.score(z[t])
        score = slow_score + q
        pred = 1 if score >= 0 else -1
        ok = float(pred == y[t])
        if t >= block_len * n_contexts:
            correct.append(ok)

        if current_was_recur:
            if pos == 0:
                first1_recur.append(ok)
                if abs(offsets[c]) >= 0.7:
                    high_residual_first1.append(ok)
            if pos < 5:
                first5_recur.append(ok)
            if pos >= block_len - 8:
                late_recur.append(ok)
        else:
            if pos < 5:
                first5_new.append(ok)

        target = int(y[t])
        base_yhat = np.tanh(slow_score)
        fast_yhat = np.tanh(score)
        block_base_loss.append(float((target - base_yhat) ** 2))
        block_fast_loss.append(float((target - fast_yhat) ** 2))

        # Fast state adapts within the current episode/context.
        q += eta_q * (target - fast_yhat)
        q = float(np.clip(q, -1.5, 1.5))

        # Slow model sees the residual after the fast state has had a chance to
        # explain context-specific bias. This helps it converge on the global rule.
        model.update(z[t], target, q)

    if current_ctx is not None:
        utility = max(float(np.mean(block_base_loss) - np.mean(block_fast_loss)), 0.0)
        cache.write(
            current_ctx,
            q,
            utility,
            now=steps,
            oracle_priority=abs(float(offsets[current_ctx])),
        )

    valid_hits = np.asarray([x for x in cache_hits if not np.isnan(x)], dtype=float)
    return {
        "overall_recurrence_accuracy": float(np.mean(correct)),
        "first1_recurrence": float(np.mean(first1_recur)),
        "first5_recurrence": float(np.mean(first5_recur)),
        "late_recurrence": float(np.mean(late_recur)),
        "first5_new_context": float(np.mean(first5_new)),
        "high_residual_first1": float(np.mean(high_residual_first1)),
        "cache_hit_rate": float(np.mean(valid_hits)) if len(valid_hits) else 0.0,
    }


def run_many(seeds: int, steps: int, capacity: int) -> dict:
    policies = ["none", "fifo", "random", "residual", "oracle", "full_table"]
    rows = []
    for s in range(seeds):
        r = {p: run_policy(s, p, steps=steps, capacity=capacity) for p in policies}
        r["scrambled_residual"] = run_policy(
            s,
            "residual",
            steps=steps,
            capacity=capacity,
            scramble=True,
        )
        rows.append(r)

    policies.append("scrambled_residual")
    metrics = list(rows[0][policies[0]])
    summary = {}
    for p in policies:
        summary[p] = {}
        for m in metrics:
            x = np.asarray([r[p][m] for r in rows], dtype=float)
            summary[p][m] = {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
            }
    return {
        "n_seeds": seeds,
        "steps": steps,
        "capacity": capacity,
        "summary": summary,
        "per_seed": rows,
    }


def print_summary(result: dict) -> None:
    print("\n=== DYN4: CONTEXT-STATE REACQUISITION ===")
    print(
        f"seeds={result['n_seeds']} steps={result['steps']} "
        f"finite_H_capacity={result['capacity']} contexts=48"
    )
    keys = [
        "first1_recurrence",
        "first5_recurrence",
        "late_recurrence",
        "high_residual_first1",
        "cache_hit_rate",
    ]
    print("policy".ljust(22) + " ".join(k[:11].rjust(12) for k in keys))
    for p, m in result["summary"].items():
        print(
            p.ljust(22)
            + " ".join(f"{m[k]['mean']:12.4f}" for k in keys)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--capacity", type=int, default=12)
    args = ap.parse_args()
    result = run_many(args.seeds, args.steps, args.capacity)
    print_summary(result)


if __name__ == "__main__":
    main()
