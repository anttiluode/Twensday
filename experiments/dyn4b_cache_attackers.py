from __future__ import annotations

import argparse
import numpy as np

from dyn4_context_reacquisition import SlowModel, make_world


class PriorityCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.ctx = np.full(capacity, -1, dtype=int)
        self.state = np.zeros(capacity, dtype=float)
        self.priority = np.zeros(capacity, dtype=float)
        self.age = np.zeros(capacity, dtype=int)

    def lookup(self, c: int):
        h = np.flatnonzero(self.ctx == c)
        return None if len(h) == 0 else float(self.state[int(h[-1])])

    def write(self, c: int, state: float, priority: float, now: int) -> None:
        h = np.flatnonzero(self.ctx == c)
        if len(h):
            i = int(h[-1])
        else:
            empty = np.flatnonzero(self.ctx < 0)
            if len(empty):
                i = int(empty[0])
            else:
                aged = self.priority * np.exp(-(now - self.age) / 6000.0)
                i = int(np.argmin(aged))
                if priority <= aged[i]:
                    return
        self.ctx[i] = c
        self.state[i] = state
        self.priority[i] = priority
        self.age[i] = now


def run(seed: int, policy: str, steps: int, capacity: int) -> dict:
    block_len = 25
    n_contexts = 48
    z, y, ctx, offsets = make_world(
        steps, seed, n_contexts=n_contexts, block_len=block_len
    )
    model = SlowModel(z.shape[1])
    cache = PriorityCache(capacity)
    q = 0.0
    eta_q = 0.22
    seen = np.zeros(n_contexts, dtype=int)

    first1 = []
    first5 = []
    high_first1 = []
    hit = []

    current_ctx = None
    was_recur = False
    base_loss: list[float] = []
    fast_loss: list[float] = []
    q_abs: list[float] = []

    def priority_value() -> float:
        if not base_loss:
            return 0.0
        if policy == "residual_improvement":
            return max(float(np.mean(base_loss) - np.mean(fast_loss)), 0.0)
        if policy == "base_error":
            return float(np.mean(base_loss))
        if policy == "state_magnitude":
            return float(np.mean(q_abs[-8:])) if q_abs else 0.0
        if policy == "final_state_magnitude":
            return abs(float(q))
        if policy == "late_fast_error":
            return float(np.mean(fast_loss[-8:]))
        raise ValueError(policy)

    for t in range(steps):
        pos = t % block_len
        c = int(ctx[t])
        if pos == 0:
            if current_ctx is not None:
                cache.write(current_ctx, q, priority_value(), t)
            current_ctx = c
            was_recur = seen[c] > 0
            seen[c] += 1
            base_loss = []
            fast_loss = []
            q_abs = []
            old = cache.lookup(c)
            hit.append(float(old is not None) if was_recur else np.nan)
            q = 0.0 if old is None else float(old)

        slow = model.score(z[t])
        score = slow + q
        pred = 1 if score >= 0 else -1
        ok = float(pred == y[t])

        if was_recur:
            if pos == 0:
                first1.append(ok)
                if abs(offsets[c]) >= 0.7:
                    high_first1.append(ok)
            if pos < 5:
                first5.append(ok)

        target = int(y[t])
        base_hat = np.tanh(slow)
        fast_hat = np.tanh(score)
        base_loss.append(float((target - base_hat) ** 2))
        fast_loss.append(float((target - fast_hat) ** 2))

        q += eta_q * (target - fast_hat)
        q = float(np.clip(q, -1.5, 1.5))
        q_abs.append(abs(q))
        model.update(z[t], target, q)

    if current_ctx is not None:
        cache.write(current_ctx, q, priority_value(), steps)

    h = np.asarray([v for v in hit if not np.isnan(v)], dtype=float)
    return {
        "first1_recurrence": float(np.mean(first1)),
        "first5_recurrence": float(np.mean(first5)),
        "high_residual_first1": float(np.mean(high_first1)),
        "cache_hit_rate": float(np.mean(h)),
    }


def run_many(seeds: int, steps: int, capacity: int) -> dict:
    policies = [
        "residual_improvement",
        "base_error",
        "state_magnitude",
        "final_state_magnitude",
        "late_fast_error",
    ]
    rows = [
        {p: run(s, p, steps, capacity) for p in policies}
        for s in range(seeds)
    ]
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
    print("\n=== DYN4B: WHAT SHOULD H RETAIN? ===")
    print(f"seeds={result['n_seeds']} steps={result['steps']} capacity={result['capacity']}")
    keys = ["first1_recurrence", "first5_recurrence", "high_residual_first1", "cache_hit_rate"]
    print("policy".ljust(24) + " ".join(k[:11].rjust(12) for k in keys))
    for p, m in result["summary"].items():
        print(p.ljust(24) + " ".join(f"{m[k]['mean']:12.4f}" for k in keys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--capacity", type=int, default=12)
    a = ap.parse_args()
    print_summary(run_many(a.seeds, a.steps, a.capacity))


if __name__ == "__main__":
    main()
