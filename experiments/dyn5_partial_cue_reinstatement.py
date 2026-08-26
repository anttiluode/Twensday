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


def aggregate_key(total: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key = np.zeros_like(total)
    mask = count > 0
    key[mask] = total[mask] / count[mask]
    return key, mask


def masked_cosine(
    query: np.ndarray,
    query_mask: np.ndarray,
    key: np.ndarray,
    key_mask: np.ndarray,
) -> float:
    mask = query_mask & key_mask
    if int(np.sum(mask)) < 3:
        return -1.0
    a = query[mask]
    b = key[mask]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return -1.0
    return float((a @ b) / denom)


class ContentMemory:
    """Finite content-addressed dormant-state memory.

    The algorithm never receives a context ID. The optional debug labels are
    stored only so the experiment can report whether a retrieved item happened
    to come from the same hidden world context; they are never used in lookup,
    replacement, merging, or prediction.
    """

    def __init__(
        self,
        capacity: int,
        replacement: str,
        seed: int,
        merge_threshold: float = 0.93,
    ) -> None:
        self.capacity = capacity
        self.replacement = replacement
        self.rng = np.random.default_rng(seed)
        self.merge_threshold = merge_threshold
        self.keys: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.states: list[float] = []
        self.priority: list[float] = []
        self.age: list[int] = []
        self.debug_context: list[int] = []

    def similarities(self, query: np.ndarray, query_mask: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                masked_cosine(query, query_mask, key, mask)
                for key, mask in zip(self.keys, self.masks)
            ],
            dtype=float,
        )

    def retrieve_nearest(
        self,
        query: np.ndarray,
        query_mask: np.ndarray,
        threshold: float,
    ) -> tuple[float | None, int | None, np.ndarray]:
        if not self.keys:
            return None, None, np.empty(0, dtype=float)
        sims = self.similarities(query, query_mask)
        i = int(np.argmax(sims))
        if sims[i] < threshold:
            return None, None, sims
        return float(self.states[i]), i, sims

    def retrieve_soft(
        self,
        query: np.ndarray,
        query_mask: np.ndarray,
        threshold: float,
        temperature: float,
        predictive_log_weight: np.ndarray | None = None,
    ) -> tuple[float | None, np.ndarray | None, np.ndarray]:
        if not self.keys:
            return None, None, np.empty(0, dtype=float)
        sims = self.similarities(query, query_mask)
        if float(np.max(sims)) < threshold:
            return None, None, sims
        logits = temperature * sims
        if predictive_log_weight is not None:
            logits = logits + predictive_log_weight
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        weights /= float(np.sum(weights)) + 1e-12
        return float(weights @ np.asarray(self.states)), weights, sims

    def write(
        self,
        key: np.ndarray,
        key_mask: np.ndarray,
        state: float,
        utility: float,
        now: int,
        debug_context: int,
    ) -> None:
        # Content-based merge only. No hidden context label is available here.
        if self.keys:
            sims = self.similarities(key, key_mask)
            i = int(np.argmax(sims))
            if sims[i] >= self.merge_threshold:
                old_key = self.keys[i]
                old_mask = self.masks[i]
                both = old_mask & key_mask
                only_new = key_mask & ~old_mask
                merged = old_key.copy()
                merged[both] = 0.70 * old_key[both] + 0.30 * key[both]
                merged[only_new] = key[only_new]
                self.keys[i] = merged
                self.masks[i] = old_mask | key_mask
                self.states[i] = 0.50 * self.states[i] + 0.50 * float(state)
                self.priority[i] = max(
                    0.70 * self.priority[i] + 0.30 * float(utility),
                    0.50 * float(utility),
                )
                self.age[i] = now
                self.debug_context[i] = debug_context
                return

        if len(self.keys) < self.capacity:
            i = len(self.keys)
            self.keys.append(key.copy())
            self.masks.append(key_mask.copy())
            self.states.append(float(state))
            self.priority.append(float(utility))
            self.age.append(now)
            self.debug_context.append(debug_context)
            return

        if self.replacement == "fifo":
            i = int(np.argmin(np.asarray(self.age)))
        elif self.replacement == "random":
            i = int(self.rng.integers(0, self.capacity))
        elif self.replacement == "residual":
            aged = np.asarray(self.priority) * np.exp(
                -(now - np.asarray(self.age)) / 6000.0
            )
            i = int(np.argmin(aged))
            if float(utility) <= float(aged[i]):
                return
        else:
            raise ValueError(self.replacement)

        self.keys[i] = key.copy()
        self.masks[i] = key_mask.copy()
        self.states[i] = float(state)
        self.priority[i] = float(utility)
        self.age[i] = now
        self.debug_context[i] = debug_context

    def scramble_states(self) -> None:
        if len(self.states) < 2:
            return
        p = self.rng.permutation(len(self.states))
        self.states = [self.states[int(i)] for i in p]


def make_world(
    steps: int,
    seed: int,
    dim: int = 12,
    families: int = 12,
    per_family: int = 4,
    block_len: int = 25,
    cue_dim: int = 32,
    cue_steps: int = 5,
    visible_fraction: float = 0.35,
    cue_noise: float = 0.12,
    unique_scale: float = 0.25,
):
    """Continuous recurring-context world with deliberately ambiguous cues.

    Contexts come in families. Members of a family share most of their cue but
    have different hidden offsets. A single partial cue often identifies the
    family but not the exact context. Several cue fragments plus task outcomes
    can disambiguate it.
    """

    rng = np.random.default_rng(seed)
    n_contexts = families * per_family

    global_w = rng.normal(size=dim)
    global_w /= np.linalg.norm(global_w) + 1e-12

    family_proto = rng.normal(size=(families, cue_dim))
    family_proto /= np.linalg.norm(family_proto, axis=1, keepdims=True) + 1e-12
    unique = rng.normal(size=(n_contexts, cue_dim))
    unique /= np.linalg.norm(unique, axis=1, keepdims=True) + 1e-12

    proto = np.empty((n_contexts, cue_dim), dtype=float)
    offsets = np.empty(n_contexts, dtype=float)
    offset_set = np.asarray([-1.0, -0.35, 0.35, 1.0], dtype=float)

    for f in range(families):
        assignment = rng.permutation(per_family)
        for j in range(per_family):
            c = f * per_family + j
            v = family_proto[f] + unique_scale * unique[c]
            proto[c] = v / (np.linalg.norm(v) + 1e-12)
            offsets[c] = float(offset_set[int(assignment[j])])

    # Same long recurrence cycle as DYN4, but the algorithm never sees this ID.
    order = rng.permutation(n_contexts)
    context = np.empty(steps, dtype=int)
    z = rng.normal(size=(steps, dim))
    target = np.empty(steps, dtype=int)
    cue = np.zeros((steps, cue_dim), dtype=float)
    cue_mask = np.zeros((steps, cue_dim), dtype=bool)

    for t in range(steps):
        c = int(order[(t // block_len) % n_contexts])
        context[t] = c
        raw = float(global_w @ z[t] + offsets[c] + 0.08 * rng.normal())
        target[t] = 1 if raw >= 0 else -1

        if t % block_len < cue_steps:
            mask = rng.random(cue_dim) < visible_fraction
            if int(np.sum(mask)) < 4:
                mask[rng.choice(cue_dim, size=4, replace=False)] = True
            noisy = proto[c] + cue_noise * rng.normal(size=cue_dim)
            cue[t, mask] = noisy[mask]
            cue_mask[t, mask] = True

    return z, target, context, offsets, cue, cue_mask


class ExactIDTable:
    """Diagnostic upper bound. This attacker is explicitly given hidden IDs."""

    def __init__(self, n_contexts: int) -> None:
        self.state = np.full(n_contexts, np.nan)

    def lookup(self, context: int) -> float | None:
        v = self.state[context]
        return None if np.isnan(v) else float(v)

    def write(self, context: int, state: float) -> None:
        self.state[context] = float(state)


def run_policy(
    seed: int,
    policy: str,
    steps: int = 12000,
    capacity: int = 24,
    block_len: int = 25,
    cue_steps: int = 5,
):
    z, target, hidden_context, offsets, cue, cue_mask = make_world(
        steps=steps,
        seed=seed,
        block_len=block_len,
        cue_steps=cue_steps,
    )
    n_contexts = len(offsets)
    cue_dim = cue.shape[1]

    model = SlowModel(z.shape[1])
    exact = ExactIDTable(n_contexts) if policy == "exact_id_table" else None

    if policy in {"none", "exact_id_table"}:
        memory = None
    else:
        replacement = "residual"
        if policy == "confirm_fifo":
            replacement = "fifo"
        elif policy == "confirm_random":
            replacement = "random"
        mem_capacity = 96 if policy == "unbounded_confirm" else capacity
        memory = ContentMemory(
            capacity=mem_capacity,
            replacement=replacement,
            seed=seed + 90000,
        )

    q = 0.0
    eta_q = 0.06
    threshold = 0.30
    temperature = 8.0
    predictive_beta = 5.0

    seen = np.zeros(n_contexts, dtype=int)
    current_context: int | None = None
    current_recurrence = False

    cue_total = np.zeros(cue_dim, dtype=float)
    cue_count = np.zeros(cue_dim, dtype=int)
    base_loss: list[float] = []
    fast_loss: list[float] = []
    predictive_log_weight: np.ndarray | None = None

    first1: list[float] = []
    first5: list[float] = []
    positions_2_to_5: list[float] = []
    late: list[float] = []
    hard_2_to_5: list[float] = []
    exact_retrieval: list[float] = []

    for t in range(steps):
        pos = t % block_len
        c = int(hidden_context[t])

        if pos == 0:
            # Finish previous block. The slow-model residual decides whether this
            # dormant q is worth scarce H capacity. Hidden context is debug only.
            if current_context is not None:
                if exact is not None:
                    exact.write(current_context, q)
                elif memory is not None:
                    key, key_mask = aggregate_key(cue_total, cue_count)
                    utility = max(
                        float(np.mean(base_loss) - np.mean(fast_loss)),
                        0.0,
                    )
                    memory.write(
                        key,
                        key_mask,
                        q,
                        utility,
                        now=t,
                        debug_context=current_context,
                    )

            current_context = c
            current_recurrence = seen[c] > 0
            seen[c] += 1
            q = 0.0
            cue_total = np.zeros(cue_dim, dtype=float)
            cue_count = np.zeros(cue_dim, dtype=int)
            base_loss = []
            fast_loss = []
            predictive_log_weight = None

            if exact is not None:
                retrieved = exact.lookup(c)
                if retrieved is not None:
                    q = retrieved

            if policy == "scrambled_confirm" and memory is not None:
                # Periodic lesion: content keys stay intact but dormant states are
                # detached from them. No hidden label is touched.
                if (t // block_len) % 11 == 0:
                    memory.scramble_states()

        if pos < cue_steps:
            mask = cue_mask[t]
            cue_total[mask] += cue[t, mask]
            cue_count[mask] += 1

        # Content-addressed retrieval. DYN5 never gets c here.
        retrieved = None
        retrieved_debug_context = None
        if memory is not None and pos < cue_steps:
            query, query_mask = aggregate_key(cue_total, cue_count)

            if policy == "eager_nearest":
                retrieved, idx, _ = memory.retrieve_nearest(
                    query, query_mask, threshold=threshold
                )
                if idx is not None:
                    retrieved_debug_context = memory.debug_context[idx]
                if retrieved is not None:
                    q = retrieved if pos == 0 else 0.80 * q + 0.20 * retrieved

            elif policy == "cue_soft":
                retrieved, weights, _ = memory.retrieve_soft(
                    query,
                    query_mask,
                    threshold=threshold,
                    temperature=temperature,
                )
                if weights is not None:
                    idx = int(np.argmax(weights))
                    retrieved_debug_context = memory.debug_context[idx]
                if retrieved is not None:
                    # Cautious cue-only reinstatement. It has no outcome evidence.
                    gate = 0.15 if pos == 0 else 0.35
                    q = (1.0 - gate) * q + gate * retrieved

            elif policy in {
                "confirm_residual",
                "confirm_fifo",
                "confirm_random",
                "unbounded_confirm",
                "scrambled_confirm",
            }:
                if predictive_log_weight is None or len(predictive_log_weight) != len(memory.keys):
                    predictive_log_weight = np.zeros(len(memory.keys), dtype=float)
                retrieved, weights, _ = memory.retrieve_soft(
                    query,
                    query_mask,
                    threshold=threshold,
                    temperature=temperature,
                    predictive_log_weight=predictive_log_weight,
                )
                if weights is not None:
                    idx = int(np.argmax(weights))
                    retrieved_debug_context = memory.debug_context[idx]
                if retrieved is not None:
                    # First cue only primes weakly. Once an actual outcome has
                    # updated candidate likelihoods, reinstatement becomes strong.
                    gate = 0.15 if pos == 0 else 0.75
                    q = (1.0 - gate) * q + gate * retrieved

            else:
                raise ValueError(policy)

        slow_score = model.score(z[t])
        score = slow_score + q
        pred = 1 if score >= 0 else -1
        ok = float(pred == target[t])

        if current_recurrence:
            if pos == 0:
                first1.append(ok)
            if pos < 5:
                first5.append(ok)
            if 1 <= pos < 5:
                positions_2_to_5.append(ok)
                if abs(float(offsets[c])) >= 0.9:
                    hard_2_to_5.append(ok)
            if pos >= block_len - 8:
                late.append(ok)
            if pos < cue_steps and retrieved_debug_context is not None:
                exact_retrieval.append(float(retrieved_debug_context == c))

        y = int(target[t])
        base_yhat = np.tanh(slow_score)
        fast_yhat = np.tanh(score)
        base_loss.append(float((y - base_yhat) ** 2))
        fast_loss.append(float((y - fast_yhat) ** 2))

        # This is the dynamic confirmation step. Outcome from the current event
        # does not retroactively change its prediction; it changes which dormant
        # candidate state is allowed to influence the *next* event.
        if (
            memory is not None
            and policy in {
                "confirm_residual",
                "confirm_fifo",
                "confirm_random",
                "unbounded_confirm",
                "scrambled_confirm",
            }
            and pos < cue_steps
            and len(memory.states) > 0
        ):
            if predictive_log_weight is None or len(predictive_log_weight) != len(memory.states):
                predictive_log_weight = np.zeros(len(memory.states), dtype=float)
            candidate_state = np.asarray(memory.states, dtype=float)
            candidate_yhat = np.tanh(slow_score + candidate_state)
            predictive_log_weight += -predictive_beta * (y - candidate_yhat) ** 2

        # Ordinary fast adaptation remains available to every policy.
        q += eta_q * (y - fast_yhat)
        q = float(np.clip(q, -1.5, 1.5))
        model.update(z[t], y, q)

    metrics = {
        "first1_recurrence": float(np.mean(first1)),
        "first5_recurrence": float(np.mean(first5)),
        "positions_2_to_5": float(np.mean(positions_2_to_5)),
        "hard_positions_2_to_5": float(np.mean(hard_2_to_5)),
        "late_recurrence": float(np.mean(late)),
        "exact_retrieval_debug": float(np.mean(exact_retrieval)) if exact_retrieval else 0.0,
    }
    return metrics


def run_many(seeds: int, steps: int, capacity: int) -> dict:
    policies = [
        "none",
        "eager_nearest",
        "cue_soft",
        "confirm_fifo",
        "confirm_random",
        "confirm_residual",
        "scrambled_confirm",
        "unbounded_confirm",
        "exact_id_table",
    ]

    rows = []
    for seed in range(seeds):
        rows.append(
            {
                policy: run_policy(
                    seed=seed,
                    policy=policy,
                    steps=steps,
                    capacity=capacity,
                )
                for policy in policies
            }
        )

    metrics = list(rows[0][policies[0]])
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for policy in policies:
        summary[policy] = {}
        for metric in metrics:
            values = np.asarray([row[policy][metric] for row in rows], dtype=float)
            summary[policy][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    return {
        "n_seeds": seeds,
        "steps": steps,
        "capacity": capacity,
        "summary": summary,
        "per_seed": rows,
    }


def print_summary(result: dict) -> None:
    print("\n=== DYN5: PARTIAL CUE -> DORMANT STATE REINSTATEMENT ===")
    print(
        f"seeds={result['n_seeds']} steps={result['steps']} "
        f"finite_H_capacity={result['capacity']} hidden_contexts=48"
    )
    keys = [
        "first1_recurrence",
        "first5_recurrence",
        "positions_2_to_5",
        "hard_positions_2_to_5",
        "late_recurrence",
        "exact_retrieval_debug",
    ]
    print("policy".ljust(22) + " ".join(k[:11].rjust(12) for k in keys))
    for policy, metrics in result["summary"].items():
        print(
            policy.ljust(22)
            + " ".join(f"{metrics[k]['mean']:12.4f}" for k in keys)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--capacity", type=int, default=24)
    args = ap.parse_args()
    result = run_many(args.seeds, args.steps, args.capacity)
    print_summary(result)


if __name__ == "__main__":
    main()
