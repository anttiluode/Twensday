from __future__ import annotations

import argparse
import numpy as np

from dyn5_partial_cue_reinstatement import SlowModel, make_world, run_policy


class ResidualStateBank:
    """A finite bag of dormant scalar states with no cue keys at all."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.state: list[float] = []
        self.priority: list[float] = []
        self.age: list[int] = []

    def write(self, state: float, utility: float, now: int) -> None:
        if len(self.state) < self.capacity:
            self.state.append(float(state))
            self.priority.append(float(utility))
            self.age.append(now)
            return
        aged = np.asarray(self.priority) * np.exp(
            -(now - np.asarray(self.age)) / 6000.0
        )
        i = int(np.argmin(aged))
        if float(utility) <= float(aged[i]):
            return
        self.state[i] = float(state)
        self.priority[i] = float(utility)
        self.age[i] = now


def summarize(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = list(values[0])
    return {
        k: {
            "mean": float(np.mean([v[k] for v in values])),
            "std": float(np.std([v[k] for v in values])),
        }
        for k in keys
    }


def run_no_memory_fast_adapt(
    seed: int,
    eta_q: float,
    steps: int,
    block_len: int = 25,
) -> dict[str, float]:
    z, target, hidden_context, offsets, _, _ = make_world(
        steps=steps,
        seed=seed,
        block_len=block_len,
    )
    model = SlowModel(z.shape[1])
    n_contexts = len(offsets)
    seen = np.zeros(n_contexts, dtype=int)
    q = 0.0

    first1: list[float] = []
    first5: list[float] = []
    p2to5: list[float] = []
    hard: list[float] = []
    late: list[float] = []

    for t in range(steps):
        pos = t % block_len
        c = int(hidden_context[t])
        if pos == 0:
            recur = seen[c] > 0
            seen[c] += 1
            q = 0.0

        slow = model.score(z[t])
        score = slow + q
        yhat = np.tanh(score)
        pred = 1 if score >= 0 else -1
        ok = float(pred == target[t])

        if recur:
            if pos == 0:
                first1.append(ok)
            if pos < 5:
                first5.append(ok)
            if 1 <= pos < 5:
                p2to5.append(ok)
                if abs(float(offsets[c])) >= 0.9:
                    hard.append(ok)
            if pos >= block_len - 8:
                late.append(ok)

        y = int(target[t])
        q += eta_q * (y - yhat)
        q = float(np.clip(q, -1.5, 1.5))
        model.update(z[t], y, q)

    return {
        "first1_recurrence": float(np.mean(first1)),
        "first5_recurrence": float(np.mean(first5)),
        "positions_2_to_5": float(np.mean(p2to5)),
        "hard_positions_2_to_5": float(np.mean(hard)),
        "late_recurrence": float(np.mean(late)),
    }


def run_outcome_candidates(
    seed: int,
    mode: str,
    steps: int,
    capacity: int,
    block_len: int = 25,
    cue_steps: int = 5,
) -> dict[str, float]:
    z, target, hidden_context, offsets, _, _ = make_world(
        steps=steps,
        seed=seed,
        block_len=block_len,
        cue_steps=cue_steps,
    )
    model = SlowModel(z.shape[1])
    n_contexts = len(offsets)
    seen = np.zeros(n_contexts, dtype=int)

    if mode == "stored_state_bag":
        bank = ResidualStateBank(capacity)
        fixed = None
    elif mode == "fixed_state_grid":
        bank = None
        fixed = np.asarray(
            [-1.20, -0.90, -0.60, -0.30, 0.0, 0.30, 0.60, 0.90, 1.20],
            dtype=float,
        )
    else:
        raise ValueError(mode)

    q = 0.0
    eta_q = 0.06
    predictive_beta = 5.0
    current_context: int | None = None
    recur = False
    base_loss: list[float] = []
    fast_loss: list[float] = []
    log_weight: np.ndarray | None = None

    first1: list[float] = []
    first5: list[float] = []
    p2to5: list[float] = []
    hard: list[float] = []
    late: list[float] = []

    for t in range(steps):
        pos = t % block_len
        c = int(hidden_context[t])

        if pos == 0:
            if current_context is not None and bank is not None:
                utility = max(
                    float(np.mean(base_loss) - np.mean(fast_loss)),
                    0.0,
                )
                bank.write(q, utility, t)

            current_context = c
            recur = seen[c] > 0
            seen[c] += 1
            q = 0.0
            base_loss = []
            fast_loss = []
            candidates = (
                np.asarray(bank.state, dtype=float)
                if bank is not None
                else fixed.copy()
            )
            log_weight = np.zeros(len(candidates), dtype=float)

        candidates = (
            np.asarray(bank.state, dtype=float)
            if bank is not None
            else fixed
        )
        if log_weight is None or len(log_weight) != len(candidates):
            log_weight = np.zeros(len(candidates), dtype=float)

        # There is deliberately NO cue. Only consequences from earlier events in
        # this block may select among candidate dormant states.
        if 0 < pos < cue_steps and len(candidates) > 0:
            logits = log_weight - float(np.max(log_weight))
            weight = np.exp(logits)
            weight /= float(np.sum(weight)) + 1e-12
            retrieved = float(weight @ candidates)
            q = 0.25 * q + 0.75 * retrieved

        slow = model.score(z[t])
        score = slow + q
        yhat = np.tanh(score)
        pred = 1 if score >= 0 else -1
        ok = float(pred == target[t])

        if recur:
            if pos == 0:
                first1.append(ok)
            if pos < 5:
                first5.append(ok)
            if 1 <= pos < 5:
                p2to5.append(ok)
                if abs(float(offsets[c])) >= 0.9:
                    hard.append(ok)
            if pos >= block_len - 8:
                late.append(ok)

        y = int(target[t])
        base_yhat = np.tanh(slow)
        base_loss.append(float((y - base_yhat) ** 2))
        fast_loss.append(float((y - yhat) ** 2))

        if pos < cue_steps and len(candidates) > 0:
            candidate_yhat = np.tanh(slow + candidates)
            log_weight += -predictive_beta * (y - candidate_yhat) ** 2

        q += eta_q * (y - yhat)
        q = float(np.clip(q, -1.5, 1.5))
        model.update(z[t], y, q)

    return {
        "first1_recurrence": float(np.mean(first1)),
        "first5_recurrence": float(np.mean(first5)),
        "positions_2_to_5": float(np.mean(p2to5)),
        "hard_positions_2_to_5": float(np.mean(hard)),
        "late_recurrence": float(np.mean(late)),
    }


def run_many(seeds: int, steps: int, capacity: int) -> dict:
    methods: dict[str, list[dict[str, float]]] = {
        "confirm_residual": [],
        "exact_id_table": [],
        "stored_state_bag": [],
        "fixed_state_grid": [],
        "adapt_eta_006": [],
        "adapt_eta_015": [],
        "adapt_eta_030": [],
        "adapt_eta_060": [],
    }

    for seed in range(seeds):
        methods["confirm_residual"].append(
            {
                k: v
                for k, v in run_policy(
                    seed,
                    "confirm_residual",
                    steps=steps,
                    capacity=capacity,
                ).items()
                if k != "exact_retrieval_debug"
            }
        )
        methods["exact_id_table"].append(
            {
                k: v
                for k, v in run_policy(
                    seed,
                    "exact_id_table",
                    steps=steps,
                    capacity=capacity,
                ).items()
                if k != "exact_retrieval_debug"
            }
        )
        methods["stored_state_bag"].append(
            run_outcome_candidates(seed, "stored_state_bag", steps, capacity)
        )
        methods["fixed_state_grid"].append(
            run_outcome_candidates(seed, "fixed_state_grid", steps, capacity)
        )
        for eta, name in [
            (0.06, "adapt_eta_006"),
            (0.15, "adapt_eta_015"),
            (0.30, "adapt_eta_030"),
            (0.60, "adapt_eta_060"),
        ]:
            methods[name].append(run_no_memory_fast_adapt(seed, eta, steps))

    return {
        "n_seeds": seeds,
        "steps": steps,
        "capacity": capacity,
        "summary": {name: summarize(rows) for name, rows in methods.items()},
    }


def print_summary(result: dict) -> None:
    print("\n=== DYN5B: DOES ASSOCIATIVE MEMORY EARN ITSELF? ===")
    print(
        f"seeds={result['n_seeds']} steps={result['steps']} "
        f"finite_H_capacity={result['capacity']}"
    )
    keys = [
        "first1_recurrence",
        "first5_recurrence",
        "positions_2_to_5",
        "hard_positions_2_to_5",
        "late_recurrence",
    ]
    print("method".ljust(22) + " ".join(k[:11].rjust(12) for k in keys))
    for name, metrics in result["summary"].items():
        print(
            name.ljust(22)
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
