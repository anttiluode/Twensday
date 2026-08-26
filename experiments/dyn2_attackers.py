from __future__ import annotations

import argparse
import numpy as np

from dyn2_circulating_memory import (
    RecurrentLoopBank,
    StructuralListener,
    make_world,
)


def make_fixed_scale_world(
    steps: int,
    seed: int,
    min_interval: int = 250,
    max_interval: int = 450,
):
    """Same sparse-cue world as DYN2 but every cue has unit scale."""
    rng = np.random.default_rng(seed)
    target = np.zeros(steps, dtype=int)
    cue = np.zeros(steps, dtype=float)
    sign = int(rng.choice([-1, 1]))
    t = 0
    while t < steps:
        interval = int(rng.integers(min_interval, max_interval + 1))
        sign *= -1
        end = min(steps, t + interval)
        target[t:end] = sign
        ncue = min(4, steps - t)
        cue[t : t + ncue] = sign + rng.normal(0.0, 0.08, ncue)
        t = end
    cue += rng.normal(0.0, 0.12, steps)
    return cue, target


def run_recurrent(
    cue: np.ndarray,
    target: np.ndarray,
    gate_mode: str,
    delay: int,
):
    bank = RecurrentLoopBank(gate_mode=gate_mode)
    listener = StructuralListener(bank.n, delay=delay)
    rewards: list[int] = []
    correct = np.zeros(len(cue), dtype=float)

    for t, (x, y) in enumerate(zip(cue, target)):
        broadcast = bank.step(float(x))
        action = listener.act(broadcast)
        reward = 1 if action == y else -1
        correct[t] = float(reward > 0)
        rewards.append(reward)
        if t >= delay:
            listener.reward(rewards[t - delay])
    return correct


def run_single_bistable(
    cue: np.ndarray,
    target: np.ndarray,
    recurrent_gain: float = 1.5,
    input_gain: float = 1.5,
):
    """Kill: one nonlinear self-state can implement a latch-like attractor."""
    q = 0.0
    correct = np.zeros(len(cue), dtype=float)
    for t, (x, y) in enumerate(zip(cue, target)):
        q = float(np.tanh(recurrent_gain * q + input_gain * x))
        action = 1 if q >= 0 else -1
        correct[t] = float(action == y)
    return correct


def run_seed(seed: int, steps: int = 9000, delay: int = 20) -> dict:
    burn = 500

    # Original DYN2 variable-amplitude world.
    cue, target, _, _, _ = make_world(steps, seed)
    single = run_single_bistable(cue, target)

    # Remove the nonstationarity that an output-gain controller ought to help.
    fixed_cue, fixed_target = make_fixed_scale_world(steps, seed)
    fixed_gate = run_recurrent(fixed_cue, fixed_target, "fixed", delay)
    rms = run_recurrent(fixed_cue, fixed_target, "rms_agc", delay)
    ais = run_recurrent(fixed_cue, fixed_target, "per_loop_homeo", delay)
    global_homeo = run_recurrent(
        fixed_cue,
        fixed_target,
        "global_homeostasis",
        delay,
    )

    return {
        "single_bistable_variable_scale": float(single[burn:].mean()),
        "fixed_scale_fixed_gate": float(fixed_gate[burn:].mean()),
        "fixed_scale_rms_agc": float(rms[burn:].mean()),
        "fixed_scale_ais": float(ais[burn:].mean()),
        "fixed_scale_global_homeostasis": float(global_homeo[burn:].mean()),
    }


def run_many(n_seeds: int, steps: int, delay: int) -> dict:
    rows = [run_seed(s, steps=steps, delay=delay) for s in range(n_seeds)]
    summary = {}
    for key in rows[0]:
        v = np.asarray([r[key] for r in rows], dtype=float)
        summary[key] = {"mean": float(v.mean()), "std": float(v.std())}
    return {
        "n_seeds": n_seeds,
        "steps": steps,
        "reward_delay": delay,
        "summary": summary,
        "development_note": (
            "Attacker parameters were explored before this committed run. "
            "Treat as development evidence, not a held-out confirmation."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--delay", type=int, default=20)
    args = ap.parse_args()
    out = run_many(args.seeds, args.steps, args.delay)

    print("\n=== DYN2 ATTACKERS ===")
    for key, s in out["summary"].items():
        print(f"{key:38s} {s['mean']:.4f} +/- {s['std']:.4f}")


if __name__ == "__main__":
    main()
