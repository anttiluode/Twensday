from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def normalize_with_reserve(v: np.ndarray, reserve: float = 0.002) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    free = 1.0 - len(v) * reserve
    if reserve < 0 or free < 0:
        raise ValueError("reserve incompatible with unit budget")
    excess = np.maximum(v - reserve, 0.0)
    if excess.sum() < 1e-15:
        return np.full(len(v), 1.0 / len(v))
    return reserve + free * excess / excess.sum()


def make_world(
    steps: int,
    seed: int,
    min_interval: int = 250,
    max_interval: int = 450,
):
    """Continuous sparse-cue memory world.

    A brief noisy cue announces a hidden +/- state. The state remains correct for
    hundreds of steps, but the cue disappears after four steps. Cue amplitude
    changes between regimes, so a useful memory mechanism must both persist and
    reacquire after differently scaled events.
    """
    rng = np.random.default_rng(seed)
    target = np.zeros(steps, dtype=int)
    cue = np.zeros(steps, dtype=float)
    cue_mask = np.zeros(steps, dtype=bool)
    scale_trace = np.zeros(steps, dtype=float)

    sign = int(rng.choice([-1, 1]))
    t = 0
    switches: list[int] = []
    while t < steps:
        interval = int(rng.integers(min_interval, max_interval + 1))
        sign *= -1
        scale = float(np.exp(rng.uniform(np.log(0.35), np.log(2.5))))
        end = min(steps, t + interval)
        target[t:end] = sign
        scale_trace[t:end] = scale

        ncue = min(4, steps - t)
        cue[t : t + ncue] = sign * scale + rng.normal(0.0, 0.08, ncue)
        cue_mask[t : t + ncue] = True
        switches.append(t)
        t = end

    # Ongoing noise means a matched digital latch still has to decide whether a
    # large excursion is a cue rather than being told a cue flag.
    cue += rng.normal(0.0, 0.12, steps)
    return cue, target, cue_mask, scale_trace, np.asarray(switches, dtype=int)


class RecurrentLoopBank:
    """Bank of two-point recurrent motifs.

    Each motif contains two fast states A and B. Neither has a long local time
    constant. Memory longer than a few steps must therefore live in A->B->A
    circulation. The motif output is an emission from B, not B itself.

    gate_mode:
      fixed             : no output-excitability adaptation
      global_homeostasis: one scalar gain shared by all motifs
      per_loop_homeo    : independent slow gain per motif (AIS-like abstraction)
      rms_agc           : ordinary per-motif RMS/EMA AGC attacker
    """

    def __init__(
        self,
        gains=(0.35, 0.50, 0.65, 0.75, 0.85, 0.92, 0.98, 1.05, 1.15),
        leak: float = 0.15,
        input_gain: float = 0.90,
        gate_mode: str = "per_loop_homeo",
        target_activity: float = 0.65,
        gate_eta: float = 0.002,
    ) -> None:
        self.gains = np.asarray(gains, dtype=float)
        self.n = len(self.gains)
        self.leak = leak
        self.input_gain = input_gain
        self.gate_mode = gate_mode
        self.target_activity = target_activity
        self.gate_eta = gate_eta

        self.a = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.gate = np.ones(self.n)
        self.ema_activity = np.full(self.n, target_activity)

    def _current_gain(self) -> np.ndarray:
        if self.gate_mode == "global_homeostasis":
            return np.full(self.n, self.gate[0])
        if self.gate_mode == "rms_agc":
            return np.clip(
                self.target_activity / (self.ema_activity + 1e-4),
                0.2,
                5.0,
            )
        return self.gate

    def step(self, x: float, cut_return: bool = False) -> np.ndarray:
        gain = self._current_gain()
        emit_a = np.tanh(gain * self.a)
        emit_b = np.tanh(gain * self.b)

        # State update. A receives external traffic and B's return. B receives A.
        # With return cut, both local states have only leak=0.15 and cannot hold
        # a hundreds-step memory.
        ret = 0.0 if cut_return else 1.0
        next_a = (
            self.leak * self.a
            + ret * self.gains * emit_b
            + self.input_gain * x
        )
        next_b = self.leak * self.b + self.gains * emit_a
        self.a = next_a
        self.b = next_b

        activity = 0.5 * (np.abs(emit_a) + np.abs(emit_b))
        if self.gate_mode == "per_loop_homeo":
            self.gate *= np.exp(
                self.gate_eta * (self.target_activity - activity)
            )
            self.gate = np.clip(self.gate, 0.2, 5.0)
        elif self.gate_mode == "global_homeostasis":
            self.gate[0] *= math.exp(
                self.gate_eta
                * (self.target_activity - float(activity.mean()))
            )
            self.gate[0] = float(np.clip(self.gate[0], 0.2, 5.0))
        elif self.gate_mode == "rms_agc":
            self.ema_activity = 0.99 * self.ema_activity + 0.01 * activity

        # Important separation: internal state B can remain nonzero while its
        # emitted traffic is changed by the output gate.
        return np.tanh(self._current_gain() * self.b)


class StructuralListener:
    """Delayed positive/conserved allocation over stable recurrent addresses."""

    def __init__(
        self,
        n: int,
        delay: int = 20,
        eta: float = 1.5,
        reserve: float = 0.002,
        learn: bool = True,
    ) -> None:
        self.n = n
        self.delay = delay
        self.eta = eta
        self.reserve = reserve
        self.learn = learn
        self.mass = np.full(n, 1.0 / n)
        self.history: list[tuple[int, np.ndarray]] = []

    def act(self, broadcast: np.ndarray) -> int:
        action = 1 if float(self.mass @ broadcast) >= 0 else -1
        self.history.append((action, broadcast.copy()))
        return action

    def reward(self, reward: int) -> None:
        if not self.learn or len(self.history) <= self.delay:
            return
        action, broadcast = self.history[-self.delay - 1]
        fitness = reward * action * broadcast
        impulse = np.maximum(self.mass * fitness, 0.0)
        self.mass = normalize_with_reserve(
            self.mass + self.eta * impulse,
            self.reserve,
        )


def run_recurrent(
    cue: np.ndarray,
    target: np.ndarray,
    gate_mode: str,
    delay: int,
    learn_structure: bool = True,
    cut_return_from: int | None = None,
):
    bank = RecurrentLoopBank(gate_mode=gate_mode)
    listener = StructuralListener(
        bank.n,
        delay=delay,
        learn=learn_structure,
    )
    correct = np.zeros(len(cue), dtype=float)
    rewards: list[int] = []

    for t, (x, y) in enumerate(zip(cue, target)):
        broadcast = bank.step(
            float(x),
            cut_return=(cut_return_from is not None and t >= cut_return_from),
        )
        action = listener.act(broadcast)
        reward = 1 if action == y else -1
        correct[t] = float(reward > 0)
        rewards.append(reward)
        if t >= delay:
            listener.reward(rewards[t - delay])

    return correct, listener.mass.copy(), bank.gate.copy()


def run_long_leak(cue: np.ndarray, target: np.ndarray, alpha: float = 0.995):
    q = 0.0
    correct = np.zeros(len(cue), dtype=float)
    for t, (x, y) in enumerate(zip(cue, target)):
        q = alpha * q + float(x)
        action = 1 if q >= 0 else -1
        correct[t] = float(action == y)
    return correct


def run_digital_latch(
    cue: np.ndarray,
    target: np.ndarray,
    smooth: float = 0.60,
    threshold: float = 0.25,
):
    """Matched boring attacker: smooth cue, update sign only on large events."""
    state = 1
    filt = 0.0
    correct = np.zeros(len(cue), dtype=float)
    for t, (x, y) in enumerate(zip(cue, target)):
        filt = smooth * filt + (1.0 - smooth) * float(x)
        if abs(filt) > threshold:
            state = 1 if filt >= 0 else -1
        correct[t] = float(state == y)
    return correct


def run_seed(seed: int, steps: int = 9000, delay: int = 20) -> dict:
    cue, target, _, _, _ = make_world(steps, seed)
    burn = 500

    ais, mass, gate = run_recurrent(
        cue,
        target,
        gate_mode="per_loop_homeo",
        delay=delay,
    )
    fixed, _, _ = run_recurrent(
        cue,
        target,
        gate_mode="fixed",
        delay=delay,
    )
    rms, _, _ = run_recurrent(
        cue,
        target,
        gate_mode="rms_agc",
        delay=delay,
    )
    global_homeo, _, _ = run_recurrent(
        cue,
        target,
        gate_mode="global_homeostasis",
        delay=delay,
    )
    no_structure, _, _ = run_recurrent(
        cue,
        target,
        gate_mode="per_loop_homeo",
        delay=delay,
        learn_structure=False,
    )
    no_return, _, _ = run_recurrent(
        cue,
        target,
        gate_mode="per_loop_homeo",
        delay=delay,
        cut_return_from=0,
    )
    long_leak = run_long_leak(cue, target)
    latch = run_digital_latch(cue, target)

    # Lesion after the recurrent machine has already been operating for a long
    # time. Do not reset state or structural mass; simply sever B -> A traffic.
    cut_t = min(5000, steps // 2 + 500)
    cut_trace, _, _ = run_recurrent(
        cue,
        target,
        gate_mode="per_loop_homeo",
        delay=delay,
        cut_return_from=cut_t,
    )
    post_cut_start = min(steps, cut_t + 200)

    out = {
        "digital_latch": float(latch[burn:].mean()),
        "dyn2_ais": float(ais[burn:].mean()),
        "rms_agc": float(rms[burn:].mean()),
        "fixed_gate": float(fixed[burn:].mean()),
        "global_homeostasis": float(global_homeo[burn:].mean()),
        "no_structural_learning": float(no_structure[burn:].mean()),
        "long_local_leak": float(long_leak[burn:].mean()),
        "no_return": float(no_return[burn:].mean()),
        "pre_cut": float(cut_trace[burn:cut_t].mean()),
        "post_cut": float(cut_trace[post_cut_start:].mean()),
        "mean_final_gate": float(np.mean(gate)),
        "max_final_mass": float(np.max(mass)),
    }
    return out


def run_many(n_seeds: int, steps: int, delay: int) -> dict:
    rows = [run_seed(s, steps=steps, delay=delay) for s in range(n_seeds)]
    summary = {}
    for key in rows[0]:
        values = np.asarray([r[key] for r in rows], dtype=float)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return {
        "n_seeds": n_seeds,
        "steps": steps,
        "reward_delay": delay,
        "summary": summary,
        "per_seed": rows,
        "development_note": (
            "Task and hyperparameters were explored while constructing DYN2. "
            "This is a development receipt, not confirmatory evidence."
        ),
    }


def print_summary(result: dict) -> None:
    print("\n=== DYN2: CIRCULATING MEMORY + OUTPUT GATE ===")
    print(
        f"seeds={result['n_seeds']} steps={result['steps']} "
        f"delayed_reward={result['reward_delay']}"
    )
    order = [
        "digital_latch",
        "dyn2_ais",
        "rms_agc",
        "fixed_gate",
        "global_homeostasis",
        "no_structural_learning",
        "long_local_leak",
        "no_return",
    ]
    for key in order:
        s = result["summary"][key]
        print(f"{key:28s} accuracy {s['mean']:.4f} +/- {s['std']:.4f}")

    print("\nreturn-path lesion after learning")
    for key in ["pre_cut", "post_cut"]:
        s = result["summary"][key]
        print(f"{key:28s} accuracy {s['mean']:.4f} +/- {s['std']:.4f}")

    for key in ["mean_final_gate", "max_final_mass"]:
        s = result["summary"][key]
        print(f"{key:28s} {s['mean']:.4f} +/- {s['std']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--delay", type=int, default=20)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    result = run_many(args.seeds, args.steps, args.delay)
    print_summary(result)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
