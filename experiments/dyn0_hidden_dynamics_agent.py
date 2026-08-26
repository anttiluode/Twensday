from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def normalize_with_reserve(v: np.ndarray, reserve: float) -> np.ndarray:
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
    n_sensors: int = 8,
    mode_len: int = 240,
    reliability_len: int = 1800,
):
    """Continuous world where mode is identifiable from dynamics, not snapshots.

    The hidden binary mode changes the sign of temporal correlation in four
    channels. Which physical channels carry useful versus misleading evidence
    changes on a slower clock. Every channel keeps roughly the same symmetric
    instantaneous marginal, so current x alone is intentionally weak evidence.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((steps, n_sensors), dtype=float)
    mode = np.empty(steps, dtype=int)
    reliable = np.empty(steps, dtype=int)

    weak = np.array([0.30, -0.30, 0.00, 0.18, -0.18, 0.24, -0.24, 0.00])
    if n_sensors != len(weak):
        weak = np.resize(weak, n_sensors)

    s = 1
    rel = 0
    next_mode = mode_len + int(rng.integers(-45, 46))
    next_rel = reliability_len + int(rng.integers(-250, 251))
    prev = np.zeros(n_sensors)

    for t in range(steps):
        if t == next_mode:
            s *= -1
            next_mode = min(
                steps,
                t + mode_len + int(rng.integers(-45, 46)),
            )
        if t == next_rel:
            rel = (rel + 1) % n_sensors
            next_rel = min(
                steps,
                t + reliability_len + int(rng.integers(-250, 251)),
            )

        mode[t] = s
        reliable[t] = rel
        coeff = weak.copy()
        coeff[rel] = 0.94 * s
        coeff[(rel + 1) % n_sensors] = 0.75 * s
        coeff[(rel + 2) % n_sensors] = -0.90 * s
        coeff[(rel + 3) % n_sensors] = -0.65 * s
        noise_scale = np.sqrt(np.maximum(1.0 - coeff * coeff, 0.02))
        cur = coeff * prev + rng.normal(size=n_sensors) * noise_scale
        x[t] = cur
        prev = cur

    return x, mode, reliable


@dataclass
class DynamicAllocator:
    """Fast temporal state + slow positive conserved structural allocation."""

    n: int = 8
    beta: float = 0.85
    reserve: float = 0.005
    eta: float = 1.0
    delay: int = 30
    fast_state: bool = True
    learn: bool = True
    scramble_addresses: bool = False
    shuffle_credit: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        self.prev = np.zeros(self.n)
        self.q = np.zeros(self.n)
        self.mass = np.full(self.n, 1.0 / self.n)
        self.history: list[tuple[int, np.ndarray]] = []
        self.rng = np.random.default_rng(self.seed)

    def observe_and_act(self, x: np.ndarray):
        # Each sensor-point turns recent temporal relation into current state.
        lag_product = x * self.prev
        if self.fast_state:
            self.q = self.beta * self.q + (1.0 - self.beta) * lag_product
            broadcast = np.tanh(3.0 * self.q)
        else:
            # Lesion: only one instantaneous lag relation survives.
            broadcast = np.tanh(3.0 * lag_product)

        if self.scramble_addresses:
            # Same broadcasts exist, but structural address changes every step.
            broadcast = broadcast[self.rng.permutation(self.n)]

        score = float(self.mass @ broadcast)
        action = 1 if score >= 0 else -1
        self.history.append((action, broadcast.copy()))
        self.prev = x.copy()
        return action, broadcast, self.mass.copy()

    def deliver_reward(self, reward: int) -> None:
        if not self.learn or len(self.history) <= self.delay:
            return

        if self.shuffle_credit:
            hi = max(1, len(self.history) - self.delay)
            action, broadcast = self.history[int(self.rng.integers(0, hi))]
        else:
            # Eligibility: attach delayed consequence to the activity that
            # produced the action, not the activity present when reward arrives.
            action, broadcast = self.history[-self.delay - 1]

        # If action was correct, reinforce local broadcasts that supported it.
        # If action was wrong, reinforce local broadcasts that opposed it.
        # There is no explicit negative structural command; conservation makes
        # competitors shrink when another claim grows.
        local_fitness = reward * action * broadcast
        impulse = np.maximum(self.mass * local_fitness, 0.0)
        self.mass = normalize_with_reserve(
            self.mass + self.eta * impulse,
            self.reserve,
        )


@dataclass
class FixedShare:
    """Mature tracking-experts attacker with full-information delayed updates."""

    n: int = 8
    eta: float = 0.8
    share: float = 0.02
    delay: int = 30

    def __post_init__(self) -> None:
        self.w = np.full(self.n, 1.0 / self.n)
        self.history: list[tuple[int, np.ndarray]] = []

    def observe_and_act(self, broadcast: np.ndarray) -> int:
        action = 1 if float(self.w @ broadcast) >= 0 else -1
        self.history.append((action, broadcast.copy()))
        return action

    def deliver_reward(self, reward: int) -> None:
        if len(self.history) <= self.delay:
            return
        action, broadcast = self.history[-self.delay - 1]
        target = action if reward > 0 else -action
        expert_action = np.where(broadcast >= 0, 1, -1)
        loss = (expert_action != target).astype(float)
        w = self.w * np.exp(-self.eta * loss)
        w /= w.sum() + 1e-15
        self.w = (1.0 - self.share) * w + self.share / self.n


@dataclass
class SignedLearner:
    """Unconstrained central signed-update attacker on the same broadcasts."""

    n: int = 8
    lr: float = 0.2
    delay: int = 30

    def __post_init__(self) -> None:
        self.w = np.zeros(self.n)
        self.history: list[tuple[int, np.ndarray, float]] = []

    def observe_and_act(self, broadcast: np.ndarray) -> int:
        y = float(np.tanh(self.w @ broadcast))
        action = 1 if y >= 0 else -1
        self.history.append((action, broadcast.copy(), y))
        return action

    def deliver_reward(self, reward: int) -> None:
        if len(self.history) <= self.delay:
            return
        action, broadcast, y = self.history[-self.delay - 1]
        target = action if reward > 0 else -action
        self.w += self.lr * (target - y) * broadcast
        norm = np.linalg.norm(self.w)
        if norm > 5.0:
            self.w *= 5.0 / norm


def run_seed(seed: int, steps: int = 9000, delay: int = 30) -> dict:
    x, mode, reliable = make_world(steps, seed)

    tw = DynamicAllocator(delay=delay, seed=seed)
    lag1 = DynamicAllocator(delay=delay, fast_state=False, seed=seed)
    scrambled = DynamicAllocator(
        delay=delay,
        scramble_addresses=True,
        seed=seed,
    )
    shuffled_credit = DynamicAllocator(
        delay=delay,
        shuffle_credit=True,
        seed=seed,
    )
    fixed_share = FixedShare(delay=delay)
    signed = SignedLearner(delay=delay)

    methods = [
        "twensday",
        "lag1_only",
        "uniform_dynamic",
        "scrambled_addresses",
        "shuffled_credit",
        "fixed_share",
        "signed_global",
        "stateless",
        "oracle_sensor",
    ]
    correct = {k: np.zeros(steps, dtype=float) for k in methods}
    reward_q = {
        k: []
        for k in ["tw", "lag1", "scrambled", "shuffled", "fixed", "signed"]
    }

    # Snapshot attacker: current sensor values only. Delayed reward reveals the
    # binary target retrospectively, but the instantaneous marginal was designed
    # to carry almost no mode information.
    stateless_q: list[tuple[int, int, np.ndarray, float]] = []
    stateless_w = np.zeros(x.shape[1])
    stateless_lr = 0.01
    useful_mass = np.zeros(steps)

    for t in range(steps):
        a_tw, broadcast, mass = tw.observe_and_act(x[t])
        a_lag1, _, _ = lag1.observe_and_act(x[t])
        a_scrambled, _, _ = scrambled.observe_and_act(x[t])
        a_shuffled, _, _ = shuffled_credit.observe_and_act(x[t])

        a_uniform = 1 if float(np.mean(broadcast)) >= 0 else -1
        a_fixed = fixed_share.observe_and_act(broadcast)
        a_signed = signed.observe_and_act(broadcast)

        stateless_y = float(np.tanh(stateless_w @ x[t]))
        a_stateless = 1 if stateless_y >= 0 else -1
        a_oracle = 1 if broadcast[reliable[t]] >= 0 else -1

        actions = {
            "twensday": a_tw,
            "lag1_only": a_lag1,
            "uniform_dynamic": a_uniform,
            "scrambled_addresses": a_scrambled,
            "shuffled_credit": a_shuffled,
            "fixed_share": a_fixed,
            "signed_global": a_signed,
            "stateless": a_stateless,
            "oracle_sensor": a_oracle,
        }
        for name, action in actions.items():
            correct[name][t] = float(action == mode[t])

        reward_q["tw"].append(1 if a_tw == mode[t] else -1)
        reward_q["lag1"].append(1 if a_lag1 == mode[t] else -1)
        reward_q["scrambled"].append(1 if a_scrambled == mode[t] else -1)
        reward_q["shuffled"].append(1 if a_shuffled == mode[t] else -1)
        reward_q["fixed"].append(1 if a_fixed == mode[t] else -1)
        reward_q["signed"].append(1 if a_signed == mode[t] else -1)
        stateless_q.append(
            (
                1 if a_stateless == mode[t] else -1,
                a_stateless,
                x[t].copy(),
                stateless_y,
            )
        )

        # There are two currently useful channels: strong and weaker positive
        # evidence. The structural learner is not told these indices.
        good = [reliable[t], (reliable[t] + 1) % x.shape[1]]
        useful_mass[t] = float(np.sum(mass[good]))

        if t >= delay:
            tw.deliver_reward(reward_q["tw"][t - delay])
            lag1.deliver_reward(reward_q["lag1"][t - delay])
            scrambled.deliver_reward(reward_q["scrambled"][t - delay])
            shuffled_credit.deliver_reward(reward_q["shuffled"][t - delay])
            fixed_share.deliver_reward(reward_q["fixed"][t - delay])
            signed.deliver_reward(reward_q["signed"][t - delay])

            rw, old_action, old_x, old_y = stateless_q[t - delay]
            old_target = old_action if rw > 0 else -old_action
            stateless_w += stateless_lr * (old_target - old_y) * old_x

    burn = 500
    result = {
        name: float(np.mean(correct[name][burn:]))
        for name in methods
    }
    result["useful_structural_mass"] = float(np.mean(useful_mass[burn:]))

    # How badly do the adaptive methods stumble when the identity of the useful
    # physical channel changes?
    switches = np.flatnonzero(reliable[1:] != reliable[:-1]) + 1
    for name in ["twensday", "fixed_share"]:
        first100: list[float] = []
        first400: list[float] = []
        for s in switches:
            first100.extend(correct[name][s:min(s + 100, steps)])
            first400.extend(correct[name][s:min(s + 400, steps)])
        result[name + "_post_reliability_switch_100"] = (
            float(np.mean(first100)) if first100 else float("nan")
        )
        result[name + "_post_reliability_switch_400"] = (
            float(np.mean(first400)) if first400 else float("nan")
        )

    return result


def run_many(
    n_seeds: int = 12,
    steps: int = 9000,
    delay: int = 30,
) -> dict:
    rows = [
        run_seed(seed, steps=steps, delay=delay)
        for seed in range(n_seeds)
    ]
    summary = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    return {
        "n_seeds": n_seeds,
        "steps": steps,
        "reward_delay": delay,
        "summary": summary,
        "per_seed": rows,
    }


def print_summary(result: dict) -> None:
    print("\n=== DYN0: HIDDEN-DYNAMICS CONTROLLER SELECTION ===")
    print(
        f"seeds={result['n_seeds']} steps={result['steps']} "
        f"delayed_reward={result['reward_delay']} steps"
    )
    order = [
        "oracle_sensor",
        "fixed_share",
        "signed_global",
        "twensday",
        "lag1_only",
        "shuffled_credit",
        "stateless",
        "uniform_dynamic",
        "scrambled_addresses",
    ]
    for key in order:
        s = result["summary"][key]
        print(
            f"{key:24s} accuracy {s['mean']:.4f} +/- {s['std']:.4f}"
        )

    s = result["summary"]["useful_structural_mass"]
    print(
        f"useful structural mass   {s['mean']:.4f} +/- {s['std']:.4f}"
    )
    for key in [
        "twensday_post_reliability_switch_100",
        "twensday_post_reliability_switch_400",
        "fixed_share_post_reliability_switch_100",
        "fixed_share_post_reliability_switch_400",
    ]:
        s = result["summary"][key]
        print(f"{key:44s} {s['mean']:.4f} +/- {s['std']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--delay", type=int, default=30)
    ap.add_argument("--write", type=Path, default=None)
    args = ap.parse_args()

    result = run_many(args.seeds, args.steps, args.delay)
    print_summary(result)

    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
