from __future__ import annotations

import argparse
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


def normalize_rows_with_reserve(v: np.ndarray, reserve: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = v.shape[-1]
    free = 1.0 - n * reserve
    if reserve < 0 or free < 0:
        raise ValueError("reserve incompatible with unit budget")
    excess = np.maximum(v - reserve, 0.0)
    denom = excess.sum(axis=-1, keepdims=True)
    return reserve + free * excess / (denom + 1e-15)


def make_world(
    steps: int,
    seed: int,
    n_sensors: int = 8,
    mode_len: int = 240,
    reliability_len: int = 1800,
):
    """Same hidden-dynamics world as DYN0.

    Current scalar values are intentionally weak evidence. The useful clue is
    temporal organization. Which physical channels deserve trust changes more
    slowly than the hidden mode itself.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((steps, n_sensors), dtype=float)
    mode = np.empty(steps, dtype=int)
    reliable = np.empty(steps, dtype=int)

    weak = np.array([0.30, -0.30, 0.00, 0.18, -0.18, 0.24, -0.24, 0.00])
    s = 1
    rel = 0
    next_mode = mode_len + int(rng.integers(-45, 46))
    next_rel = reliability_len + int(rng.integers(-250, 251))
    prev = np.zeros(n_sensors)

    for t in range(steps):
        if t == next_mode:
            s *= -1
            next_mode = min(steps, t + mode_len + int(rng.integers(-45, 46)))
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
        noise = np.sqrt(np.maximum(1.0 - coeff * coeff, 0.02))
        cur = coeff * prev + rng.normal(size=n_sensors) * noise
        x[t] = cur
        prev = cur

    return x, mode, reliable


class TwoLevelAllocator:
    """Shared delayed positive/conserved learning for local and global mass."""

    def __init__(
        self,
        n_sensors: int,
        n_local_features: int,
        delay: int,
        eta_global: float = 0.5,
        eta_local: float = 1.0,
        reserve_global: float = 0.005,
        reserve_local: float = 1e-5,
        learn_local: bool = True,
    ) -> None:
        self.n = n_sensors
        self.k = n_local_features
        self.delay = delay
        self.eta_global = eta_global
        self.eta_local = eta_local
        self.reserve_global = reserve_global
        self.reserve_local = reserve_local
        self.learn_local = learn_local
        self.local_mass = np.full((self.n, self.k), 1.0 / self.k)
        self.global_mass = np.full(self.n, 1.0 / self.n)
        self.history: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    def act(self, features: np.ndarray) -> tuple[int, np.ndarray]:
        local = np.tanh(2.0 * np.sum(self.local_mass * features, axis=1))
        score = float(self.global_mass @ local)
        action = 1 if score >= 0 else -1
        self.history.append(
            (action, local.copy(), features.copy(), self.global_mass.copy())
        )
        return action, local

    def reward(self, reward: int) -> None:
        if len(self.history) <= self.delay:
            return
        action, local, features, old_global = self.history[-self.delay - 1]

        global_fitness = reward * action * local
        global_impulse = np.maximum(self.global_mass * global_fitness, 0.0)
        self.global_mass = normalize_with_reserve(
            self.global_mass + self.eta_global * global_impulse,
            self.reserve_global,
        )

        if self.learn_local:
            local_fitness = reward * action * old_global[:, None] * features
            local_impulse = np.maximum(self.local_mass * local_fitness, 0.0)
            self.local_mass = normalize_rows_with_reserve(
                self.local_mass + self.eta_local * local_impulse,
                self.reserve_local,
            )


class ActiveTemporalBank:
    """Basket-inspired generic time-window + nonlinear compartment bank.

    This is not a biophysical basket-cell model. Raw scalar input is split into
    positive/negative drive and accumulated at several time constants. Each
    window is exposed through compressive, near-linear, and expansive local
    transfer shapes with several thresholds. No x(t)*x(t-1) coordinate exists.
    """

    def __init__(
        self,
        n_sensors: int = 8,
        taus=(1, 2, 4, 8, 16, 32),
        powers=(0.5, 1.0, 2.0),
        biases=(0.2, 0.5, 0.9, 1.4),
    ) -> None:
        self.n = n_sensors
        self.taus = np.asarray(taus, dtype=float)
        self.alpha = np.exp(-1.0 / self.taus)
        self.powers = np.asarray(powers, dtype=float)
        self.biases = np.asarray(biases, dtype=float)
        self.pos_state = np.zeros((self.n, len(self.taus)))
        self.neg_state = np.zeros_like(self.pos_state)
        self.base_k = len(self.taus) * len(self.powers) * len(self.biases)
        self.feature_k = 2 * self.base_k  # paired +/- fixed directions

    def step(self, x: np.ndarray) -> np.ndarray:
        pos = np.maximum(x, 0.0)[:, None]
        neg = np.maximum(-x, 0.0)[:, None]
        self.pos_state = (
            self.alpha[None, :] * self.pos_state
            + (1.0 - self.alpha)[None, :] * pos
        )
        self.neg_state = (
            self.alpha[None, :] * self.neg_state
            + (1.0 - self.alpha)[None, :] * neg
        )

        activity = (
            (self.pos_state[:, :, None] + 1e-6) ** self.powers[None, None, :]
            + (self.neg_state[:, :, None] + 1e-6) ** self.powers[None, None, :]
        )
        base = np.tanh(
            3.0
            * (
                activity[:, :, :, None]
                - self.biases[None, None, None, :]
            )
        ).reshape(self.n, -1)
        return np.concatenate([base, -base], axis=1)

    def power_mass(self, local_mass: np.ndarray) -> np.ndarray:
        # Pair +/- directions back together, then aggregate by transfer power.
        base = local_mass[:, : self.base_k] + local_mass[:, self.base_k :]
        arr = base.reshape(
            self.n,
            len(self.taus),
            len(self.powers),
            len(self.biases),
        )
        out = arr.sum(axis=(0, 1, 3))
        return out / out.sum()


class PassiveLeakyBank:
    """Kill: same family of time constants, but no rectification/active branch nonlinearity."""

    def __init__(self, n_sensors: int = 8, taus=(1, 2, 4, 8, 16, 32)) -> None:
        self.n = n_sensors
        self.alpha = np.exp(-1.0 / np.asarray(taus, dtype=float))
        self.state = np.zeros((self.n, len(self.alpha)))
        self.feature_k = 2 * len(self.alpha)

    def step(self, x: np.ndarray) -> np.ndarray:
        self.state = (
            self.alpha[None, :] * self.state
            + (1.0 - self.alpha)[None, :] * x[:, None]
        )
        return np.concatenate([self.state, -self.state], axis=1)


class RandomNonlinearBank:
    """Matched-size generic tanh reservoir attacker."""

    def __init__(self, n_sensors: int = 8, k: int = 72, seed: int = 999) -> None:
        rng = np.random.default_rng(seed)
        self.n = n_sensors
        self.k = k
        self.a = rng.uniform(-0.97, 0.97, k)
        self.b = rng.uniform(0.4, 1.6, k) * rng.choice([-1.0, 1.0], k)
        self.c = rng.uniform(-0.6, 0.6, k)
        self.state = np.zeros((self.n, k))
        self.feature_k = 2 * k

    def step(self, x: np.ndarray) -> np.ndarray:
        self.state = np.tanh(
            self.a[None, :] * self.state
            + self.b[None, :] * x[:, None]
            + self.c[None, :]
        )
        return np.concatenate([self.state, -self.state], axis=1)


class HandLagDetector:
    """Ceiling/control: DYN0's task-matched temporal coordinate supplied by hand."""

    def __init__(self, n_sensors: int = 8, beta: float = 0.85) -> None:
        self.prev = np.zeros(n_sensors)
        self.q = np.zeros(n_sensors)
        self.beta = beta

    def step(self, x: np.ndarray) -> np.ndarray:
        lag_product = x * self.prev
        self.q = self.beta * self.q + (1.0 - self.beta) * lag_product
        b = np.tanh(3.0 * self.q)
        self.prev = x.copy()
        # One local feature per sensor; global structural allocation still learns.
        return b[:, None]


def run_machine(
    bank,
    x: np.ndarray,
    mode: np.ndarray,
    delay: int,
    learn_local: bool = True,
    reserve_local: float = 1e-5,
):
    alloc = TwoLevelAllocator(
        n_sensors=x.shape[1],
        n_local_features=bank.feature_k if hasattr(bank, "feature_k") else 1,
        delay=delay,
        learn_local=learn_local,
        reserve_local=reserve_local,
    )
    correct = np.zeros(len(x), dtype=float)
    rewards: list[int] = []

    for t in range(len(x)):
        features = bank.step(x[t])
        action, _ = alloc.act(features)
        correct[t] = float(action == mode[t])
        rewards.append(1 if action == mode[t] else -1)
        if t >= delay:
            alloc.reward(rewards[t - delay])

    return correct, alloc


def run_hand_lag(x: np.ndarray, mode: np.ndarray, delay: int):
    bank = HandLagDetector(x.shape[1])
    # Its one feature is already signed evidence; local mass is irrelevant.
    correct, alloc = run_machine(
        bank,
        x,
        mode,
        delay,
        learn_local=False,
        reserve_local=0.0,
    )
    return correct, alloc


def run_seed(seed: int, steps: int = 9000, delay: int = 30) -> dict:
    x, mode, _ = make_world(steps, seed)
    burn = 500

    active_bank = ActiveTemporalBank()
    active, active_alloc = run_machine(active_bank, x, mode, delay)

    active_no_local, _ = run_machine(
        ActiveTemporalBank(),
        x,
        mode,
        delay,
        learn_local=False,
    )

    passive, _ = run_machine(
        PassiveLeakyBank(),
        x,
        mode,
        delay,
        reserve_local=1e-4,
    )

    random_bank = RandomNonlinearBank(k=active_bank.base_k)
    random_nl, _ = run_machine(random_bank, x, mode, delay)

    hand_lag, _ = run_hand_lag(x, mode, delay)
    power_mass = active_bank.power_mass(active_alloc.local_mass)

    return {
        "hand_lag": float(hand_lag[burn:].mean()),
        "basket_active": float(active[burn:].mean()),
        "random_nonlinear": float(random_nl[burn:].mean()),
        "active_no_local_learning": float(active_no_local[burn:].mean()),
        "passive_leaky": float(passive[burn:].mean()),
        "mass_compressive": float(power_mass[0]),
        "mass_linear": float(power_mass[1]),
        "mass_expansive": float(power_mass[2]),
    }


def run_many(n_seeds: int, steps: int, delay: int) -> dict:
    rows = [run_seed(s, steps=steps, delay=delay) for s in range(n_seeds)]
    summary = {}
    for key in rows[0]:
        values = np.asarray([r[key] for r in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return {
        "n_seeds": n_seeds,
        "steps": steps,
        "reward_delay": delay,
        "summary": summary,
        "per_seed": rows,
    }


def print_summary(result: dict) -> None:
    print("\n=== DYN1: ACTIVE TEMPORAL BANK ===")
    print(
        f"seeds={result['n_seeds']} steps={result['steps']} "
        f"delayed_reward={result['reward_delay']}"
    )
    for key in [
        "hand_lag",
        "basket_active",
        "random_nonlinear",
        "active_no_local_learning",
        "passive_leaky",
    ]:
        s = result["summary"][key]
        print(f"{key:28s} accuracy {s['mean']:.4f} +/- {s['std']:.4f}")
    print("\nfinal active-bank structural mass by crude transfer family")
    for key in ["mass_compressive", "mass_linear", "mass_expansive"]:
        s = result["summary"][key]
        print(f"{key:28s} {s['mean']:.4f} +/- {s['std']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--delay", type=int, default=30)
    args = ap.parse_args()
    result = run_many(args.seeds, args.steps, args.delay)
    print_summary(result)


if __name__ == "__main__":
    main()
