from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import numpy as np

from dyn8_evolving_hypotheses import (
    HypothesisBirthWorld,
    exact_information_gain,
    softmax,
    support_prediction,
)


@dataclass
class Episode:
    action: int
    y: float
    sigma: float
    priority: float
    contaminated: bool


class WrongModelState:
    """The explanations that exist while the old evidence is arriving."""

    def __init__(self, world: HypothesisBirthWorld, seed: int, slots: int, order: int):
        rng = np.random.default_rng(seed + 31_337)
        features = world.contribution.shape[1]
        self.models: list[tuple[int, ...]] = []
        while len(self.models) < slots:
            support = tuple(sorted(int(v) for v in rng.choice(features, size=order, replace=False)))
            if support != world.true_support and support not in self.models:
                self.models.append(support)
        self.pred = np.stack(
            [support_prediction(world.contribution, s) for s in self.models], axis=0
        )
        self.logw = np.zeros(slots, dtype=float)
        self.persistent = np.zeros(len(world.x), dtype=float)
        self.action_seen = np.zeros(len(world.x), dtype=int)

    def before_observation(self, action: int, y: float, sigma: float) -> tuple[float, float]:
        posterior = softmax(self.logw)
        ensemble = float(posterior @ self.pred[:, action])
        z2 = ((y - ensemble) / sigma) ** 2
        excess = max(z2 - 1.0, 0.0)
        self.persistent[action] = 0.82 * self.persistent[action] + 0.18 * excess
        self.action_seen[action] += 1
        return excess, float(self.persistent[action])

    def update(self, action: int, y: float, sigma: float) -> None:
        self.logw += -0.5 * ((y - self.pred[:, action]) / sigma) ** 2


class MemoryPolicy:
    def __init__(
        self,
        name: str,
        capacity: int,
        world: HypothesisBirthWorld,
        seed: int,
    ) -> None:
        self.name = name
        self.capacity = capacity
        self.world = world
        self.rng = np.random.default_rng(seed + 91_771)
        self.items: list[Episode] = []
        self.seen = 0

    def _append_or_replace_lowest(self, episode: Episode, score: float | None = None) -> None:
        if len(self.items) < self.capacity:
            self.items.append(episode)
            return
        weakest = min(range(len(self.items)), key=lambda i: self.items[i].priority)
        candidate = episode.priority if score is None else score
        if candidate > self.items[weakest].priority:
            episode.priority = candidate
            self.items[weakest] = episode

    def write(
        self,
        *,
        action: int,
        y: float,
        sigma: float,
        raw_excess: float,
        persistent_excess: float,
        contaminated: bool,
    ) -> None:
        self.seen += 1
        ep = Episode(action, y, sigma, 0.0, contaminated)

        if self.name == "full_history":
            ep.priority = 1.0
            self.items.append(ep)
            return

        if self.name == "reservoir":
            if len(self.items) < self.capacity:
                self.items.append(ep)
            else:
                j = int(self.rng.integers(self.seen))
                if j < self.capacity:
                    self.items[j] = ep
            return

        if self.name == "recent":
            self.items.append(ep)
            if len(self.items) > self.capacity:
                self.items.pop(0)
            return

        if self.name == "leverage":
            # Grammar-aware but hypothesis-agnostic: retain measurements with
            # high total primitive energy relative to instrument noise.
            row = self.world.contribution[action]
            value = float(np.sum(row * row) / (sigma * sigma))
            # Weighted-reservoir key; does not look at y.
            ep.priority = float(self.rng.random() ** (1.0 / max(value, 1e-9)))
            self._append_or_replace_lowest(ep)
            return

        if self.name == "raw_residual":
            ep.priority = raw_excess
            self._append_or_replace_lowest(ep)
            return

        if self.name == "stubborn_diverse":
            # Persistent mismatch gets value; repeated storage from one action
            # is discounted so H cannot spend every slot on one stubborn site.
            same = sum(int(item.action == action) for item in self.items)
            score = persistent_excess / np.sqrt(1.0 + same)
            ep.priority = float(score)
            if len(self.items) < self.capacity:
                self.items.append(ep)
                return

            # Recompute existing utilities under current multiplicities.
            counts: dict[int, int] = {}
            for item in self.items:
                counts[item.action] = counts.get(item.action, 0) + 1
            utility = [
                item.priority / np.sqrt(max(counts[item.action], 1))
                for item in self.items
            ]
            weakest = int(np.argmin(utility))
            if score > utility[weakest]:
                self.items[weakest] = ep
            return

        if self.name == "oracle_future":
            # Diagnostic only: knows the future true theory and values evidence
            # where it differs most from the current wrong ensemble family.
            true = float(self.world.true_mean[action])
            wrong = float(np.mean([m[action] for m in WrongModelCache.get(self.world)]))
            ep.priority = ((true - wrong) / sigma) ** 2
            self._append_or_replace_lowest(ep)
            return

        raise ValueError(self.name)


class WrongModelCache:
    _cache: dict[int, np.ndarray] = {}

    @classmethod
    def set(cls, world: HypothesisBirthWorld, pred: np.ndarray) -> None:
        cls._cache[id(world)] = pred

    @classmethod
    def get(cls, world: HypothesisBirthWorld) -> np.ndarray:
        return cls._cache[id(world)]


def robust_memory_scores(
    world: HypothesisBirthWorld,
    supports: list[tuple[int, ...]],
    items: list[Episode],
) -> np.ndarray:
    """Score late-born theories using only what survived in memory.

    Clipping prevents one gross outlier from completely determining the result.
    Every finite memory policy gets the same evaluator.
    """
    if not items:
        return np.zeros(len(supports), dtype=float)
    score = np.zeros(len(supports), dtype=float)
    predictions = np.stack(
        [support_prediction(world.contribution, support) for support in supports],
        axis=0,
    )
    for ep in items:
        z2 = ((ep.y - predictions[:, ep.action]) / ep.sigma) ** 2
        score += -0.5 * np.minimum(z2, 9.0)
    return score


def top_population_from_memory(
    world: HypothesisBirthWorld,
    items: list[Episode],
    order: int,
    slots: int,
) -> tuple[list[tuple[int, ...]], np.ndarray, np.ndarray]:
    supports = list(itertools.combinations(range(world.contribution.shape[1]), order))
    score = robust_memory_scores(world, supports, items)
    top = np.argsort(score)[-slots:][::-1]
    chosen = [supports[int(i)] for i in top]
    chosen_score = score[top]
    pred = np.stack(
        [support_prediction(world.contribution, support) for support in chosen],
        axis=0,
    )
    return chosen, chosen_score, pred


def clean_prediction_mse(
    world: HypothesisBirthWorld,
    score: np.ndarray,
    prediction: np.ndarray,
) -> float:
    posterior = softmax(score)
    mean = posterior @ prediction
    return float(np.mean((mean - world.true_mean) ** 2))


def followup_eig(
    world: HypothesisBirthWorld,
    models: list[tuple[int, ...]],
    score: np.ndarray,
    prediction: np.ndarray,
    steps: int,
    seed: int,
) -> tuple[float, float]:
    # Fresh clean action streams: the question here is whether old memory put
    # the future theory into the live population, not whether later corruption
    # can be robustly filtered.
    streams = [
        np.random.default_rng(seed * 100_003 + 101 * a + 53)
        for a in range(len(world.x))
    ]
    logw = score.copy()
    noisy = 0
    for _ in range(steps):
        posterior = softmax(logw)
        info = np.asarray(
            [
                exact_information_gain(
                    posterior,
                    prediction[:, a],
                    float(world.sigma[a]),
                )
                for a in range(len(world.x))
            ]
        )
        action = int(np.argmax(info))
        if action in world.noisy_actions:
            noisy += 1
        y = float(world.true_mean[action] + streams[action].normal(0.0, world.sigma[action]))
        logw += -0.5 * ((y - prediction[:, action]) / world.sigma[action]) ** 2
    return clean_prediction_mse(world, logw, prediction), noisy / max(steps, 1)


@dataclass
class Result:
    truth_in_top6: float
    truth_top1: float
    pre_mse: float
    post_mse: float
    contaminated_fraction: float
    noisy_action_fraction: float


def run_one(
    seed: int,
    method: str,
    *,
    features: int,
    actions: int,
    order: int,
    slots: int,
    capacity: int,
    history_steps: int,
    followup_steps: int,
    contamination: float,
    contamination_scale: float,
) -> Result:
    world = HypothesisBirthWorld(seed, features=features, actions=actions, order=order)
    wrong = WrongModelState(world, seed, slots, order)
    WrongModelCache.set(world, wrong.pred)
    memory = MemoryPolicy(method, capacity, world, seed)
    contam_rng = np.random.default_rng(seed + 7_123_991)

    # Everyone sees exactly the same balanced historical sequence.
    offset = seed % actions
    for t in range(history_steps):
        action = int((t + offset) % actions)
        y = world.observe(action)
        is_contaminated = bool(contam_rng.random() < contamination)
        if is_contaminated:
            y += float(contam_rng.normal(0.0, contamination_scale * world.sigma[action]))

        raw, persistent = wrong.before_observation(action, y, float(world.sigma[action]))
        memory.write(
            action=action,
            y=y,
            sigma=float(world.sigma[action]),
            raw_excess=raw,
            persistent_excess=persistent,
            contaminated=is_contaminated,
        )
        wrong.update(action, y, float(world.sigma[action]))

    models, score, pred = top_population_from_memory(world, memory.items, order, slots)
    truth_in = float(world.true_support in models)
    truth_top1 = float(models[0] == world.true_support)
    pre_mse = clean_prediction_mse(world, score, pred)
    post_mse, noisy_frac = followup_eig(
        world, models, score, pred, followup_steps, seed + 99_001
    )
    contam_frac = float(np.mean([ep.contaminated for ep in memory.items])) if memory.items else 0.0
    return Result(truth_in, truth_top1, pre_mse, post_mse, contam_frac, noisy_frac)


def mean(vals: list[float]) -> float:
    return float(np.mean(np.asarray(vals, dtype=float)))


def report(title: str, results: dict[str, list[Result]]) -> None:
    print(title)
    print(
        f"{'method':26s} {'truth@6':>8s} {'top1':>8s} {'pre MSE':>10s} "
        f"{'post MSE':>10s} {'mem contam':>11s} {'follow noisy':>12s}"
    )
    for name, vals in results.items():
        print(
            f"{name:26s} "
            f"{mean([v.truth_in_top6 for v in vals]):8.3f} "
            f"{mean([v.truth_top1 for v in vals]):8.3f} "
            f"{mean([v.pre_mse for v in vals]):10.4f} "
            f"{mean([v.post_mse for v in vals]):10.4f} "
            f"{mean([v.contaminated_fraction for v in vals]):11.3f} "
            f"{mean([v.noisy_action_fraction for v in vals]):12.3f}"
        )
    print()


def run_gate(args: argparse.Namespace) -> None:
    finite = [
        "reservoir",
        "recent",
        "leverage",
        "raw_residual",
        "stubborn_diverse",
        "oracle_future",
    ]

    for contamination, label in [
        (0.0, "DYN9A — clean forgotten history"),
        (args.contamination, "DYN9B — contaminated forgotten history"),
    ]:
        for capacity in args.capacities:
            results: dict[str, list[Result]] = {"full_history": []}
            for m in finite:
                results[m] = []

            for seed in range(args.seeds):
                results["full_history"].append(
                    run_one(
                        seed,
                        "full_history",
                        features=args.features,
                        actions=args.actions,
                        order=args.order,
                        slots=args.slots,
                        capacity=args.history_steps,
                        history_steps=args.history_steps,
                        followup_steps=args.followup_steps,
                        contamination=contamination,
                        contamination_scale=args.contamination_scale,
                    )
                )
                for method in finite:
                    results[method].append(
                        run_one(
                            seed,
                            method,
                            features=args.features,
                            actions=args.actions,
                            order=args.order,
                            slots=args.slots,
                            capacity=capacity,
                            history_steps=args.history_steps,
                            followup_steps=args.followup_steps,
                            contamination=contamination,
                            contamination_scale=args.contamination_scale,
                        )
                    )

            print(
                f"{label}; memory={capacity}; history={args.history_steps}; "
                f"future live slots={args.slots}; follow-up EIG={args.followup_steps}"
            )
            report("", results)


def parse_capacities(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--actions", type=int, default=21)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--capacities", type=parse_capacities, default=parse_capacities("8,16,32"))
    parser.add_argument("--history-steps", type=int, default=126)
    parser.add_argument("--followup-steps", type=int, default=10)
    parser.add_argument("--contamination", type=float, default=0.08)
    parser.add_argument("--contamination-scale", type=float, default=5.0)
    args = parser.parse_args()
    run_gate(args)


if __name__ == "__main__":
    main()
