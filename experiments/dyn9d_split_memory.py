from __future__ import annotations

import argparse
import numpy as np

from dyn8_evolving_hypotheses import HypothesisBirthWorld
from dyn9_future_theory_memory import (
    Episode,
    Result,
    WrongModelCache,
    WrongModelState,
    followup_eig,
    mean,
    run_one,
    top_population_from_memory,
    clean_prediction_mse,
)


class SplitMemory:
    """One hard budget split between representative and stubborn evidence.

    Samples with persistent standardized mismatch enter the stubborn lane;
    ordinary traffic competes in a reservoir lane. The lane capacities sum to
    the advertised memory budget, so there is no hidden extra raw storage.
    """

    def __init__(self, capacity: int, seed: int, threshold: float = 1.5) -> None:
        self.stubborn_cap = max(1, capacity // 2)
        self.representative_cap = capacity - self.stubborn_cap
        self.threshold = threshold
        self.rng = np.random.default_rng(seed + 771_991)
        self.stubborn: list[Episode] = []
        self.representative: list[Episode] = []
        self.rep_seen = 0

    @property
    def items(self) -> list[Episode]:
        return self.stubborn + self.representative

    def _write_stubborn(self, ep: Episode, persistent: float) -> None:
        same = sum(int(item.action == ep.action) for item in self.stubborn)
        ep.priority = float(persistent / np.sqrt(1.0 + same))
        if len(self.stubborn) < self.stubborn_cap:
            self.stubborn.append(ep)
            return
        counts: dict[int, int] = {}
        for item in self.stubborn:
            counts[item.action] = counts.get(item.action, 0) + 1
        utility = [
            item.priority / np.sqrt(max(counts[item.action], 1))
            for item in self.stubborn
        ]
        weakest = int(np.argmin(utility))
        if ep.priority > utility[weakest]:
            self.stubborn[weakest] = ep

    def _write_reservoir(self, ep: Episode) -> None:
        if self.representative_cap <= 0:
            return
        self.rep_seen += 1
        if len(self.representative) < self.representative_cap:
            self.representative.append(ep)
            return
        j = int(self.rng.integers(self.rep_seen))
        if j < self.representative_cap:
            self.representative[j] = ep

    def write(
        self,
        *,
        action: int,
        y: float,
        sigma: float,
        persistent_excess: float,
        contaminated: bool,
    ) -> None:
        ep = Episode(action, y, sigma, 0.0, contaminated)
        if persistent_excess >= self.threshold:
            self._write_stubborn(ep, persistent_excess)
        else:
            self._write_reservoir(ep)


def run_split(
    seed: int,
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
    threshold: float,
) -> Result:
    world = HypothesisBirthWorld(seed, features=features, actions=actions, order=order)
    wrong = WrongModelState(world, seed, slots, order)
    WrongModelCache.set(world, wrong.pred)
    memory = SplitMemory(capacity, seed, threshold=threshold)
    contam_rng = np.random.default_rng(seed + 7_123_991)
    offset = seed % actions

    for t in range(history_steps):
        action = int((t + offset) % actions)
        y = world.observe(action)
        is_contaminated = bool(contam_rng.random() < contamination)
        if is_contaminated:
            y += float(contam_rng.normal(0.0, contamination_scale * world.sigma[action]))
        _, persistent = wrong.before_observation(action, y, float(world.sigma[action]))
        memory.write(
            action=action,
            y=y,
            sigma=float(world.sigma[action]),
            persistent_excess=persistent,
            contaminated=is_contaminated,
        )
        wrong.update(action, y, float(world.sigma[action]))

    models, score, pred = top_population_from_memory(world, memory.items, order, slots)
    truth_in = float(world.true_support in models)
    truth_top1 = float(models[0] == world.true_support)
    pre_mse = clean_prediction_mse(world, score, pred)
    post_mse, noisy_frac = followup_eig(world, models, score, pred, followup_steps, seed + 99_001)
    contam_frac = float(np.mean([ep.contaminated for ep in memory.items])) if memory.items else 0.0
    return Result(truth_in, truth_top1, pre_mse, post_mse, contam_frac, noisy_frac)


def summarize(label: str, rows: dict[str, list[Result]]) -> None:
    print(label)
    print(f"{'method':22s} {'truth@6':>8s} {'top1':>8s} {'pre MSE':>10s} {'post MSE':>10s} {'contam':>9s}")
    for name, vals in rows.items():
        print(
            f"{name:22s} "
            f"{mean([v.truth_in_top6 for v in vals]):8.3f} "
            f"{mean([v.truth_top1 for v in vals]):8.3f} "
            f"{mean([v.pre_mse for v in vals]):10.4f} "
            f"{mean([v.post_mse for v in vals]):10.4f} "
            f"{mean([v.contaminated_fraction for v in vals]):9.3f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=48)
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--actions', type=int, default=25)
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--slots', type=int, default=6)
    parser.add_argument('--capacity', type=int, default=6)
    parser.add_argument('--history-steps', type=int, default=150)
    parser.add_argument('--followup-steps', type=int, default=12)
    parser.add_argument('--contamination', type=float, default=0.10)
    parser.add_argument('--contamination-scale', type=float, default=6.0)
    parser.add_argument('--thresholds', type=str, default='0.75,1.5,3.0')
    args = parser.parse_args()

    methods = ['reservoir', 'recent', 'leverage', 'raw_residual', 'stubborn_diverse']
    thresholds = [float(x) for x in args.thresholds.split(',')]

    for contamination, label in [(0.0, 'DYN9D clean'), (args.contamination, 'DYN9D contaminated')]:
        rows: dict[str, list[Result]] = {m: [] for m in methods}
        for th in thresholds:
            rows[f'split_{th:g}'] = []
        for seed in range(args.seeds):
            for method in methods:
                rows[method].append(
                    run_one(
                        seed,
                        method,
                        features=args.features,
                        actions=args.actions,
                        order=args.order,
                        slots=args.slots,
                        capacity=args.capacity,
                        history_steps=args.history_steps,
                        followup_steps=args.followup_steps,
                        contamination=contamination,
                        contamination_scale=args.contamination_scale,
                    )
                )
            for th in thresholds:
                rows[f'split_{th:g}'].append(
                    run_split(
                        seed,
                        features=args.features,
                        actions=args.actions,
                        order=args.order,
                        slots=args.slots,
                        capacity=args.capacity,
                        history_steps=args.history_steps,
                        followup_steps=args.followup_steps,
                        contamination=contamination,
                        contamination_scale=args.contamination_scale,
                        threshold=th,
                    )
                )
        summarize(f'{label}; hard memory={args.capacity}', rows)


if __name__ == '__main__':
    main()
