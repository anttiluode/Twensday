from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


def features(x: float) -> np.ndarray:
    return np.asarray(
        [1.0, x, x * x, np.sin(np.pi * x), np.cos(np.pi * x)],
        dtype=float,
    )


GRID_X = np.linspace(-1.0, 1.0, 101)
GRID_F = np.stack([features(float(x)) for x in GRID_X])


class RegionModel:
    """Recursive least-squares predictor plus a lagged model for prequential progress."""

    def __init__(self, dim: int = 5) -> None:
        self.w = np.zeros(dim, dtype=float)
        self.precision_inv = np.eye(dim, dtype=float)
        self.lag_w = np.zeros(dim, dtype=float)
        self.count = 0
        self.raw_error_ema = 1.0
        self.progress_ema = 0.0

    def observe(self, x: float, y: float) -> tuple[float, float]:
        f = features(x)

        prediction = float(self.w @ f)
        lag_prediction = float(self.lag_w @ f)
        loss = float((y - prediction) ** 2)
        lag_loss = float((y - lag_prediction) ** 2)

        # Both models are judged on the same fresh observation BEFORE either
        # is updated from it. Memorising the present sample cannot count as
        # learning progress.
        progress = lag_loss - loss

        ema = 0.12
        self.raw_error_ema = (1.0 - ema) * self.raw_error_ema + ema * loss
        self.progress_ema = (
            (1.0 - ema) * self.progress_ema + ema * progress
        )

        pf = self.precision_inv @ f
        gain = pf / (1.0 + float(f @ pf))
        self.w += gain * (y - prediction)
        self.precision_inv -= np.outer(gain, f) @ self.precision_inv

        # Slow shadow of the predictor. If current learning genuinely improves
        # prediction, it should beat this lagged version on future observations.
        lag_rate = 0.08
        self.lag_w = (1.0 - lag_rate) * self.lag_w + lag_rate * self.w
        self.count += 1
        return loss, progress


class World:
    """Independent region streams so each policy gets the same nth sample per region."""

    def __init__(
        self,
        seed: int,
        learnable_regions: int,
        noise_regions: int,
    ) -> None:
        self.learnable_regions = learnable_regions
        self.noise_regions = noise_regions
        self.region_count = learnable_regions + noise_regions

        rng = np.random.default_rng(seed + 100_003)
        theta = rng.normal(size=(learnable_regions, GRID_F.shape[1]))
        for i in range(learnable_regions):
            rms = float(np.sqrt(np.mean((GRID_F @ theta[i]) ** 2)))
            theta[i] /= max(rms, 1e-12)
        self.theta = theta

        # Separate deterministic stream per region. Policy A sampling region 3
        # for the nth time sees exactly what policy B sees on its nth sample.
        self.streams = [
            np.random.default_rng(seed * 10_007 + 97 * i + 19)
            for i in range(self.region_count)
        ]

    def sample(self, region: int) -> tuple[float, float]:
        rng = self.streams[region]
        x = float(rng.uniform(-1.0, 1.0))
        if region < self.learnable_regions:
            y = float(
                self.theta[region] @ features(x)
                + rng.normal(0.0, 0.05)
            )
        else:
            # Noisy TV: output is independent of x and therefore remains
            # observation-level unpredictable under the available model.
            y = float(rng.normal(0.0, 1.5))
        return x, y


def learnable_mse(models: list[RegionModel], world: World) -> float:
    mse = []
    for i in range(world.learnable_regions):
        prediction = GRID_F @ models[i].w
        target = GRID_F @ world.theta[i]
        mse.append(float(np.mean((prediction - target) ** 2)))
    return float(np.mean(mse))


def region_mse(models: list[RegionModel], world: World) -> np.ndarray:
    values = []
    for i in range(world.learnable_regions):
        prediction = GRID_F @ models[i].w
        target = GRID_F @ world.theta[i]
        values.append(float(np.mean((prediction - target) ** 2)))
    return np.asarray(values, dtype=float)


def epistemic_score(model: RegionModel) -> float:
    # Average linear-predictor posterior geometry. This is deliberately a
    # boring uncertainty baseline: it does not look at residual magnitude.
    probe = GRID_F[::4]
    variance = np.einsum(
        "nd,df,nf->n",
        probe,
        model.precision_inv,
        probe,
    )
    return float(np.mean(variance))


@dataclass
class RunResult:
    hit_step: int
    hit: bool
    final_mse: float
    noise_fraction: float
    checkpoints: dict[int, tuple[float, float]]


def choose_region(
    policy: str,
    models: list[RegionModel],
    world: World,
    rng: np.random.Generator,
    warmup: int,
    progress_beta: float,
) -> int:
    counts = np.asarray([m.count for m in models], dtype=int)

    if int(np.min(counts)) < warmup:
        candidates = np.flatnonzero(counts == np.min(counts))
        return int(rng.choice(candidates))

    if policy == "random":
        return int(rng.integers(len(models)))

    if policy == "count_balanced":
        candidates = np.flatnonzero(counts == np.min(counts))
        return int(rng.choice(candidates))

    if policy == "raw_error":
        score = np.asarray([m.raw_error_ema for m in models])
        return int(np.argmax(score))

    if policy == "uncertainty":
        score = np.asarray([epistemic_score(m) for m in models])
        return int(np.argmax(score))

    if policy == "learning_progress":
        # Positive prequential improvement plus a weak revisit pressure.
        # The revisit term prevents one lucky region from permanently locking
        # out regions whose progress estimate has gone stale.
        score = np.asarray(
            [
                max(0.0, m.progress_ema)
                + progress_beta / np.sqrt(m.count + 1.0)
                for m in models
            ]
        )
        score += rng.random(len(models)) * 1e-12
        return int(np.argmax(score))

    if policy == "oracle_reducible":
        # Diagnostic upper bound: it knows which regions are learnable and
        # spends the next sample where true remaining model error is largest.
        score = region_mse(models, world)
        return int(np.argmax(score))

    raise ValueError(f"unknown policy: {policy}")


def run_policy(
    policy: str,
    seed: int,
    *,
    learnable_regions: int,
    noise_regions: int,
    max_steps: int,
    eval_every: int,
    threshold: float,
    warmup: int,
    progress_beta: float,
) -> RunResult:
    world = World(seed, learnable_regions, noise_regions)
    models = [RegionModel() for _ in range(world.region_count)]
    policy_rng = np.random.default_rng(seed + 7_000_001)

    noise_samples = 0
    hit_step = max_steps + eval_every
    hit = False
    checkpoints: dict[int, tuple[float, float]] = {}

    for t in range(max_steps):
        region = choose_region(
            policy,
            models,
            world,
            policy_rng,
            warmup,
            progress_beta,
        )
        if region >= learnable_regions:
            noise_samples += 1

        x, y = world.sample(region)
        models[region].observe(x, y)

        step = t + 1
        if step % eval_every == 0:
            mse = learnable_mse(models, world)
            noise_fraction = noise_samples / step
            checkpoints[step] = (mse, noise_fraction)
            if (not hit) and mse < threshold:
                hit_step = step
                hit = True

    final_mse = learnable_mse(models, world)
    return RunResult(
        hit_step=hit_step,
        hit=hit,
        final_mse=final_mse,
        noise_fraction=noise_samples / max_steps,
        checkpoints=checkpoints,
    )


def mean_std(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def run_many(args: argparse.Namespace) -> None:
    policies = [
        "random",
        "count_balanced",
        "uncertainty",
        "raw_error",
        "learning_progress",
        "oracle_reducible",
    ]
    all_results: dict[str, list[RunResult]] = {p: [] for p in policies}

    for seed in range(args.seeds):
        for policy in policies:
            all_results[policy].append(
                run_policy(
                    policy,
                    seed,
                    learnable_regions=args.learnable,
                    noise_regions=args.noise,
                    max_steps=args.max_steps,
                    eval_every=args.eval_every,
                    threshold=args.threshold,
                    warmup=args.warmup,
                    progress_beta=args.progress_beta,
                )
            )

    checkpoints = [
        step
        for step in (360, 600, 900, args.max_steps)
        if step <= args.max_steps and step % args.eval_every == 0
    ]
    checkpoints = sorted(set(checkpoints))

    print("DYN6 — curiosity must survive noisy TV")
    print(
        f"{args.learnable} learnable regions + {args.noise} stochastic regions; "
        f"{args.seeds} seeds; threshold MSE={args.threshold:.4f}"
    )
    print(
        "Learning progress = fresh-sample loss(lagged model) "
        "- loss(current model), EMA-smoothed."
    )
    print()

    for step in checkpoints:
        print(f"checkpoint {step}")
        print(
            f"{'policy':20s} {'learnable MSE':>16s} "
            f"{'noise fraction':>16s}"
        )
        for policy in policies:
            mse = [r.checkpoints[step][0] for r in all_results[policy]]
            noise = [r.checkpoints[step][1] for r in all_results[policy]]
            mse_mean, mse_std = mean_std(mse)
            noise_mean, noise_std = mean_std(noise)
            print(
                f"{policy:20s} "
                f"{mse_mean:8.5f} +/- {mse_std:7.5f} "
                f"{noise_mean:8.3f} +/- {noise_std:7.3f}"
            )
        print()

    print("sample efficiency")
    print(
        f"{'policy':20s} {'mean hit':>10s} {'median hit':>12s} "
        f"{'hit rate':>10s} {'final MSE':>12s} {'final noise':>12s}"
    )
    for policy in policies:
        results = all_results[policy]
        hit_steps = np.asarray([r.hit_step for r in results], dtype=float)
        hit_rate = float(np.mean([r.hit for r in results]))
        final_mse = float(np.mean([r.final_mse for r in results]))
        noise = float(np.mean([r.noise_fraction for r in results]))
        print(
            f"{policy:20s} "
            f"{float(np.mean(hit_steps)):10.1f} "
            f"{float(np.median(hit_steps)):12.1f} "
            f"{hit_rate:10.3f} "
            f"{final_mse:12.5f} "
            f"{noise:12.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--learnable", type=int, default=6)
    parser.add_argument("--noise", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--progress-beta", type=float, default=0.01)
    args = parser.parse_args()
    run_many(args)


if __name__ == "__main__":
    main()
