from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


GH_X, GH_W = np.polynomial.hermite.hermgauss(5)
GH_W = GH_W / np.sqrt(np.pi)


def entropy(p: np.ndarray) -> float:
    q = p[p > 1e-300]
    return float(-np.sum(q * np.log(q)))


def logsumexp(values: np.ndarray) -> float:
    m = float(np.max(values))
    return m + float(np.log(np.sum(np.exp(values - m))))


def posterior_update(
    posterior: np.ndarray,
    means: np.ndarray,
    sigma: float,
    y: float,
) -> np.ndarray:
    logp = (
        np.log(posterior + 1e-300)
        - 0.5 * ((y - means) / sigma) ** 2
        - np.log(sigma)
    )
    logp -= float(np.max(logp))
    p = np.exp(logp)
    p /= float(np.sum(p))
    return p


def predictive_log_density(
    posterior: np.ndarray,
    means: np.ndarray,
    sigma: float,
    y: float,
) -> float:
    terms = (
        np.log(posterior + 1e-300)
        - 0.5 * ((y - means) / sigma) ** 2
        - np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
    )
    return logsumexp(terms)


def expected_information_gain(
    posterior: np.ndarray,
    means: np.ndarray,
    sigma: float,
) -> float:
    """E[H(posterior) - H(posterior after y)] under the current mixture."""
    h0 = entropy(posterior)
    y = (
        means[:, None]
        + np.sqrt(2.0) * sigma * GH_X[None, :]
    ).reshape(-1)
    sample_weight = (posterior[:, None] * GH_W[None, :]).reshape(-1)

    logpost = (
        np.log(posterior[None, :] + 1e-300)
        - 0.5 * ((y[:, None] - means[None, :]) / sigma) ** 2
        - np.log(sigma)
    )
    row_max = np.max(logpost, axis=1, keepdims=True)
    norm = row_max + np.log(
        np.sum(np.exp(logpost - row_max), axis=1, keepdims=True)
    )
    post = np.exp(logpost - norm)
    post_entropy = -np.sum(
        post * np.log(post + 1e-300),
        axis=1,
    )
    return h0 - float(np.sum(sample_weight * post_entropy))


def oracle_true_information(
    posterior: np.ndarray,
    means: np.ndarray,
    sigma: float,
    true_hypothesis: int,
) -> float:
    """Truth-aware diagnostic: expected log evidence for the true model."""
    score = 0.0
    mu_true = float(means[true_hypothesis])
    for z, weight in zip(GH_X, GH_W):
        y = mu_true + np.sqrt(2.0) * sigma * float(z)
        log_true = (
            -0.5 * ((y - mu_true) / sigma) ** 2
            - np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
        )
        log_mix = predictive_log_density(posterior, means, sigma, y)
        score += float(weight) * (float(log_true) - log_mix)
    return float(score)


class HypothesisWorld:
    """Competing smooth laws plus action-dependent noisy instruments."""

    def __init__(
        self,
        seed: int,
        hypotheses: int,
        actions: int,
    ) -> None:
        rng = np.random.default_rng(seed + 123_457)
        self.x = np.linspace(-1.0, 1.0, actions)

        design = np.stack(
            [
                np.ones(actions),
                self.x,
                self.x**2,
                np.sin(np.pi * self.x),
                np.cos(2.0 * np.pi * self.x),
                np.sin(2.0 * np.pi * self.x),
            ],
            axis=1,
        )

        # Hypotheses share a broad family so ordinary observations can leave
        # several live explanations. Each hypothesis is a nearby law.
        base = rng.normal(size=design.shape[1])
        coeff = base[None, :] + 0.35 * rng.normal(
            size=(hypotheses, design.shape[1])
        )
        means = coeff @ design.T
        means /= max(float(np.sqrt(np.mean(means**2))), 1e-12)
        self.means = means

        self.true_hypothesis = int(rng.integers(hypotheses))

        self.sigma = np.full(actions, 0.28, dtype=float)
        noisy = sorted(
            set(
                [
                    0,
                    actions // 4,
                    actions // 2,
                    (3 * actions) // 4,
                    actions - 1,
                ]
            )
        )
        self.noisy_actions = set(noisy)
        self.sigma[noisy] = 1.40

        # Same nth observation from an action for every policy.
        self.streams = [
            np.random.default_rng(seed * 10_007 + 31 * a + 7)
            for a in range(actions)
        ]

    def observe(self, action: int) -> float:
        rng = self.streams[action]
        return float(
            self.means[self.true_hypothesis, action]
            + rng.normal(0.0, self.sigma[action])
        )


class ProgressState:
    """DYN6-style prequential progress, now attached to experiment actions."""

    def __init__(self, actions: int, hypotheses: int) -> None:
        self.count = np.zeros(actions, dtype=int)
        self.progress = np.zeros(actions, dtype=float)
        self.lag_posterior = np.full(
            hypotheses,
            1.0 / hypotheses,
            dtype=float,
        )

    def observe(
        self,
        action: int,
        posterior_before: np.ndarray,
        means: np.ndarray,
        sigma: float,
        y: float,
        posterior_after: np.ndarray,
    ) -> None:
        current_logp = predictive_log_density(
            posterior_before,
            means,
            sigma,
            y,
        )
        lagged_logp = predictive_log_density(
            self.lag_posterior,
            means,
            sigma,
            y,
        )
        fresh_progress = current_logp - lagged_logp
        self.progress[action] = (
            0.85 * self.progress[action]
            + 0.15 * fresh_progress
        )

        self.lag_posterior = (
            0.92 * self.lag_posterior
            + 0.08 * posterior_after
        )
        self.lag_posterior /= float(np.sum(self.lag_posterior))
        self.count[action] += 1


@dataclass
class RunResult:
    hit_step: int
    hit: bool
    correct: bool
    true_probability: float
    final_entropy: float
    noisy_fraction: float
    checkpoints: dict[int, tuple[float, float, float, float]]


def choose_action(
    policy: str,
    posterior: np.ndarray,
    world: HypothesisWorld,
    state: ProgressState,
    rng: np.random.Generator,
    progress_beta: float,
) -> int:
    actions = len(world.x)

    if policy == "random":
        return int(rng.integers(actions))

    if policy == "coverage":
        candidates = np.flatnonzero(state.count == np.min(state.count))
        return int(rng.choice(candidates))

    weighted_mean = posterior @ world.means
    disagreement = np.sum(
        posterior[:, None] * (world.means - weighted_mean) ** 2,
        axis=0,
    )

    if policy == "raw_predictive_variance":
        # This deliberately conflates hypothesis disagreement with instrument
        # noise, reproducing a research-scale noisy-TV trap.
        return int(np.argmax(disagreement + world.sigma**2))

    if policy == "model_disagreement":
        # Query-by-committee style score, but it does not discount a bad
        # instrument whose means happen to disagree strongly.
        return int(np.argmax(disagreement))

    if policy == "learning_progress":
        score = (
            np.maximum(0.0, state.progress)
            + progress_beta / np.sqrt(state.count + 1.0)
        )
        score += rng.random(actions) * 1e-12
        return int(np.argmax(score))

    if policy == "expected_information_gain":
        score = np.asarray(
            [
                expected_information_gain(
                    posterior,
                    world.means[:, a],
                    float(world.sigma[a]),
                )
                for a in range(actions)
            ]
        )
        return int(np.argmax(score))

    if policy == "oracle_true_information":
        score = np.asarray(
            [
                oracle_true_information(
                    posterior,
                    world.means[:, a],
                    float(world.sigma[a]),
                    world.true_hypothesis,
                )
                for a in range(actions)
            ]
        )
        return int(np.argmax(score))

    raise ValueError(policy)


def run_policy(
    policy: str,
    seed: int,
    *,
    hypotheses: int,
    actions: int,
    max_steps: int,
    threshold: float,
    progress_beta: float,
) -> RunResult:
    world = HypothesisWorld(seed, hypotheses, actions)
    posterior = np.full(hypotheses, 1.0 / hypotheses)
    state = ProgressState(actions, hypotheses)
    rng = np.random.default_rng(seed + 8_000_003)

    hit_step = max_steps + 1
    hit = False
    noisy = 0
    checkpoints: dict[int, tuple[float, float, float, float]] = {}
    checkpoint_steps = {5, 10, 20, 40, max_steps}

    for t in range(max_steps):
        action = choose_action(
            policy,
            posterior,
            world,
            state,
            rng,
            progress_beta,
        )
        if action in world.noisy_actions:
            noisy += 1

        y = world.observe(action)
        before = posterior.copy()
        posterior = posterior_update(
            posterior,
            world.means[:, action],
            float(world.sigma[action]),
            y,
        )
        state.observe(
            action,
            before,
            world.means[:, action],
            float(world.sigma[action]),
            y,
            posterior,
        )

        step = t + 1
        p_true = float(posterior[world.true_hypothesis])
        if (not hit) and p_true >= threshold:
            hit_step = step
            hit = True

        if step in checkpoint_steps:
            checkpoints[step] = (
                float(np.argmax(posterior) == world.true_hypothesis),
                p_true,
                entropy(posterior),
                noisy / step,
            )

    return RunResult(
        hit_step=hit_step,
        hit=hit,
        correct=bool(
            int(np.argmax(posterior)) == world.true_hypothesis
        ),
        true_probability=float(
            posterior[world.true_hypothesis]
        ),
        final_entropy=entropy(posterior),
        noisy_fraction=noisy / max_steps,
        checkpoints=checkpoints,
    )


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def run_many(args: argparse.Namespace) -> None:
    policies = [
        "random",
        "coverage",
        "raw_predictive_variance",
        "learning_progress",
        "model_disagreement",
        "expected_information_gain",
        "oracle_true_information",
    ]
    results: dict[str, list[RunResult]] = {p: [] for p in policies}

    for seed in range(args.seeds):
        for policy in policies:
            results[policy].append(
                run_policy(
                    policy,
                    seed,
                    hypotheses=args.hypotheses,
                    actions=args.actions,
                    max_steps=args.max_steps,
                    threshold=args.threshold,
                    progress_beta=args.progress_beta,
                )
            )

    checkpoint_steps = sorted(
        {s for s in (5, 10, 20, 40, args.max_steps) if s <= args.max_steps}
    )

    print("DYN7 — curiosity becomes experiment choice")
    print(
        f"{args.hypotheses} hypotheses, {args.actions} possible experiments, "
        f"{args.seeds} seeds, posterior threshold={args.threshold:.2f}"
    )
    print()

    for step in checkpoint_steps:
        print(f"checkpoint {step}")
        print(
            f"{'policy':25s} {'accuracy':>10s} {'P(true)':>10s} "
            f"{'entropy':>10s} {'noisy frac':>12s}"
        )
        for policy in policies:
            vals = [r.checkpoints[step] for r in results[policy]]
            print(
                f"{policy:25s} "
                f"{mean([v[0] for v in vals]):10.3f} "
                f"{mean([v[1] for v in vals]):10.3f} "
                f"{mean([v[2] for v in vals]):10.3f} "
                f"{mean([v[3] for v in vals]):12.3f}"
            )
        print()

    print("sample efficiency")
    print(
        f"{'policy':25s} {'mean hit':>10s} {'median':>10s} "
        f"{'hit rate':>10s} {'final acc':>10s} {'final noise':>12s}"
    )
    for policy in policies:
        vals = results[policy]
        hit_steps = np.asarray([r.hit_step for r in vals], dtype=float)
        print(
            f"{policy:25s} "
            f"{float(np.mean(hit_steps)):10.2f} "
            f"{float(np.median(hit_steps)):10.2f} "
            f"{mean([float(r.hit) for r in vals]):10.3f} "
            f"{mean([float(r.correct) for r in vals]):10.3f} "
            f"{mean([r.noisy_fraction for r in vals]):12.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--hypotheses", type=int, default=12)
    parser.add_argument("--actions", type=int, default=21)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--progress-beta", type=float, default=0.03)
    args = parser.parse_args()
    run_many(args)


if __name__ == "__main__":
    main()
