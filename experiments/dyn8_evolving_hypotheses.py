from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass

import numpy as np


GH_X, GH_W = np.polynomial.hermite.hermgauss(3)
GH_W = GH_W / np.sqrt(np.pi)


def entropy(p: np.ndarray) -> float:
    q = p[p > 1e-300]
    return float(-np.sum(q * np.log(q)))


def softmax(logw: np.ndarray) -> np.ndarray:
    z = logw - float(np.max(logw))
    w = np.exp(z)
    w /= float(np.sum(w)) + 1e-300
    return w


def exact_information_gain(
    posterior: np.ndarray,
    means: np.ndarray,
    sigma: float,
) -> float:
    """DYN7-style E[H(before)-H(after y)] for a finite hypothesis set."""
    h0 = entropy(posterior)
    y = (
        means[:, None]
        + np.sqrt(2.0) * sigma * GH_X[None, :]
    ).reshape(-1)
    sample_weight = (
        posterior[:, None] * GH_W[None, :]
    ).reshape(-1)

    logpost = (
        np.log(posterior[None, :] + 1e-300)
        - 0.5 * ((y[:, None] - means[None, :]) / sigma) ** 2
    )
    row_max = np.max(logpost, axis=1, keepdims=True)
    post = np.exp(logpost - row_max)
    post /= np.sum(post, axis=1, keepdims=True)
    post_entropy = -np.sum(
        post * np.log(post + 1e-300),
        axis=1,
    )
    return h0 - float(np.sum(sample_weight * post_entropy))


def gaussian_information_gain(
    posterior: np.ndarray,
    means: np.ndarray,
    sigma: float,
) -> float:
    """Cheap Gaussian approximation to mutual information.

    It discounts known observation noise, unlike raw model disagreement.
    DYN8A uses it so the 364-model catalog remains cheap enough to attack.
    DYN8B repeats the smaller problem with exact_information_gain.
    """
    mean = float(posterior @ means)
    epistemic_var = float(
        posterior @ ((means - mean) ** 2)
    )
    return 0.5 * float(
        np.log1p(epistemic_var / (sigma**2))
    )


def make_library(actions: int, features: int) -> tuple[np.ndarray, np.ndarray]:
    """A grammar of reusable primitives, not a catalog of explanations."""
    x = np.linspace(-1.0, 1.0, actions)
    cols: list[np.ndarray] = [
        x,
        x**2 - float(np.mean(x**2)),
        x**3,
    ]
    freq = 1
    while len(cols) < features:
        cols.append(np.sin(np.pi * freq * x))
        if len(cols) < features:
            cols.append(np.cos(np.pi * freq * x))
        freq += 1

    phi = np.stack(cols[:features], axis=1)
    phi -= np.mean(phi, axis=0, keepdims=True)
    phi /= np.sqrt(np.mean(phi**2, axis=0, keepdims=True)) + 1e-12

    # Fixed feature amplitudes keep DYN8 about structure birth/death rather
    # than simultaneous nonlinear parameter fitting. The combination is not
    # supplied to finite-population learners.
    coeff = np.asarray(
        [0.65 + 0.08 * ((7 * j) % 5) for j in range(features)],
        dtype=float,
    )
    coeff *= np.where(np.arange(features) % 2 == 0, 1.0, -1.0)
    return x, phi * coeff[None, :]


def support_prediction(
    contribution: np.ndarray,
    support: tuple[int, ...],
) -> np.ndarray:
    return np.sum(contribution[:, list(support)], axis=1)


class HypothesisBirthWorld:
    """Sparse hidden law plus several deliberately bad instruments."""

    def __init__(
        self,
        seed: int,
        *,
        features: int,
        actions: int,
        order: int,
    ) -> None:
        self.x, self.contribution = make_library(actions, features)
        rng = np.random.default_rng(seed + 123_457)
        self.true_support = tuple(
            sorted(
                int(v)
                for v in rng.choice(
                    features,
                    size=order,
                    replace=False,
                )
            )
        )
        self.true_mean = support_prediction(
            self.contribution,
            self.true_support,
        )

        self.sigma = np.full(actions, 0.22, dtype=float)
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
        self.sigma[noisy] = 0.90

        # Every policy gets the same nth outcome from each experiment.
        self.streams = [
            np.random.default_rng(seed * 10_007 + 37 * a + 11)
            for a in range(actions)
        ]

    def observe(self, action: int) -> float:
        return float(
            self.true_mean[action]
            + self.streams[action].normal(
                0.0,
                self.sigma[action],
            )
        )


class EvidenceSummary:
    """Finite action-level sufficient statistics shared by all attackers.

    This deliberately gives every finite-population method a fair way to score
    a newly born explanation against old evidence. H is therefore tested as a
    *proposal mechanism*, not as the only store of evidence.
    """

    def __init__(self, actions: int) -> None:
        self.count = np.zeros(actions, dtype=int)
        self.sum_y = np.zeros(actions, dtype=float)
        self.sum_y2 = np.zeros(actions, dtype=float)

    def add(self, action: int, y: float) -> None:
        self.count[action] += 1
        self.sum_y[action] += y
        self.sum_y2[action] += y * y

    def log_likelihood(
        self,
        prediction: np.ndarray,
        sigma: np.ndarray,
    ) -> float:
        sse = (
            self.sum_y2
            - 2.0 * prediction * self.sum_y
            + self.count * prediction**2
        )
        return float(
            np.sum(
                -0.5 * sse / (sigma**2)
                - self.count * np.log(sigma)
            )
        )


class StubbornMemory:
    """Tiny H: observations that the current explanation population misses."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: list[tuple[float, int, float, float]] = []

    def write(
        self,
        priority: float,
        action: int,
        y: float,
        sigma: float,
    ) -> None:
        item = (
            float(priority),
            int(action),
            float(y),
            float(sigma),
        )
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        weakest = min(
            range(len(self.items)),
            key=lambda i: self.items[i][0],
        )
        if priority > self.items[weakest][0]:
            self.items[weakest] = item

    def loss(self, prediction: np.ndarray) -> float:
        if not self.items:
            return float("inf")
        return float(
            np.mean(
                [
                    ((y - prediction[action]) / sigma) ** 2
                    for _, action, y, sigma in self.items
                ]
            )
        )


class FiniteExplanationPopulation:
    """K live sparse explanations; the true one is absent at initialization."""

    def __init__(
        self,
        world: HypothesisBirthWorld,
        seed: int,
        *,
        slots: int,
        order: int,
        memory_capacity: int,
    ) -> None:
        self.world = world
        self.rng = np.random.default_rng(seed + 8_000_003)
        self.slots = slots
        self.order = order
        self.models: list[tuple[int, ...]] = []
        features = world.contribution.shape[1]
        while len(self.models) < slots:
            support = tuple(
                sorted(
                    int(v)
                    for v in self.rng.choice(
                        features,
                        size=order,
                        replace=False,
                    )
                )
            )
            if (
                support != world.true_support
                and support not in self.models
            ):
                self.models.append(support)

        self.evidence = EvidenceSummary(len(world.x))
        self.memory = StubbornMemory(memory_capacity)
        self.action_count = np.zeros(len(world.x), dtype=int)

    def predictions(self) -> np.ndarray:
        return np.stack(
            [
                support_prediction(
                    self.world.contribution,
                    support,
                )
                for support in self.models
            ],
            axis=0,
        )

    def log_weights(self) -> np.ndarray:
        return np.asarray(
            [
                self.evidence.log_likelihood(
                    prediction,
                    self.world.sigma,
                )
                for prediction in self.predictions()
            ],
            dtype=float,
        )

    def weights(self) -> np.ndarray:
        return softmax(self.log_weights())

    def choose_action(
        self,
        acquisition: str,
        *,
        information_mode: str,
        explore_beta: float,
    ) -> int:
        actions = len(self.world.x)
        if acquisition == "random":
            return int(self.rng.integers(actions))
        if acquisition == "coverage":
            candidates = np.flatnonzero(
                self.action_count == np.min(self.action_count)
            )
            return int(self.rng.choice(candidates))

        posterior = self.weights()
        prediction = self.predictions()
        info_fn = (
            exact_information_gain
            if information_mode == "exact"
            else gaussian_information_gain
        )
        score = np.asarray(
            [
                info_fn(
                    posterior,
                    prediction[:, action],
                    float(self.world.sigma[action]),
                )
                for action in range(actions)
            ],
            dtype=float,
        )
        if acquisition == "hybrid":
            score += explore_beta / np.sqrt(
                self.action_count + 1.0
            )
        score += self.rng.random(actions) * 1e-12
        return int(np.argmax(score))

    def observe(self, action: int, y: float) -> None:
        posterior = self.weights()
        prediction = self.predictions()
        ensemble = float(posterior @ prediction[:, action])
        sigma = float(self.world.sigma[action])

        # Excess standardized error: noisy instruments do not automatically
        # receive larger residual priority merely because sigma is larger.
        priority = max(
            ((y - ensemble) / sigma) ** 2 - 1.0,
            0.0,
        )
        self.memory.write(priority, action, y, sigma)
        self.evidence.add(action, y)
        self.action_count[action] += 1

    def replace_worst(
        self,
        child: tuple[int, ...] | None,
    ) -> None:
        if child is None or child in self.models:
            return
        worst = int(np.argmin(self.log_weights()))
        self.models[worst] = child

    def residual_birth(self) -> None:
        """Twensday proposal: let stubborn H suggest one structural mutation."""
        if not self.memory.items:
            return
        logw = self.log_weights()
        parent = self.models[int(np.argmax(logw))]
        base_loss = self.memory.loss(
            support_prediction(
                self.world.contribution,
                parent,
            )
        )

        features = self.world.contribution.shape[1]
        best_child: tuple[int, ...] | None = None
        best_loss = base_loss
        for removed in parent:
            for added in range(features):
                if added in parent:
                    continue
                child = tuple(
                    sorted((set(parent) - {removed}) | {added})
                )
                if child in self.models:
                    continue
                loss = self.memory.loss(
                    support_prediction(
                        self.world.contribution,
                        child,
                    )
                )
                if loss < best_loss:
                    best_loss = loss
                    best_child = child
        self.replace_worst(best_child)

    def beam_birth(self) -> None:
        """Boring attacker: exhaustive one-swap search on all evidence."""
        logw = self.log_weights()
        best_ll = float(np.max(logw))
        best_child: tuple[int, ...] | None = None
        features = self.world.contribution.shape[1]

        for parent in self.models:
            for removed in parent:
                for added in range(features):
                    if added in parent:
                        continue
                    child = tuple(
                        sorted((set(parent) - {removed}) | {added})
                    )
                    if child in self.models:
                        continue
                    prediction = support_prediction(
                        self.world.contribution,
                        child,
                    )
                    ll = self.evidence.log_likelihood(
                        prediction,
                        self.world.sigma,
                    )
                    if ll > best_ll:
                        best_ll = ll
                        best_child = child
        self.replace_worst(best_child)

    def smc_rejuvenate(self, moves: int) -> None:
        """Equal-slot resample-move SMC-style structure attacker."""
        weights = self.weights()
        parents = [
            self.models[int(self.rng.choice(self.slots, p=weights))]
            for _ in range(self.slots)
        ]
        features = self.world.contribution.shape[1]
        new_models: list[tuple[int, ...]] = []

        for parent in parents:
            current = parent
            current_ll = self.evidence.log_likelihood(
                support_prediction(
                    self.world.contribution,
                    current,
                ),
                self.world.sigma,
            )
            for _ in range(moves):
                removed = int(self.rng.choice(current))
                available = [
                    j for j in range(features) if j not in current
                ]
                added = int(self.rng.choice(available))
                proposal = tuple(
                    sorted((set(current) - {removed}) | {added})
                )
                proposal_ll = self.evidence.log_likelihood(
                    support_prediction(
                        self.world.contribution,
                        proposal,
                    ),
                    self.world.sigma,
                )
                if math.log(self.rng.random() + 1e-300) < min(
                    0.0,
                    proposal_ll - current_ll,
                ):
                    current = proposal
                    current_ll = proposal_ll
            if current not in new_models:
                new_models.append(current)

        # Diversity refill after resampling duplicates.
        tries = 0
        while len(new_models) < self.slots and tries < 500:
            parent = parents[int(self.rng.integers(len(parents)))]
            removed = int(self.rng.choice(parent))
            available = [
                j for j in range(features) if j not in parent
            ]
            added = int(self.rng.choice(available))
            proposal = tuple(
                sorted((set(parent) - {removed}) | {added})
            )
            if proposal not in new_models:
                new_models.append(proposal)
            tries += 1

        self.models = new_models[: self.slots]

    def predictive_mse(self) -> float:
        prediction = self.weights() @ self.predictions()
        return float(
            np.mean((prediction - self.world.true_mean) ** 2)
        )

    def contains_truth(self) -> bool:
        return self.world.true_support in self.models


@dataclass
class RunResult:
    structure_hit: int
    mse_hit: int
    found_truth: bool
    final_mse: float
    noisy_fraction: float


def run_finite(
    seed: int,
    *,
    birth: str,
    acquisition: str,
    information_mode: str,
    features: int,
    actions: int,
    order: int,
    slots: int,
    memory_capacity: int,
    max_steps: int,
    birth_interval: int,
    mse_threshold: float,
    explore_beta: float,
    smc_moves: int,
) -> RunResult:
    world = HypothesisBirthWorld(
        seed,
        features=features,
        actions=actions,
        order=order,
    )
    population = FiniteExplanationPopulation(
        world,
        seed,
        slots=slots,
        order=order,
        memory_capacity=memory_capacity,
    )

    structure_hit = max_steps + 1
    mse_hit = max_steps + 1
    noisy = 0

    for t in range(max_steps):
        action = population.choose_action(
            acquisition,
            information_mode=information_mode,
            explore_beta=explore_beta,
        )
        if action in world.noisy_actions:
            noisy += 1
        y = world.observe(action)
        population.observe(action, y)

        if (t + 1) % birth_interval == 0:
            if birth == "residual":
                population.residual_birth()
            elif birth == "beam":
                population.beam_birth()
            elif birth == "smc":
                population.smc_rejuvenate(smc_moves)
            elif birth != "none":
                raise ValueError(birth)

        if (
            structure_hit == max_steps + 1
            and population.contains_truth()
        ):
            structure_hit = t + 1
        if (
            mse_hit == max_steps + 1
            and population.predictive_mse() < mse_threshold
        ):
            mse_hit = t + 1

    return RunResult(
        structure_hit=structure_hit,
        mse_hit=mse_hit,
        found_truth=population.contains_truth(),
        final_mse=population.predictive_mse(),
        noisy_fraction=noisy / max_steps,
    )


def run_catalog(
    seed: int,
    *,
    acquisition: str,
    information_mode: str,
    features: int,
    actions: int,
    order: int,
    max_steps: int,
    mse_threshold: float,
) -> RunResult:
    world = HypothesisBirthWorld(
        seed,
        features=features,
        actions=actions,
        order=order,
    )
    supports = list(
        itertools.combinations(range(features), order)
    )
    prediction = np.stack(
        [
            support_prediction(world.contribution, support)
            for support in supports
        ],
        axis=0,
    )
    true_index = supports.index(world.true_support)
    logw = np.zeros(len(supports), dtype=float)
    action_count = np.zeros(actions, dtype=int)
    rng = np.random.default_rng(seed + 4_000_001)
    structure_hit = max_steps + 1
    mse_hit = max_steps + 1
    noisy = 0

    info_fn = (
        exact_information_gain
        if information_mode == "exact"
        else gaussian_information_gain
    )

    for t in range(max_steps):
        posterior = softmax(logw)
        if acquisition == "random":
            action = int(rng.integers(actions))
        else:
            score = np.asarray(
                [
                    info_fn(
                        posterior,
                        prediction[:, a],
                        float(world.sigma[a]),
                    )
                    for a in range(actions)
                ]
            )
            action = int(np.argmax(score))

        if action in world.noisy_actions:
            noisy += 1
        y = world.observe(action)
        sigma = float(world.sigma[action])
        logw += -0.5 * (
            (y - prediction[:, action]) / sigma
        ) ** 2
        action_count[action] += 1

        posterior = softmax(logw)
        if (
            structure_hit == max_steps + 1
            and posterior[true_index] >= 0.95
        ):
            structure_hit = t + 1
        mse = float(
            np.mean(
                (posterior @ prediction - world.true_mean) ** 2
            )
        )
        if mse_hit == max_steps + 1 and mse < mse_threshold:
            mse_hit = t + 1

    posterior = softmax(logw)
    final_mse = float(
        np.mean((posterior @ prediction - world.true_mean) ** 2)
    )
    return RunResult(
        structure_hit=structure_hit,
        mse_hit=mse_hit,
        found_truth=bool(posterior[true_index] >= 0.95),
        final_mse=final_mse,
        noisy_fraction=noisy / max_steps,
    )


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def report(
    title: str,
    results: dict[str, list[RunResult]],
    *,
    features: int,
    order: int,
    slots: int,
    memory_capacity: int,
) -> None:
    catalog_size = math.comb(features, order)
    print(title)
    print(
        f"grammar features={features}, hidden order={order}, "
        f"catalog size={catalog_size}, finite slots={slots}, H={memory_capacity}"
    )
    print(
        f"{'method':24s} {'struct hit':>10s} {'mse hit':>10s} "
        f"{'found':>8s} {'final mse':>11s} {'noisy':>9s}"
    )
    for name, vals in results.items():
        print(
            f"{name:24s} "
            f"{mean([float(v.structure_hit) for v in vals]):10.2f} "
            f"{mean([float(v.mse_hit) for v in vals]):10.2f} "
            f"{mean([float(v.found_truth) for v in vals]):8.3f} "
            f"{mean([v.final_mse for v in vals]):11.5f} "
            f"{mean([v.noisy_fraction for v in vals]):9.3f}"
        )
    print()


def run_gate(args: argparse.Namespace) -> None:
    # DYN8A: harder combinatorial catalog, cheap noise-aware information score.
    hard_methods = {
        "residual_info": ("residual", "info"),
        "residual_hybrid": ("residual", "hybrid"),
        "residual_random": ("residual", "random"),
        "beam_info": ("beam", "info"),
        "smc_info": ("smc", "info"),
        "smc_hybrid": ("smc", "hybrid"),
        "fixed_info": ("none", "info"),
    }
    hard: dict[str, list[RunResult]] = {
        name: [] for name in hard_methods
    }
    hard["catalog_info"] = []
    hard["catalog_random"] = []

    for seed in range(args.seeds):
        for name, (birth, acquisition) in hard_methods.items():
            hard[name].append(
                run_finite(
                    seed,
                    birth=birth,
                    acquisition=acquisition,
                    information_mode="gaussian",
                    features=args.features,
                    actions=args.actions,
                    order=args.order,
                    slots=args.slots,
                    memory_capacity=args.memory,
                    max_steps=args.max_steps,
                    birth_interval=args.birth_interval,
                    mse_threshold=args.mse_threshold,
                    explore_beta=args.explore_beta,
                    smc_moves=args.smc_moves,
                )
            )
        hard["catalog_info"].append(
            run_catalog(
                seed,
                acquisition="info",
                information_mode="gaussian",
                features=args.features,
                actions=args.actions,
                order=args.order,
                max_steps=args.max_steps,
                mse_threshold=args.mse_threshold,
            )
        )
        hard["catalog_random"].append(
            run_catalog(
                seed,
                acquisition="random",
                information_mode="gaussian",
                features=args.features,
                actions=args.actions,
                order=args.order,
                max_steps=args.max_steps,
                mse_threshold=args.mse_threshold,
            )
        )

    report(
        "DYN8A — finite evolving explanations, hard combinatorial world",
        hard,
        features=args.features,
        order=args.order,
        slots=args.slots,
        memory_capacity=args.memory,
    )

    # DYN8B: smaller catalog so DYN7's exact EIG can be used for everybody.
    exact_features = args.exact_features
    exact_methods = {
        "residual_exact": ("residual", "info"),
        "residual_random": ("residual", "random"),
        "beam_exact": ("beam", "info"),
        "smc_exact": ("smc", "info"),
        "fixed_exact": ("none", "info"),
    }
    exact: dict[str, list[RunResult]] = {
        name: [] for name in exact_methods
    }
    exact["catalog_exact"] = []
    exact["catalog_random"] = []

    for seed in range(args.exact_seeds):
        for name, (birth, acquisition) in exact_methods.items():
            exact[name].append(
                run_finite(
                    seed,
                    birth=birth,
                    acquisition=acquisition,
                    information_mode="exact",
                    features=exact_features,
                    actions=args.actions,
                    order=args.order,
                    slots=args.slots,
                    memory_capacity=args.memory,
                    max_steps=args.exact_max_steps,
                    birth_interval=args.exact_birth_interval,
                    mse_threshold=args.mse_threshold,
                    explore_beta=args.explore_beta,
                    smc_moves=args.smc_moves,
                )
            )
        exact["catalog_exact"].append(
            run_catalog(
                seed,
                acquisition="info",
                information_mode="exact",
                features=exact_features,
                actions=args.actions,
                order=args.order,
                max_steps=args.exact_max_steps,
                mse_threshold=args.mse_threshold,
            )
        )
        exact["catalog_random"].append(
            run_catalog(
                seed,
                acquisition="random",
                information_mode="exact",
                features=exact_features,
                actions=args.actions,
                order=args.order,
                max_steps=args.exact_max_steps,
                mse_threshold=args.mse_threshold,
            )
        )

    report(
        "DYN8B — exact DYN7 information gain on a smaller unknown catalog",
        exact,
        features=exact_features,
        order=args.order,
        slots=args.slots,
        memory_capacity=args.memory,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--features", type=int, default=14)
    parser.add_argument("--exact-features", type=int, default=10)
    parser.add_argument("--exact-seeds", type=int, default=24)
    parser.add_argument("--actions", type=int, default=21)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--memory", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--exact-max-steps", type=int, default=30)
    parser.add_argument("--birth-interval", type=int, default=4)
    parser.add_argument("--exact-birth-interval", type=int, default=3)
    parser.add_argument("--mse-threshold", type=float, default=0.05)
    parser.add_argument("--explore-beta", type=float, default=0.03)
    parser.add_argument("--smc-moves", type=int, default=3)
    args = parser.parse_args()
    run_gate(args)


if __name__ == "__main__":
    main()
