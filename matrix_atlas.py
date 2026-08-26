"""Linear operator atlas for Twensday.

This module strips the growing-matrix idea down to its Gate-15 linear core.
Each point owns a fixed dense signed basis R and a nonnegative conserved
structural allocation m. The effective input-space row is

    w_eff = m @ R

Atlas 0 compares the repo's positive-only conserved growth against the best
operator reachable inside exactly the same convex-hull representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

Array = np.ndarray


def dense_signed_basis(seed: int, n_features: int = 36, n_inputs: int = 6) -> Array:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_features, n_inputs))
    magnitudes = rng.uniform(0.5, 1.0, size=(n_features, n_inputs))
    R = signs * magnitudes
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return R


def project_growth_budget(values: Array, reserve: float) -> Array:
    """Parent-repo style positive mass normalization with a reserve floor."""
    v = np.asarray(values, dtype=float)
    free = 1.0 - len(v) * float(reserve)
    if reserve < 0.0 or free < -1e-12:
        raise ValueError("reserve incompatible with unit budget")
    free = max(free, 0.0)
    excess = np.maximum(v - reserve, 0.0)
    if excess.sum() < 1e-12:
        return np.full(len(v), 1.0 / len(v))
    return reserve + free * excess / excess.sum()


def _project_simplex(v: Array) -> Array:
    """Euclidean projection onto {x >= 0, sum x = 1}."""
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    keep = u - css / ind > 0.0
    if not np.any(keep):
        return np.full(len(v), 1.0 / len(v))
    rho = ind[keep][-1]
    theta = css[keep][-1] / rho
    return np.maximum(v - theta, 0.0)


def project_euclidean_with_reserve(values: Array, reserve: float) -> Array:
    """Euclidean projection onto the reserve-constrained probability simplex."""
    v = np.asarray(values, dtype=float)
    free = 1.0 - len(v) * float(reserve)
    if reserve < 0.0 or free <= 0.0:
        raise ValueError("reserve incompatible with unit budget")
    z = (v - reserve) / free
    return reserve + free * _project_simplex(z)


@dataclass
class PositiveGrowthRow:
    basis: Array
    reserve: float = 0.001
    growth_rate: float = 0.25
    score_decay: float = 0.97

    def __post_init__(self) -> None:
        self.basis = np.asarray(self.basis, dtype=float)
        self.mass = np.full(len(self.basis), 1.0 / len(self.basis), dtype=float)
        self.score = np.zeros(len(self.basis), dtype=float)

    @property
    def effective_row(self) -> Array:
        return self.mass @ self.basis

    @property
    def effective_feature_count(self) -> float:
        p = self.mass / self.mass.sum()
        return float(np.exp(-np.sum(p * np.log(p + 1e-15))))

    def fit(self, target_row: Array, steps: int = 1200) -> "PositiveGrowthRow":
        target = np.asarray(target_row, dtype=float)
        for _ in range(int(steps)):
            error = target - self.effective_row
            # Exact expectation for isotropic Gaussian x of the parent's local
            # consequence x eligibility score in the linear case.
            raw = self.mass * (self.basis @ error)
            impulse = np.maximum(raw, 0.0)
            self.score = self.score_decay * self.score + (1.0 - self.score_decay) * impulse
            self.mass = project_growth_budget(
                self.mass + self.growth_rate * self.score,
                self.reserve,
            )
        return self


def best_reachable_row(
    basis: Array,
    target_row: Array,
    reserve: float = 0.001,
    steps: int = 5000,
) -> tuple[Array, Array]:
    """Projected-gradient attacker for the best row in the same convex hull."""
    R = np.asarray(basis, dtype=float)
    target = np.asarray(target_row, dtype=float)
    mass = np.full(len(R), 1.0 / len(R), dtype=float)
    lipschitz = float(np.linalg.norm(R, 2) ** 2)
    lr = 0.9 / max(lipschitz, 1e-12)
    for _ in range(int(steps)):
        gradient = (mass @ R - target) @ R.T
        mass = project_euclidean_with_reserve(mass - lr * gradient, reserve)
    return mass, mass @ R


def _scaled_orthogonal(seed: int, n: int, scale: float = 0.5) -> Array:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return scale * q


def _low_rank(seed: int, n: int, rank: int = 2, scale: float = 0.5) -> Array:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, n))
    spectral = np.linalg.norm(W, 2)
    return W * (scale / max(spectral, 1e-12))


def task_matrix(
    name: str,
    bases: list[Array],
    seed: int,
    reserve: float,
) -> tuple[Array, str]:
    n = len(bases)
    if name == "reachable_sparse":
        rng = np.random.default_rng(seed + 7000)
        rows = []
        for R in bases:
            k = len(R)
            mass = np.full(k, reserve)
            free = 1.0 - k * reserve
            chosen = rng.choice(k, size=4, replace=False)
            weights = rng.uniform(0.2, 1.0, size=4)
            weights /= weights.sum()
            mass[chosen] += free * weights
            rows.append(mass @ R)
        return np.asarray(rows), "target generated inside each row's reachable convex hull"
    if name == "selector":
        return 0.5 * np.eye(n), "scaled identity / one-input selectors"
    if name == "ring":
        W = np.zeros((n, n), dtype=float)
        for i in range(n):
            W[i, (i - 1) % n] = 0.5
        return W, "scaled cyclic permutation; operator-shape probe, not a trained memory task"
    if name == "orthogonal_mix":
        return _scaled_orthogonal(seed + 9000, n), "dense scaled orthogonal mixing matrix"
    if name == "low_rank":
        return _low_rank(seed + 11000, n, rank=2), "rank-2 dense projector/mixer"
    raise ValueError(f"unknown task {name!r}")


def matrix_rank(W: Array, tol: float = 1e-3) -> int:
    s = np.linalg.svd(np.asarray(W), compute_uv=False)
    return int(np.sum(s > tol))


def heldout_nmse(W: Array, target: Array, seed: int, n_samples: int = 4096) -> float:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, target.shape[1]))
    y = X @ target.T
    pred = X @ W.T
    return float(np.mean((pred - y) ** 2) / (np.var(y) + 1e-12))


def run_task(
    task: str,
    seed: int = 0,
    n_points: int = 6,
    n_features: int = 36,
    reserve: float = 0.001,
) -> dict:
    bases = [dense_signed_basis(seed * 1000 + 100 + p, n_features, n_points) for p in range(n_points)]
    target, description = task_matrix(task, bases, seed, reserve)

    grown_rows = []
    hull_rows = []
    masses = []
    occupancies = []
    for p, R in enumerate(bases):
        learner = PositiveGrowthRow(R, reserve=reserve).fit(target[p])
        _, hull = best_reachable_row(R, target[p], reserve=reserve)
        grown_rows.append(learner.effective_row)
        hull_rows.append(hull)
        masses.append(learner.mass.copy())
        occupancies.append(learner.effective_feature_count)

    grown = np.asarray(grown_rows)
    hull = np.asarray(hull_rows)
    target_norm = float(np.linalg.norm(target)) + 1e-12
    s_target = np.linalg.svd(target, compute_uv=False)
    s_grown = np.linalg.svd(grown, compute_uv=False)

    return {
        "task": task,
        "description": description,
        "seed": int(seed),
        "target_relative_fro_error": float(np.linalg.norm(grown - target) / target_norm),
        "hull_relative_fro_error": float(np.linalg.norm(hull - target) / target_norm),
        "growth_to_hull_relative_gap": float(np.linalg.norm(grown - hull) / target_norm),
        "heldout_nmse": heldout_nmse(grown, target, seed + 50000),
        "target_rank": matrix_rank(target),
        "grown_rank": matrix_rank(grown),
        "target_singular_values": s_target.tolist(),
        "grown_singular_values": s_grown.tolist(),
        "mean_effective_feature_count": float(np.mean(occupancies)),
        "mean_max_mass": float(np.mean([m.max() for m in masses])),
        "target_matrix": target.tolist(),
        "grown_matrix": grown.tolist(),
        "hull_matrix": hull.tolist(),
    }


def run_atlas(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict:
    tasks = ("reachable_sparse", "selector", "ring", "orthogonal_mix", "low_rank")
    rows = {task: [run_task(task, seed=s) for s in seeds] for task in tasks}
    summary = {}
    scalar_keys = (
        "target_relative_fro_error",
        "hull_relative_fro_error",
        "growth_to_hull_relative_gap",
        "heldout_nmse",
        "mean_effective_feature_count",
        "mean_max_mass",
    )
    for task, task_rows in rows.items():
        summary[task] = {}
        for key in scalar_keys:
            vals = np.array([float(r[key]) for r in task_rows])
            summary[task][key] = {"mean": float(vals.mean()), "std": float(vals.std())}
        summary[task]["grown_rank_values"] = [int(r["grown_rank"]) for r in task_rows]
        summary[task]["target_rank_values"] = [int(r["target_rank"]) for r in task_rows]
    return {"summary": summary, "per_seed": rows}
