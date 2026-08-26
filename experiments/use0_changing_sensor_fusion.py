from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def project_growth_budget(values: np.ndarray, reserve: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    free = 1.0 - len(v) * float(reserve)
    if reserve < 0.0 or free < 0.0:
        raise ValueError("reserve incompatible with unit budget")
    excess = np.maximum(v - reserve, 0.0)
    if excess.sum() < 1e-12:
        return np.full(len(v), 1.0 / len(v))
    return reserve + free * excess / excess.sum()


def _project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(np.asarray(v, dtype=float))[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(u) + 1)
    keep = u - css / ind > 0.0
    if not np.any(keep):
        return np.full(len(v), 1.0 / len(v))
    rho = ind[keep][-1]
    theta = css[keep][-1] / rho
    return np.maximum(v - theta, 0.0)


def project_with_reserve(values: np.ndarray, reserve: float) -> np.ndarray:
    free = 1.0 - len(values) * float(reserve)
    if reserve < 0.0 or free <= 0.0:
        raise ValueError("reserve incompatible with unit budget")
    return reserve + free * _project_simplex((np.asarray(values) - reserve) / free)


def make_world(seed: int, steps: int = 4800, n_sensors: int = 8, segment: int = 400):
    rng = np.random.default_rng(seed)
    latent = np.zeros(steps)
    innovation = rng.normal(size=steps)
    for t in range(1, steps):
        latent[t] = 0.92 * latent[t - 1] + 0.35 * innovation[t]
    latent /= np.std(latent)

    # Four sensors take turns being highly trustworthy, then return later.
    best = np.asarray([(t // segment) % 4 for t in range(steps)], dtype=int)
    sensors = np.empty((steps, n_sensors), dtype=float)
    for t in range(steps):
        for i in range(n_sensors):
            if i == best[t]:
                sensors[t, i] = latent[t] + 0.08 * rng.normal()
            else:
                gain = 0.35 + 0.15 * (i % 4)
                sensors[t, i] = (
                    gain * latent[t]
                    + 0.70 * rng.normal()
                    + 0.25 * np.sin(0.013 * t + 0.7 * i)
                )
    return latent, sensors, best


def run_twensday(latent, sensors, best, reserve=0.002, growth_rate=12.0):
    n = sensors.shape[1]
    mass = np.full(n, 1.0 / n)
    losses, best_mass = [], []
    for t, y in enumerate(latent):
        h = sensors[t]
        pred = float(mass @ h)
        error = y - pred
        losses.append(error * error)
        best_mass.append(mass[best[t]])

        # Gate-15 / Atlas-0 style local positive consequence signal:
        # existing structural mass gates the candidate's eligibility;
        # only positive evidence grows; normalization supplies retraction.
        impulse = np.maximum(mass * (h * error), 0.0)
        mass = project_growth_budget(mass + growth_rate * impulse, reserve)
    return np.asarray(losses), np.asarray(best_mass)


def run_signed_simplex(latent, sensors, best, reserve=0.01, learning_rate=0.4):
    n = sensors.shape[1]
    mass = np.full(n, 1.0 / n)
    losses, best_mass = [], []
    for t, y in enumerate(latent):
        h = sensors[t]
        pred = float(mass @ h)
        error = y - pred
        losses.append(error * error)
        best_mass.append(mass[best[t]])
        gradient = -error * h
        mass = project_with_reserve(mass - learning_rate * gradient, reserve)
    return np.asarray(losses), np.asarray(best_mass)


def run_fixed_share(latent, sensors, best, eta=12.0, share=0.05):
    """Exponentially weighted experts plus a small fixed share/exploration term."""
    n = sensors.shape[1]
    weight = np.full(n, 1.0 / n)
    losses, best_mass = [], []
    for t, y in enumerate(latent):
        h = sensors[t]
        pred = float(weight @ h)
        error = y - pred
        losses.append(error * error)
        best_mass.append(weight[best[t]])

        individual_loss = np.minimum((h - y) ** 2, 5.0)
        weight *= np.exp(-eta * individual_loss)
        weight /= weight.sum()
        weight = (1.0 - share) * weight + share / n
    return np.asarray(losses), np.asarray(best_mass)


def _post_switch_mean(values, segment: int, width: int) -> float:
    switches = range(segment, len(values), segment)
    return float(np.mean([values[i : i + width].mean() for i in switches]))


def run(seed: int, steps: int = 4800, n_sensors: int = 8, segment: int = 400) -> dict:
    latent, sensors, best = make_world(seed, steps, n_sensors, segment)
    tw_loss, tw_mass = run_twensday(latent, sensors, best)
    sg_loss, sg_mass = run_signed_simplex(latent, sensors, best)
    fs_loss, fs_mass = run_fixed_share(latent, sensors, best)

    uniform = np.mean((sensors.mean(axis=1) - latent) ** 2)
    oracle = np.mean((sensors[np.arange(steps), best] - latent) ** 2)

    return {
        "seed": seed,
        "twensday_mse": float(tw_loss.mean()),
        "signed_simplex_mse": float(sg_loss.mean()),
        "fixed_share_mse": float(fs_loss.mean()),
        "uniform_mse": float(uniform),
        "oracle_mse": float(oracle),
        "twensday_post50_mse": _post_switch_mean(tw_loss, segment, 50),
        "signed_simplex_post50_mse": _post_switch_mean(sg_loss, segment, 50),
        "fixed_share_post50_mse": _post_switch_mean(fs_loss, segment, 50),
        "twensday_post200_mse": _post_switch_mean(tw_loss, segment, 200),
        "signed_simplex_post200_mse": _post_switch_mean(sg_loss, segment, 200),
        "fixed_share_post200_mse": _post_switch_mean(fs_loss, segment, 200),
        "twensday_best_mass_post200": _post_switch_mean(tw_mass, segment, 200),
        "signed_simplex_best_mass_post200": _post_switch_mean(sg_mass, segment, 200),
        "fixed_share_best_mass_post200": _post_switch_mean(fs_mass, segment, 200),
    }


def main() -> None:
    rows = [run(seed) for seed in range(20)]
    keys = [k for k in rows[0] if k != "seed"]
    summary = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in keys
    }
    payload = {"development_note": "parameters explored before this committed receipt", "summary": summary, "per_seed": rows}
    print(json.dumps(payload["summary"], indent=2))
    out = Path(__file__).resolve().parents[1] / "results" / "use0_changing_sensor_fusion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
