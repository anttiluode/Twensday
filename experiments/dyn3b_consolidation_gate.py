from __future__ import annotations

import argparse
import numpy as np

from dyn3_complementary_timescales import (
    LinearCortex,
    Metrics,
    SurpriseMemory,
    make_world,
    mean,
)


def run_policy(
    world,
    seed: int,
    policy: str,
    capacity: int = 256,
    short_delay: int = 40,
    long_delay: int = 800,
    erase_at: int = 6500,
) -> dict:
    rng = np.random.default_rng(seed + 50000)
    cortex = LinearCortex(world.z.shape[1], lr=0.004)
    memory = SurpriseMemory(capacity, world.z.shape[1])
    metrics = Metrics.make()

    replay_attempted = 0
    replay_used = 0

    for t in range(len(world.y)):
        z = world.z[t]
        y = int(world.y[t])

        pred = cortex.predict(z)
        if t >= 500:
            metrics.novel.append(float(pred == y))

        for delay, is_long in [(short_delay, False), (long_delay, True)]:
            j = t - delay
            if j >= 0:
                truth = int(world.y[j])
                exc = bool(world.is_exception[j])
                mp = memory.lookup(j)
                rp = mp if mp is not None else cortex.predict(world.z[j])
                if t >= 500:
                    metrics.add_recall(rp, truth, exc, is_long)

        if t == erase_at:
            memory.clear()
        if t > erase_at and t - long_delay >= 0:
            j = t - long_delay
            if j < erase_at:
                truth = int(world.y[j])
                exc = bool(world.is_exception[j])
                metrics.add_recall(
                    cortex.predict(world.z[j]),
                    truth,
                    exc,
                    long=True,
                    erased=True,
                )

        # Direct slow learning from the current experience.
        cortex.update(z, y)

        if t % 25 == 0:
            memory.refresh_priorities(cortex, t)
        margin_now = y * np.tanh(cortex.score(z))
        priority = 0.5 * (1.0 - margin_now)
        memory.write(t, z, y, priority=priority, now=t)

        # Replay/consolidation is deliberately a separate decision from retain.
        if policy != "no_replay":
            candidates = memory.sample(rng, 18)
            used = 0
            for rz, ry in candidates:
                if used >= 6:
                    break
                margin = ry * np.tanh(cortex.score(rz))
                replay_attempted += 1

                if policy == "uniform":
                    accept = True
                elif policy == "congruent":
                    accept = margin > 0.0
                elif policy == "moderate_congruent":
                    accept = 0.0 < margin < 0.80
                elif policy == "conflicting":
                    accept = margin < 0.0
                elif policy == "soft_congruent":
                    # Same-information stochastic gate. Negative-margin events
                    # are unlikely to teach cortex; positive uncertain events
                    # are most likely. Very certain events still replay rarely.
                    p = 1.0 / (1.0 + np.exp(-6.0 * margin))
                    p *= np.exp(-max(margin, 0.0) / 1.4)
                    accept = rng.random() < p
                else:
                    raise ValueError(policy)

                if accept:
                    cortex.update(rz, ry, scale=0.45)
                    replay_used += 1
                    used += 1

    return {
        "novel_accuracy": mean(metrics.novel),
        "short_exception_recall": mean(metrics.short_exc),
        "long_regular_recall": mean(metrics.long_regular),
        "long_exception_recall": mean(metrics.long_exc),
        "post_erase_regular_recall": mean(metrics.post_erase_regular),
        "post_erase_exception_recall": mean(metrics.post_erase_exc),
        "replay_acceptance": replay_used / max(replay_attempted, 1),
    }


def run_seed(seed: int, steps: int) -> dict:
    world = make_world(steps, seed)
    policies = [
        "no_replay",
        "uniform",
        "congruent",
        "moderate_congruent",
        "soft_congruent",
        "conflicting",
    ]
    return {p: run_policy(world, seed, p) for p in policies}


def run_many(seeds: int, steps: int) -> dict:
    rows = [run_seed(s, steps) for s in range(seeds)]
    policies = list(rows[0])
    metrics = list(rows[0][policies[0]])
    summary = {}
    for p in policies:
        summary[p] = {}
        for m in metrics:
            x = np.asarray([r[p][m] for r in rows], dtype=float)
            summary[p][m] = {
                "mean": float(np.nanmean(x)),
                "std": float(np.nanstd(x)),
            }
    return {"n_seeds": seeds, "steps": steps, "summary": summary, "per_seed": rows}


def print_summary(result: dict) -> None:
    print("\n=== DYN3B: RETAIN != CONSOLIDATE ===")
    print(f"seeds={result['n_seeds']} steps={result['steps']}")
    keys = [
        "novel_accuracy",
        "short_exception_recall",
        "long_regular_recall",
        "long_exception_recall",
        "post_erase_regular_recall",
        "replay_acceptance",
    ]
    print("policy".ljust(23) + " ".join(k[:10].rjust(11) for k in keys))
    for p, m in result["summary"].items():
        vals = [m[k]["mean"] for k in keys]
        print(p.ljust(23) + " ".join(f"{v:11.4f}" for v in vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=9000)
    args = ap.parse_args()
    result = run_many(args.seeds, args.steps)
    print_summary(result)


if __name__ == "__main__":
    main()
