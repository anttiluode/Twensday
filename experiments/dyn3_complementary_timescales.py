from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np


def sign01(v: float) -> int:
    return 1 if v >= 0 else -1


@dataclass
class World:
    z: np.ndarray
    y: np.ndarray
    schema_y: np.ndarray
    is_exception: np.ndarray


def make_world(
    steps: int,
    seed: int,
    dim: int = 16,
    exception_rate: float = 0.12,
) -> World:
    """Continuous stream with reusable structure plus one-shot deviations.

    Each item has a semantic vector z. A fixed hidden linear schema determines
    its ordinary label. A minority of items are idiosyncratic exceptions whose
    exact label cannot be inferred from z alone.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=dim)
    w /= np.linalg.norm(w) + 1e-12

    z = rng.normal(size=(steps, dim))
    # Give the schema a modest margin so there is a learnable regularity rather
    # than labels dominated by points arbitrarily close to the boundary.
    raw = z @ w + 0.15 * rng.normal(size=steps)
    schema_y = np.where(raw >= 0.0, 1, -1).astype(int)
    is_exception = rng.random(steps) < exception_rate
    y = schema_y.copy()
    y[is_exception] *= -1
    return World(z=z, y=y, schema_y=schema_y, is_exception=is_exception)


class LinearCortex:
    """Slow reusable predictor over semantic features."""

    def __init__(self, dim: int, lr: float) -> None:
        self.w = np.zeros(dim, dtype=float)
        self.lr = lr

    def score(self, z: np.ndarray) -> float:
        return float(self.w @ z)

    def predict(self, z: np.ndarray) -> int:
        return sign01(self.score(z))

    def update(self, z: np.ndarray, target: int, scale: float = 1.0) -> None:
        # Smooth signed error keeps this NumPy-only and deliberately boring.
        yhat = np.tanh(self.score(z))
        self.w += self.lr * scale * (target - yhat) * z
        n = np.linalg.norm(self.w)
        if n > 8.0:
            self.w *= 8.0 / n


class EpisodicFIFO:
    """Finite fast-write exact-address store. This is intentionally RAG-like."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.ids = np.full(capacity, -1, dtype=int)
        self.z = None
        self.y = np.zeros(capacity, dtype=int)
        self.ptr = 0
        self.size = 0

    def _ensure(self, dim: int) -> None:
        if self.z is None:
            self.z = np.zeros((self.capacity, dim), dtype=float)

    def write(self, item_id: int, z: np.ndarray, y: int) -> None:
        self._ensure(len(z))
        self.ids[self.ptr] = item_id
        self.z[self.ptr] = z
        self.y[self.ptr] = y
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def lookup(self, item_id: int):
        if self.size == 0:
            return None
        hits = np.flatnonzero(self.ids == item_id)
        if len(hits) == 0:
            return None
        return int(self.y[int(hits[-1])])

    def sample(self, rng: np.random.Generator, n: int):
        if self.size == 0 or self.z is None:
            return []
        valid = np.flatnonzero(self.ids >= 0)
        idx = rng.choice(valid, size=min(n, len(valid)), replace=False)
        return [(self.z[i].copy(), int(self.y[i])) for i in idx]

    def clear(self) -> None:
        self.ids[:] = -1
        self.y[:] = 0
        if self.z is not None:
            self.z[:] = 0.0
        self.ptr = 0
        self.size = 0


class SurpriseMemory:
    """Fast store whose finite slots are reprioritized by slow cortical surprise.

    This is the first tiny M->H feedback test: as cortex learns an item well,
    that episode becomes cheaper to evict. Persistent exceptions remain costly.
    """

    def __init__(self, capacity: int, dim: int) -> None:
        self.capacity = capacity
        self.ids = np.full(capacity, -1, dtype=int)
        self.z = np.zeros((capacity, dim), dtype=float)
        self.y = np.zeros(capacity, dtype=int)
        self.priority = np.zeros(capacity, dtype=float)
        self.age = np.zeros(capacity, dtype=int)
        self.size = 0

    def lookup(self, item_id: int):
        hits = np.flatnonzero(self.ids == item_id)
        if len(hits) == 0:
            return None
        return int(self.y[int(hits[-1])])

    def refresh_priorities(self, cortex: LinearCortex, now: int) -> None:
        valid = np.flatnonzero(self.ids >= 0)
        if len(valid) == 0:
            return
        for i in valid:
            margin = self.y[i] * np.tanh(cortex.score(self.z[i]))
            surprise = 0.5 * (1.0 - margin)
            # A weak age pressure prevents ancient early-learning mistakes from
            # becoming immortal once the cortex has learned them.
            age_factor = np.exp(-(now - self.age[i]) / 2500.0)
            self.priority[i] = 0.85 * self.priority[i] + 0.15 * surprise * age_factor

    def write(self, item_id: int, z: np.ndarray, y: int, priority: float, now: int) -> None:
        empty = np.flatnonzero(self.ids < 0)
        if len(empty):
            i = int(empty[0])
            self.size += 1
        else:
            i = int(np.argmin(self.priority))
            if priority <= self.priority[i]:
                return
        self.ids[i] = item_id
        self.z[i] = z
        self.y[i] = y
        self.priority[i] = priority
        self.age[i] = now

    def sample(self, rng: np.random.Generator, n: int):
        valid = np.flatnonzero(self.ids >= 0)
        if len(valid) == 0:
            return []
        idx = rng.choice(valid, size=min(n, len(valid)), replace=False)
        return [(self.z[i].copy(), int(self.y[i])) for i in idx]

    def clear(self) -> None:
        self.ids[:] = -1
        self.y[:] = 0
        self.priority[:] = 0.0
        self.age[:] = 0
        self.z[:] = 0.0
        self.size = 0


@dataclass
class Metrics:
    novel: list[float]
    short_all: list[float]
    short_exc: list[float]
    long_all: list[float]
    long_exc: list[float]
    long_regular: list[float]
    post_erase_regular: list[float]
    post_erase_exc: list[float]

    @classmethod
    def make(cls):
        return cls([], [], [], [], [], [], [], [])

    def add_recall(self, pred: int, truth: int, exc: bool, long: bool, erased: bool = False):
        ok = float(pred == truth)
        if erased:
            (self.post_erase_exc if exc else self.post_erase_regular).append(ok)
            return
        if long:
            self.long_all.append(ok)
            (self.long_exc if exc else self.long_regular).append(ok)
        else:
            self.short_all.append(ok)
            if exc:
                self.short_exc.append(ok)


def mean(x: list[float]) -> float:
    return float(np.mean(x)) if x else float("nan")


def run_variant(
    world: World,
    seed: int,
    variant: str,
    capacity: int = 256,
    short_delay: int = 40,
    long_delay: int = 800,
    erase_at: int = 6500,
) -> dict:
    rng = np.random.default_rng(seed + 10000)
    dim = world.z.shape[1]

    if variant == "fast_cortex":
        cortex = LinearCortex(dim, lr=0.055)
    else:
        cortex = LinearCortex(dim, lr=0.004)

    memory = None
    if variant in {"fifo_replay", "fifo_no_replay", "hpc_only"}:
        memory = EpisodicFIFO(capacity)
    elif variant in {"surprise_replay", "surprise_no_replay"}:
        memory = SurpriseMemory(capacity, dim)

    metrics = Metrics.make()
    steps = len(world.y)

    for t in range(steps):
        z = world.z[t]
        y = int(world.y[t])

        # New-item prediction happens before the answer is seen: schema/generalization.
        if variant == "hpc_only":
            novel_pred = 1  # unseen id has no episodic answer
        else:
            novel_pred = cortex.predict(z)
        if t >= 500:
            metrics.novel.append(float(novel_pred == y))

        # Recall probes do not train the model. They ask whether an earlier exact
        # event is still recoverable at two very different lifetimes.
        for delay, is_long in [(short_delay, False), (long_delay, True)]:
            j = t - delay
            if j >= 0:
                truth = int(world.y[j])
                exc = bool(world.is_exception[j])
                pred = None if memory is None else memory.lookup(j)
                if pred is None:
                    pred = 1 if variant == "hpc_only" else cortex.predict(world.z[j])
                if t >= 500:
                    metrics.add_recall(pred, truth, exc, is_long)

        # After erase_at, run a second long-delay probe specifically to ask what
        # remains when the fast store is absent. We do not stop slow learning.
        if t == erase_at and memory is not None:
            memory.clear()
        if t > erase_at and t - long_delay >= 0:
            j = t - long_delay
            if j < erase_at:  # item was encoded before the lesion
                truth = int(world.y[j])
                exc = bool(world.is_exception[j])
                pred = 1 if variant == "hpc_only" else cortex.predict(world.z[j])
                metrics.add_recall(pred, truth, exc, long=True, erased=True)

        # Reveal the current label and update/write.
        if variant != "hpc_only":
            cortex.update(z, y)

        if memory is not None:
            if isinstance(memory, SurpriseMemory):
                if t % 25 == 0:
                    memory.refresh_priorities(cortex, t)
                margin = y * np.tanh(cortex.score(z))
                priority = 0.5 * (1.0 - margin)
                memory.write(t, z, y, priority=priority, now=t)
            else:
                memory.write(t, z, y)

        # Replay or the same-compute boring attacker.
        if variant in {"fifo_replay", "surprise_replay"} and memory is not None:
            for rz, ry in memory.sample(rng, 6):
                cortex.update(rz, ry, scale=0.45)
        elif variant == "repeat_current":
            for _ in range(6):
                cortex.update(z, y, scale=0.45)

    return {
        "novel_accuracy": mean(metrics.novel),
        "short_recall": mean(metrics.short_all),
        "short_exception_recall": mean(metrics.short_exc),
        "long_recall": mean(metrics.long_all),
        "long_regular_recall": mean(metrics.long_regular),
        "long_exception_recall": mean(metrics.long_exc),
        "post_erase_regular_recall": mean(metrics.post_erase_regular),
        "post_erase_exception_recall": mean(metrics.post_erase_exc),
    }


def run_seed(seed: int, steps: int) -> dict:
    world = make_world(steps, seed)
    variants = [
        "slow_cortex",
        "fast_cortex",
        "repeat_current",
        "hpc_only",
        "fifo_no_replay",
        "fifo_replay",
        "surprise_no_replay",
        "surprise_replay",
    ]
    return {v: run_variant(world, seed, v) for v in variants}


def run_many(seeds: int, steps: int) -> dict:
    rows = [run_seed(s, steps) for s in range(seeds)]
    variants = list(rows[0].keys())
    metrics = list(rows[0][variants[0]].keys())
    summary = {}
    for v in variants:
        summary[v] = {}
        for m in metrics:
            a = np.asarray([r[v][m] for r in rows], dtype=float)
            summary[v][m] = {
                "mean": float(np.nanmean(a)),
                "std": float(np.nanstd(a)),
            }
    return {"n_seeds": seeds, "steps": steps, "summary": summary, "per_seed": rows}


def print_summary(result: dict) -> None:
    print("\n=== DYN3: COMPLEMENTARY TIMESCALES ===")
    print(f"seeds={result['n_seeds']} steps={result['steps']}")
    cols = [
        "novel_accuracy",
        "short_exception_recall",
        "long_regular_recall",
        "long_exception_recall",
        "post_erase_regular_recall",
        "post_erase_exception_recall",
    ]
    print("method".ljust(23) + " ".join(c[:11].rjust(12) for c in cols))
    for v, metrics in result["summary"].items():
        vals = [metrics[c]["mean"] for c in cols]
        print(v.ljust(23) + " ".join(f"{x:12.4f}" for x in vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=9000)
    args = ap.parse_args()
    result = run_many(args.seeds, args.steps)
    print_summary(result)


if __name__ == "__main__":
    main()
