from __future__ import annotations

import argparse
import numpy as np

from dyn3_complementary_timescales import (
    EpisodicFIFO,
    LinearCortex,
    SurpriseMemory,
    make_world,
)


class RandomEvictMemory:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.ids = np.full(capacity, -1, dtype=int)
        self.y = np.zeros(capacity, dtype=int)
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0

    def write(self, item_id: int, z: np.ndarray, y: int) -> None:
        empty = np.flatnonzero(self.ids < 0)
        if len(empty):
            i = int(empty[0])
        else:
            i = int(self.rng.integers(0, self.capacity))
        self.ids[i] = item_id
        self.y[i] = y
        self.n_seen += 1

    def lookup(self, item_id: int):
        hits = np.flatnonzero(self.ids == item_id)
        if len(hits) == 0:
            return None
        return int(self.y[int(hits[-1])])


class OracleExceptionMemory:
    """Unfair ceiling: told the synthetic exception flag."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.ids = np.full(capacity, -1, dtype=int)
        self.y = np.zeros(capacity, dtype=int)
        self.ptr = 0

    def write(self, item_id: int, z: np.ndarray, y: int, is_exception: bool) -> None:
        if not is_exception:
            return
        self.ids[self.ptr] = item_id
        self.y[self.ptr] = y
        self.ptr = (self.ptr + 1) % self.capacity

    def lookup(self, item_id: int):
        hits = np.flatnonzero(self.ids == item_id)
        if len(hits) == 0:
            return None
        return int(self.y[int(hits[-1])])


def memory_ids(memory) -> np.ndarray:
    return memory.ids[memory.ids >= 0]


def run_one(world, seed: int, capacity: int, policy: str, delay: int = 800) -> dict:
    cortex = LinearCortex(world.z.shape[1], lr=0.004)
    if policy == "fifo":
        memory = EpisodicFIFO(capacity)
    elif policy == "random_evict":
        memory = RandomEvictMemory(capacity, seed + 100000)
    elif policy == "surprise":
        memory = SurpriseMemory(capacity, world.z.shape[1])
    elif policy == "oracle_exception":
        memory = OracleExceptionMemory(capacity)
    else:
        raise ValueError(policy)

    exc_ok = []
    reg_ok = []
    all_ok = []

    for t in range(len(world.y)):
        j = t - delay
        if j >= 500:
            truth = int(world.y[j])
            p = memory.lookup(j)
            if p is None:
                p = cortex.predict(world.z[j])
            ok = float(p == truth)
            all_ok.append(ok)
            (exc_ok if world.is_exception[j] else reg_ok).append(ok)

        z = world.z[t]
        y = int(world.y[t])
        cortex.update(z, y)

        if policy == "surprise":
            if t % 25 == 0:
                memory.refresh_priorities(cortex, t)
            margin = y * np.tanh(cortex.score(z))
            priority = 0.5 * (1.0 - margin)
            memory.write(t, z, y, priority=priority, now=t)
        elif policy == "oracle_exception":
            memory.write(t, z, y, bool(world.is_exception[t]))
        else:
            memory.write(t, z, y)

    ids = memory_ids(memory)
    exc_frac = float(np.mean(world.is_exception[ids])) if len(ids) else float("nan")
    return {
        "long_all": float(np.mean(all_ok)),
        "long_regular": float(np.mean(reg_ok)),
        "long_exception": float(np.mean(exc_ok)),
        "final_exception_fraction_in_memory": exc_frac,
    }


def run_seed(seed: int, steps: int, capacities: list[int]) -> dict:
    world = make_world(steps, seed)
    policies = ["fifo", "random_evict", "surprise", "oracle_exception"]
    out = {}
    for c in capacities:
        out[str(c)] = {p: run_one(world, seed, c, p) for p in policies}
    return out


def run_many(seeds: int, steps: int, capacities: list[int]) -> dict:
    rows = [run_seed(s, steps, capacities) for s in range(seeds)]
    policies = ["fifo", "random_evict", "surprise", "oracle_exception"]
    metrics = [
        "long_all",
        "long_regular",
        "long_exception",
        "final_exception_fraction_in_memory",
    ]
    summary = {}
    for c in capacities:
        ck = str(c)
        summary[ck] = {}
        for p in policies:
            summary[ck][p] = {}
            for m in metrics:
                x = np.asarray([r[ck][p][m] for r in rows], dtype=float)
                summary[ck][p][m] = {
                    "mean": float(np.nanmean(x)),
                    "std": float(np.nanstd(x)),
                }
    return {
        "n_seeds": seeds,
        "steps": steps,
        "capacities": capacities,
        "summary": summary,
        "per_seed": rows,
    }


def print_summary(result: dict) -> None:
    print("\n=== DYN3C: SLOW MODEL -> FAST MEMORY ALLOCATION ===")
    print(f"seeds={result['n_seeds']} steps={result['steps']} long_delay=800")
    for c in result["capacities"]:
        print(f"\ncapacity={c}")
        print("policy".ljust(20) + " long_all   long_reg   long_exc   mem_exc_frac")
        for p, m in result["summary"][str(c)].items():
            print(
                p.ljust(20)
                + f" {m['long_all']['mean']:8.4f}"
                + f" {m['long_regular']['mean']:10.4f}"
                + f" {m['long_exception']['mean']:10.4f}"
                + f" {m['final_exception_fraction_in_memory']['mean']:13.4f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--capacities", type=int, nargs="+", default=[32, 64, 128, 256])
    args = ap.parse_args()
    result = run_many(args.seeds, args.steps, args.capacities)
    print_summary(result)


if __name__ == "__main__":
    main()
