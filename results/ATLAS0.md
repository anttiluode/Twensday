# Atlas 0 — linear reachability: basis geometry already chooses the matrix family

Development receipt, not confirmatory evidence.

## Question

Twensday begins by stripping the parent architecture to the Gate-15 linear case:

```text
fixed dense signed basis R
nonnegative conserved structural mass m

w_eff = m^T R
```

For six points, stack the effective rows into a `6 x 6` matrix `W_eff`.

The first question is not whether the growth rule is clever. It is:

> **What matrices are even reachable when every row must live inside the convex hull of a fixed dense random basis?**

To separate representation from optimization, every target gets two learners:

1. the repo's positive-only conserved growth;
2. a projected-gradient attacker that finds the best row reachable inside exactly the same reserve-constrained convex hull.

An unconstrained linear model could represent every target matrix in this experiment exactly, so any hull error is purely representational.

Five seeds, 36 dense random candidate directions per row, 6 inputs / 6 outputs, reserve `0.001`.

## Results

| target family | grown relative Frobenius error | best-hull error | held-out NMSE | mean effective occupied features |
|---|---:|---:|---:|---:|
| reachable sparse | `0.0125 ± 0.0093` | `0.0006 ± 0.0012` | `0.00025 ± 0.00027` | `16.70 ± 2.75` |
| selector / identity | `0.1618 ± 0.0198` | `0.1352 ± 0.0228` | `0.0266 ± 0.0062` | `11.60 ± 1.03` |
| cyclic ring / permutation | `0.1860 ± 0.0441` | `0.1611 ± 0.0528` | `0.0365 ± 0.0172` | `11.33 ± 1.15` |
| dense orthogonal mix | `0.0705 ± 0.0452` | `0.0589 ± 0.0502` | `0.00698 ± 0.00901` | `12.18 ± 1.88` |
| rank-2 dense target | `0.0182 ± 0.0227` | approximately `0` | `0.00084 ± 0.00150` | `26.14 ± 1.20` |

## First answer: the basis geometry is already a computational prior

The basis rows are deliberately dense: every raw input has a substantial positive or negative coefficient in every candidate direction.

That makes dense mixed target rows relatively natural members of their convex hull.

It makes clean coordinate selectors less natural.

Thus the surprising ordering:

```text
dense orthogonal mix
    easier

selector / one-hot routing
    harder

cyclic one-hot ring
    harder still
```

is largely reproduced by the best-reachable hull attacker.

So this is not evidence that the positive-growth rule "likes mixing" as an optimization strategy.

It is evidence that:

> **the fixed candidate basis determines which operator shapes are structurally cheap or expensive before learning starts.**

That is immediately relevant to the old dendrite/morphology intuition. A physical implementation would not merely provide more capacity; its available local transfer directions would define a prior over reachable effective matrices.

## Second answer: structural sparsity and operator sparsity are different

The `reachable_sparse` targets were constructed from only four basis directions per row, so they are exactly inside the representation (up to the reserve floor).

Yet positive growth finishes with an average effective occupancy of about:

```text
16.7 / 36 features
```

while still achieving very low output error.

So the growth rule can reproduce a sparse target **operator** using a much less sparse structural allocation.

A pretty sparse `M` is therefore not required for a simple effective computation, and conversely a sparse task does not guarantee sparse internal structure.

## Third answer: low operator rank does not imply low structural occupancy

The rank-2 targets are essentially exactly reachable by the convex-hull attacker.

Positive growth also fits them well:

```text
relative matrix error  ~0.018
held-out NMSE          ~0.00084
```

But the average structural occupancy is the largest of all five families:

```text
~26 / 36 features
```

The grown numerical ranks across the five seeds are:

```text
2, 4, 3, 2, 3
```

The small extra singular directions are approximation residue, while the main target is rank 2.

So:

> **low-rank effective computation can be implemented by diffuse structural mass over many generic coordinates.**

Again, structural complexity and effective matrix complexity are not interchangeable measurements.

## Representation gap versus growth gap

This distinction should become standard in Twensday.

```text
target W*
   ↓
[representation]
best reachable W_hull
   ↓
[growth dynamics]
actual W_grown
```

For selectors and ring matrices, much of the error already exists at `W_hull`.

That is a **basis/capacity problem**.

The remaining difference between `W_grown` and `W_hull` is an **optimization/growth-rule problem**.

Do not blame one on the other.

## What this does not show

- dense random bases are universally good;
- biological dendrites correspond to these random directions;
- orthogonal matrices are intrinsically easier for structural growth;
- the recurrent ring task has been learned dynamically here (it has not — this is only a target operator shape);
- sparse matrices are undesirable;
- the positive-growth rule is competitive with ordinary optimization.

## Next atlas

The immediate next useful experiment is a **basis-geometry sweep**.

Compare at least:

```text
dense random signed basis
axis-enriched basis
orthogonal basis
paired +/- basis
sparse random basis
learned / adapted basis (later)
```

and ask how the same target families move in and out of the reachable hull.

That would turn the old morphology question into a precise computational one:

> **What geometry of available local directions gives a useful prior over the matrices a growing point may need?**

Committed summary: `results/atlas0_summary.json`.
The experiment runner writes the full per-seed matrices to `results/atlas0_linear_reachability.json` when run.
