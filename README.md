# Twensday — what can a growing matrix compute?

`Twensday` starts where [`yrotisopeRweN`](https://github.com/anttiluode/yrotisopeRweN) stopped.

The earlier repo spent its time earning small mechanisms one at a time: receiver-relative fit, transient composition, context, delayed credit, finite allocation, continuous state, recurrent traffic, recurrent growth, anonymous return discovery, and finally growth in a generic mixed coordinate basis.

This repo asks a different question:

> **What family of operators does that machinery actually produce?**

Not another neuron metaphor. Not another named learning rule. Inspect the matrices.

## Start here

Open [`index.html`](./index.html).

It is a standalone explanatory browser demo with no dependencies. It shows:

- a diffuse finite structural budget becoming sparse;
- structural mass `M` separately from the effective operator it induces;
- the difference between explicit relation coordinates and a generic signed basis;
- reallocation when utility changes;
- why stable addressability matters;
- the mathematical meaning of the Gate-15 linear case.

The HTML is an **illustration**, not an experimental receipt. The tested gates live in `yrotisopeRweN`.

## The core object

In the simplest generic linear form, one point has fixed candidate directions `r_k` and nonnegative structural masses `m_k`:

```text
h_k = r_k^T x

y = tanh(g * sum_k m_k h_k)

m_k >= 0
sum_k m_k = 1
```

Therefore

```text
w_eff = sum_k m_k r_k
```

So the **structural variables** are positive and conserved, but the **effective operator** can be signed because the fixed basis directions are signed.

One point chooses one effective row from the convex hull of its available basis directions. A population stacks those rows into a matrix. If some input coordinates are peer broadcasts, part of that matrix is recurrent.

That immediately gives Twensday several concrete mathematical questions:

- What region of matrix space is reachable for a given basis and reserve?
- What ranks, singular spectra, sparsity patterns, and recurrent eigenvalues are produced by growth?
- Which tasks produce feedforward matrices, recurrent matrices, switching matrices, or mixed structures?
- What changes when the local feature map is nonlinear?
- How much of the result is simply constrained regression in an awkward parameterization?

See [`WHAT_CAN_IT_COMPUTE.md`](./WHAT_CAN_IT_COMPUTE.md).

## Direct lineage

The shortest useful ancestry is:

- [`FunctionalArbors`](https://github.com/anttiluode/FunctionalArbors) — geometry as a parameterization of computation; structural BSS experiments and strong FIR attackers.
- [`Monday`](https://github.com/anttiluode/Monday) — persistent structure as computation; Representation / Factorization / Use.
- [`Tuesday`](https://github.com/anttiluode/Tuesday) — temporal identifiability / ICA–AMUSE–SOBI–IVA branch. Useful machinery, but not the destination here.
- [`yrotisopeRweN`](https://github.com/anttiluode/yrotisopeRweN) — direct parent of Twensday. The growing-matrix and recurrent-growth gates are there.

A much earlier conceptual ancestor is [`Sunday`](https://github.com/anttiluode/Sunday), where transient activity versus persistent transfer structure became a recurring question.

## Two biology papers that actually changed the abstraction

These are motivation, not validation of the software model.

- Ido Aizenbud et al., **Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons**, PNAS (2026). https://doi.org/10.1073/pnas.2533168123
- Christophe Leterrier, **The Axon Initial Segment: An Updated Viewpoint**, Journal of Neuroscience 38(9):2135–2145 (2018). https://doi.org/10.1523/JNEUROSCI.1922-17.2018

The modest lessons carried forward were:

```text
Aizenbud:
extended / compartmentalized / nonlinear local integration can matter
not simply "more branches = smarter"

Leterrier:
continuous internal computation -> distinct output/broadcast boundary
not "output means reset the whole cell"
```

Neither paper implies a `6 x 6` matrix, the eligibility rule, conserved mass, or the recurrent-growth mechanism used in the repos.

## What yrotisopeRweN had actually established by Gate 15

Very compressed:

```text
receiver-relative state can gate what matters now
        ↓
fixed structure + fast state can select different compositions
        ↓
past input can persist in receiver state
        ↓
eligibility can carry delayed causal credit
        ↓
positive growth + finite capacity can become structural allocation
        ↓
a dense potential matrix can become sparse and later regrow elsewhere
        ↓
internal computation can continue while output is emitted
        ↓
state can live in recurrent traffic between points
        ↓
finite growth can build a recurrent path when persistence is useful
        ↓
useful return traffic need not be pre-named
        ↓
semantic internal coordinates are unnecessary
stable internal coordinates are necessary
```

The strongest negative result at the end is important: for the simple recurrent memory toy, **neither nonlinearity nor overcompleteness was necessary**. Six dense linear mixed features for six raw streams worked.

That is exactly why Twensday should now characterize the operator rather than add decorative complexity.

## First research program

Twensday should begin by producing **matrix atlases** rather than another chain of mechanisms.

Suggested sequence:

1. **Linear reachability** — sample many random bases and tasks; compare the grown `W_eff` to unconstrained least squares and to the best point in the basis convex hull.
2. **Matrix taxonomy** — tasks designed to require selection, mixing, rotation, low-rank projection, recurrence, switching, integration, and oscillation.
3. **Population spectra** — grow 4–16 point systems and inspect rank, singular values, recurrent eigenvalues, sparsity, strongly connected components, and attractors.
4. **Capacity laws** — sweep total budget and reserve. Measure adaptation speed versus stability.
5. **Nonlinearity attack** — construct tasks where a linear basis provably cannot solve the target and ask whether local nonlinear conjunctions genuinely add a function class.
6. **Boring attackers** — least squares, sparse regression, ordinary RNNs, reservoir computing, and matched task-specific algorithms are always allowed to win.

## Rule for this repo

> **Do not infer intelligence from a pretty matrix. Identify what operator was learned, what task demanded it, what simpler model reproduces it, and what perturbation destroys it.**
