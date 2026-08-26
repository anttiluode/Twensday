# What can the growing-matrix system compute?

This note is the mathematical starting point for Twensday.

The main purpose is to stop using the word **matrix** loosely. There are several different objects in the lineage and they should be measured separately.

## 1. Structural mass is not necessarily the effective matrix

In the Gate-10 relation basis, candidate coordinates were explicit pairwise products:

```text
q_ij(x) = A_i B_j
```

A nonnegative conserved field `M_ij` weighted those coordinates. There, calling `M` a matrix is literal: it is a bilinear operator over the supplied product basis.

Gate 15 removed semantic coordinates. Let fixed candidate feature directions be rows of

```text
R in R^(K x d)
```

and let structural mass be

```text
m in R^K
m_k >= reserve
sum_k m_k = 1
```

For the linear feature ablation,

```text
h = R x

y_pre = m^T h
```

therefore

```text
y_pre = m^T R x
      = w_eff^T x

w_eff^T = m^T R
```

So `m` is the structural allocation and `w_eff` is the actual input-space row vector.

A population of `P` points gives

```text
W_eff in R^(P x d)
```

by stacking the effective rows.

## 2. The reachable linear operators form a constrained set

If `m` is nonnegative and sums to one, then

```text
w_eff = sum_k m_k r_k
```

is a convex combination of the fixed basis rows `r_k`.

Ignoring the small reserve floor for a moment:

> **one point can only produce rows inside the convex hull of its available basis directions.**

The reserve shrinks this reachable region further toward the average basis direction because every candidate must retain some mass.

This gives a clean attacker for every linear experiment:

1. solve unconstrained least squares for the desired row `w*`;
2. solve the convex-hull projection problem for the best reachable `w_hull`;
3. compare the positive-growth dynamics to that optimum.

If growth performs poorly relative to `w_hull`, that is an optimization failure.

If `w_hull` itself performs poorly relative to unconstrained `w*`, that is a representation/capacity failure.

Those are different failures.

## 3. Positive structural mass does not imply positive effective weights

The basis rows can contain positive and negative coefficients.

Therefore a convex combination of basis rows can still be signed:

```text
m_k >= 0
but
w_eff_j can be < 0 or > 0
```

This is one reason the structural-growth interpretation is less restrictive than a raw positive-weight network might appear.

But it remains constrained: signed coefficients are available only insofar as the fixed basis geometry spans the needed directions within its convex hull.

## 4. In a recurrent population, part of W_eff is a recurrent matrix

Partition each point's raw input coordinates into external and peer-broadcast channels:

```text
x_t = [u_t, y_(t-1)]
```

Then each effective row can be partitioned:

```text
w_p = [b_p, a_p]
```

and the population approximately has the form

```text
y_t = f(B u_t + A y_(t-1))
```

where rows of `A` and `B` are induced by each point's structural allocation.

This means the grown system can be analyzed with ordinary recurrent-system tools:

- eigenvalues and spectral radius of `A`;
- strongly connected components of the effective graph;
- transient amplification;
- fixed points and attractors after the nonlinearity `f`;
- controllability/observability with respect to external channels;
- switching after structural reallocation.

Gate 12–14 already showed one tiny example: a two-point loop can support a persistent signed state, and growth allocates recurrent mass only when the task requires persistence.

Twensday should generalize that by **measuring the resulting A matrices directly**.

## 5. Continuous local state makes the operator dynamical

Gate 11 used persistent local feature states:

```text
z_k(t+1) = alpha_k z_k(t) + (1-alpha_k) phi_k(x_t)
```

and a readout such as

```text
y_t = f(sum_k m_k z_k(t))
```

Even with fixed structural mass `m`, this is no longer just a static matrix multiply. It is a bank of temporal filters followed by a constrained readout.

With recurrence between points, the architecture is close to a constrained reservoir/RNN:

```text
fixed local feature/filter bank
+ finite nonnegative structural allocation
+ continuous recurrence
+ delayed eligibility-driven structural change
```

That gives another obvious attacker: standard reservoir computing with an unconstrained trained readout.

## 6. Nonlinear local features change the function class

If

```text
h_k = phi(r_k^T x)
```

then

```text
y = f(sum_k m_k h_k)
```

cannot generally be collapsed into one raw-space matrix.

It becomes a fixed-feature nonlinear model with constrained output weights.

The important lesson from Gate 15 is negative: the simple persistent-state task did **not** require that nonlinearity. A linear mixed basis solved it.

Therefore Twensday should only credit nonlinear local composition on tasks with a clear linear impossibility or a strong linear attacker.

Candidate tasks:

- XOR/parity-like local conjunctions;
- bilinear relation selection;
- multiplicative context gating;
- phase-dependent conjunctions;
- functions with matched linear statistics but different higher-order structure.

## 7. Expected matrix families worth searching for

Twensday can deliberately ask the system to grow different operator shapes.

### Selector / router

One or a few dominant input directions.

Expected signature:

```text
low effective rank
high structural sparsity
few large coefficients
```

### Mixer / rotation-like operator

Several comparable signed directions.

Expected signature:

```text
broader structural occupancy
multiple singular values
possibly poor fit if convex hull lacks the required directions
```

### Low-rank projector

A population shares a smaller latent subspace.

Expected signature:

```text
rank(W_eff) << number of points
correlated rows
```

### Recurrent memory loop

Peer-broadcast submatrix `A` crosses a persistence/gain threshold.

Expected signature:

```text
strongly connected component
spectral radius near/above the task-relevant persistence boundary
stable nonlinear fixed points
```

### Oscillator / sequence carrier

Recurrent eigenstructure becomes complex or cyclic.

Expected signature:

```text
complex-conjugate eigenvalues
cycle-like directed graph
phase progression / delayed transitions
```

This has not yet been earned by the existing gates; it is a Twensday target, not a result.

### Context-dependent operator

Fast state `G(t)` changes the effective computation without changing structural mass.

A useful notation from the parent repo is

```text
W_eff(t) = M ⊙ W ⊙ G(t)
```

but in a generic feature basis the exact factorization may differ. The important measurement is the family of effective operators visited under different fast states while `M` remains fixed.

## 8. Matrix atlas: the first Twensday experiment suite

For every task, log at least:

```text
structural mass / entropy
W_eff
rank and singular values
row/column norms
recurrent submatrix A
spectral radius(A)
strongly connected components
adaptation time after task reversal
held-out task error
```

And compare to:

```text
unconstrained least squares
best convex-hull reachable solution
L1 / sparse regression
small ordinary RNN
reservoir + trained readout
matched task-specific baseline
```

The repo should then be able to answer the user's actual question empirically:

> **When this growth rule is asked to solve different problems, what kinds of matrices does it repeatedly choose to become?**

That is a much better target for Twensday than adding another mechanism before the current one is understood.
