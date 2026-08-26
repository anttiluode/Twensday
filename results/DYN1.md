# DYN1 — active temporal compartments can discover some hidden dynamics

Development receipt, not confirmatory evidence. The task, repertoire, and hyperparameters were designed while building the gate.

## Question

DYN0 worked because every point was handed the task-matched temporal statistic

```text
x(t) * x(t-1)
```

and only had to learn which physical broadcast to trust.

DYN1 removes that gift:

> **Can a continuously running point receiving only a raw scalar stream discover a useful temporal listener by allocating finite structure over a generic bank of local dynamics?**

The biological motivation came from two fast-spiking basket-cell modeling papers by Liu & Sun:

- *Spatial integration of dendrites in fast-spiking basket cells* (Frontiers in Neuroscience, 2023; doi:10.3389/fnins.2023.1132980)
- *Temporal integration on the dendrites of fast-spiking basket cells* (Scientific Reports, 2024; doi:10.1038/s41598-024-81655-w)

Those studies report coexistence of supralinear and sublinear dendritic integration, strong influence of active dendritic spiking and morphology on somatic output, and temporal-window-dependent responses. They motivate a mixed active temporal repertoire. They do **not** imply the software bank below, and DYN1 is not a basket-cell model.

## World

Exactly the same hidden-dynamics world as DYN0:

```text
9000 continuous time steps
8 sensor streams
hidden binary mode changes ~every 240 steps
which physical streams are useful changes ~every 1800 steps
reward arrives 30 steps after the action
no trial reset
```

The present scalar values are deliberately poor evidence of the hidden mode. The useful information is in temporal organization.

## The active bank

Each point receives only raw `x(t)`.

It splits that stream into positive and negative local drive and maintains several continuously occupied windows:

```text
tau = [1, 2, 4, 8, 16, 32] simulation steps
```

Each window is then exposed through three crude local transfer families:

```text
compressive     power 0.5
near-linear     power 1.0
expansive       power 2.0
```

with several fixed thresholds and paired positive/negative candidate directions.

The words `compressive` and `expansive` are only a computational analogy to sublinear and supralinear integration. They are not fitted biophysics.

Each sensor-point owns conserved positive structural mass over its local active features. A second conserved mass vector chooses which sensor-points deserve trust.

So DYN1 has two slow allocation problems:

```text
within point:
which local temporal computation is useful?

across points:
which physical stream is useful?
```

Both receive only the same delayed binary consequence used in DYN0.

## Controls / attackers

### `hand_lag`

DYN0's exact task-matched coordinate is supplied by hand. This is the ceiling/control, not a fair generic learner.

### `passive_leaky`

Same family of time constants but only signed linear leaky states. No rectification and no active local transfer bank.

### `active_no_local_learning`

The full active repertoire exists, but local structural mass is frozen uniformly. Only the outer sensor trust can adapt.

### `random_nonlinear`

A matched-size generic tanh state bank with random recurrent coefficients, input gains, and biases. This asks whether the basket-inspired repertoire is doing more than generic reservoir luck.

A trained GRU/RNN, RLS, and adaptive FIR are **not yet included**. DYN1 therefore does not establish competitiveness with mature temporal models.

## GitHub reproduction

GitHub Actions independently ran the committed experiment on Ubuntu 24.04 / Python 3.12.

Twelve seeds, 9000 steps each, delayed reward 30 steps:

| method / lesion | accuracy |
|---|---:|
| task-matched hand lag detector | **`0.9699 +/- 0.0042`** |
| **basket-inspired active temporal bank** | **`0.8042 +/- 0.0057`** |
| matched random nonlinear bank | `0.7717 +/- 0.0102` |
| active bank, local structural learning frozen | `0.5069 +/- 0.0077` |
| passive leaky bank | `0.5065 +/- 0.0140` |

Final structural mass in the active bank, grouped only by crude transfer family:

```text
compressive    0.6477 +/- 0.0217
near-linear    0.0402 +/- 0.0155
expansive      0.3121 +/- 0.0151
```

## What DYN1 earns

### 1. Passive temporal memory is not enough in this task

A bank of ordinary leaky states with several time constants stays at chance:

```text
0.5065
```

So merely preserving multiple low-pass versions of the raw stream does not expose the hidden temporal relation in a form the one-sided structural learner can use.

### 2. An active nonlinear temporal repertoire creates a useful function class

The mixed active bank reaches:

```text
0.8042
```

from raw scalar streams with no explicit `x(t)x(t-1)` coordinate.

That is a real escape from the DYN0 cheat, though a partial one.

### 3. Slow local structural selection is essential

Freeze local feature allocation while leaving the exact same active states present:

```text
0.8042 -> 0.5069
```

So the result is not merely the existence of a nonlinear reservoir. Delayed consequence has to make some local dynamical coordinates structurally privileged.

This extends Twensday's old distinction:

```text
fast state creates candidate meanings
slow structure decides which meanings deserve persistent capacity
```

### 4. The biological prior helps, but modestly

Matched random nonlinear dynamics reach:

```text
0.7717
```

The basket-inspired bank reaches:

```text
0.8042
```

The gap is real in this development run but not large enough to declare a special biological advantage. A better tuned or trained reservoir may erase it.

### 5. The exact digital answer remains far better

The hand-written lag detector reaches:

```text
0.9699
```

So DYN1 has **not** discovered the best temporal representation. It has shown that a generic active temporal body plus delayed structural selection can discover *some* useful representation.

That is a much narrower result.

## An important surprise

The synthetic learner did **not** reproduce the basket-cell papers' qualitative statement that supralinear compartments dominate somatic firing.

Its final mass is mostly on the crude compressive family:

```text
~65% compressive
~31% expansive
~4% near-linear
```

This is not a contradiction of the biology. The software transfer powers are only loose analogues, the world is an AR-sign discrimination task rather than synaptic integration, and structural mass measures learned utility rather than contribution to biological somatic firing.

It is nevertheless an excellent warning:

> **biology may suggest the repertoire without predicting which member a different computational task will select.**

That is exactly how the biological inspiration should be used in Twensday.

## Next wall

Do not add more biological ornament yet.

Attack DYN1 with stronger standard machinery:

```text
explicit delay-line + linear/nonlinear classifier
adaptive FIR / RLS
small echo-state network with trained readout
small matched-parameter GRU / RNN
```

Then make the temporal problem itself nonstationary: let the useful timescale or temporal relation change while the physical stream remains the same. Ask whether the slow local structural distribution retains a repertoire that can be rapidly re-expressed when an old dynamical regime returns.

That would test something more specifically Twensday-like than static feature quality:

> **can persistent structural investment act as a reusable repertoire over which fast state produces several different effective temporal operators?**

## Chandelier/AIS stays separate

The chandelier-cell / AIS literature suggests another degree of freedom: output excitability can be changed without reallocating which internal computation is trusted.

That should not be mixed into DYN1. A later gate can independently test whether an AIS-like output threshold and selective chandelier-like group modulation stabilize or adapt recurrent traffic better than ordinary normalization / AGC / learned bias controls.

For now DYN1 earns only this:

> **A raw continuous stream, a mixed active temporal repertoire, delayed consequence, and finite local allocation can discover a useful temporal listener that passive leaky state alone cannot. The result is incomplete and still substantially worse than the task-matched digital statistic.**
