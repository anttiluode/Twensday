# Dynamic AI — the Twensday working question

This is a working research direction, not a novelty claim and not a definition of intelligence.

The phrase is useful only if it means something narrower than "a neural network with hidden state." RNNs, reservoirs, state-space models, spiking networks, adaptive filters and control systems are already dynamical machines.

Twensday's candidate object is:

```text
FAST STATE
what has been happening recently?
what does incoming traffic mean in the state I am already in?

CONTINUOUS BROADCAST
my current state changes what I emit
other points continue changing while I emit

DELAYED CREDIT
which earlier local state / interaction caused a later consequence?

SLOW FINITE STRUCTURE
which possible computations / routes deserve persistent capacity?

RECURRENT TRAFFIC
state may live locally, in circulation between points, or in both
```

A compact abstraction is:

```text
(q(t+1), y(t)) = F(q(t), u(t), peer(t); M(t), theta(t))
M(t+1) = G(M(t), eligibility(t), consequence(t))
```

where `q` is fast computational state, `M` is slower structural allocation, and `theta` is an optional output/expression state kept distinct from the internal computation.

The important loop is:

```text
structure M
    ↓
shapes fast dynamics
    ↓
traffic + state produce action
    ↓
consequence returns later
    ↓
credit reaches old local activity
    ↓
finite structure reallocates
    ↓
future dynamics now run through a different machine
    ↺
```

## Why this came out of the neuron thought

The biological inspiration is modest:

```text
a real neuron is already in a state when a signal arrives
signals alter that state
that state changes the effect of later signals
neurons continue broadcasting into other continuously evolving neurons
some traffic returns
plasticity changes slower structure while all of this is happening
```

The software abstraction does not require literal dendrites, phases, Koopman modes, AMUSE, AIS geometry, or any particular biological mechanism.

The signal-train viewpoint changes the grammar from

```text
input -> function -> output
```

to

```text
signal perturbs an already-moving system
        ↓
changed system emits new perturbations
        ↓
those perturb other moving systems
        ↓
some return to a system that is no longer in the same state
```

## DYN0 — fast temporal state and slow structural trust

[`results/DYN0.md`](results/DYN0.md) gives the first concrete job.

A hidden condition is identifiable from temporal correlation but not from current sensor values. Sensor points turn recent temporal relation into state. A decision point receives delayed binary success/failure and reallocates finite structural mass toward useful physical broadcasts.

Twelve-seed development result:

```text
oracle currently-best sensor      0.9753
Fixed Share attacker              0.9674
Twensday dynamic allocator        0.9646
global signed learner             0.9534
one-lag-only state                0.8658
shuffled delayed credit           0.7636
stateless current-value learner   0.5077
uniform dynamic broadcasts        0.5052
scrambled structural addresses    0.5043
```

DYN0 does not beat mature Fixed Share. It earns only:

```text
fast temporal state != slow structural trust
```

Its cheat was that the useful local statistic `x(t) * x(t-1)` was supplied by hand.

## DYN1 — active temporal listeners from raw streams

[`results/DYN1.md`](results/DYN1.md) removes the hand-written lag-product coordinate.

The point receives raw traffic and owns a mixed bank of continuously occupied time windows with crude compressive, near-linear, and expansive local transfer shapes. Slow consequence-driven allocation chooses which local temporal coordinates and which physical streams deserve capacity.

The repertoire was motivated by fast-spiking basket-cell work showing coexistence of sublinear and supralinear dendritic integration, strong effects of active dendritic spikes, and temporal-window-dependent integration. The software bank is only an abstraction of that lesson.

GitHub Actions, twelve seeds:

```text
task-matched hand lag detector     0.9699
basket-inspired active bank        0.8042
matched random nonlinear bank      0.7717
active bank, local growth frozen   0.5069
passive leaky bank                 0.5065
```

DYN1 earns:

```text
passive multiscale memory alone was not enough
active local temporal nonlinearity created useful coordinates
slow local structural selection was necessary to exploit them
```

But the exact digital temporal statistic still wins badly. The biological prior helps only modestly over matched generic nonlinear dynamics.

## What the AIS paper added

Christophe Leterrier's AIS review sharpens a distinction we had been blurring.

The axon initial segment is both electrogenic output machinery and a soma/axon boundary involved in maintaining polarity and selectively sorting/retrieving traffic. Its channel composition and morphology can also adapt excitability on multiple timescales.

The useful software abstraction is therefore broader than "gain control":

```text
internal state exists
        !=
it must be exported
        !=
every internal state/traffic class must be exported identically
```

That is a hypothesis generator, not evidence that an AIS-like software component is useful.

## DYN2 — circulating memory, then slaughter it

[`results/DYN2.md`](results/DYN2.md) makes memory live in two-point recurrent traffic.

A brief cue announces a hidden +/- state and then disappears for hundreds of steps. Individual points have very short leak. Nine candidate A->B->A motifs provide possible circulating memories. Delayed consequence reallocates structural trust across stable motif addresses.

A local AIS-like output gain was also tested, separately from the internal A/B state.

First GitHub run:

```text
hand-written digital latch        0.9891
DYN2 + local AIS-like gate        0.9822
ordinary per-loop RMS AGC         0.9747
same loops, fixed gate            0.9626
global homeostasis                0.9512
freeze structural learning        0.9248
one long linear local leak        0.8033
remove return traffic             0.5052
```

The strongest mechanistic result is the return lesion. After normal operation and learning, sever only B->A traffic:

```text
before cut    0.9765
post cut      0.5015
```

So state genuinely lived in circulation in that machine.

Then the attackers killed the larger claims.

### Recurrence is not justified by one-bit memory

A single nonlinear bistable state gets:

```text
0.9891
```

It matches the digital latch and slightly beats DYN2. Therefore DYN2 shows a possible *location* for memory, not a reason to distribute that memory across points.

### AIS-like gain is mostly AGC here

Remove cue-amplitude drift:

```text
ordinary RMS AGC                  0.9938
AIS-like local homeostasis        0.9935
fixed output gate                 0.9921
global homeostasis                0.9722
```

The special advantage disappears. The current AIS-like rule is an AGC-class stabilizer on this task and should not be promoted as a new pillar.

The more interesting AIS hint that remains untested is **selective expression/routing**: internal computation can remain present while only some of it becomes network traffic.

## DYN3 — distributed state + fast selective expression

Do **not** build another one-bit loop.

The next task must make the network do something a single matched scalar state cannot do.

Candidate world:

```text
several partial states live at different points
slow structure learns a reusable communication repertoire
context/query changes much faster than structure can change
same persistent structure must support several different effective routings
only the context-relevant internal state should become useful network traffic
```

This directly attacks a stronger Twensday possibility:

> **Can one slowly learned structure support several fast state-dependent effective operators without retraining?**

Mandatory attackers:

```text
single/vector recurrent unit with matched total state
ordinary GRU/RNN
attention / explicit multiplexer
fixed reservoir + trained readout
context-indexed experts
```

A selective AIS/chandelier-inspired output gate only earns itself if it does something ordinary contextual gating/attention cannot do as cheaply under the architectural constraints.

## DYN4 — close the perception/action loop

Only after the distributed-state problem survives should action change the world:

```text
hidden changing dynamics
        ↓
continuous sensing
        ↓
dynamic point network
        ↓
action
        ↓
plant trajectory changes
        ↺
```

Good later tasks remain boring and useful:

```text
switching linear plant control
adaptive noise cancellation
changing vibration path
sensor fault / drift compensation
online beam / filter selection
small simulated body with changing actuator/sensor geometry
```

## What would count as interesting

Not "it has dynamics." That is old.

Not "it can remember a bit." DYN2 just demonstrated why that is too weak.

Not "the matrix looks brain-like."

A useful result would be one of these:

1. under one-sided/local structural update constraints, fast state + finite allocation approaches stronger centrally optimized adaptation;
2. slow structure gives rapid reacquisition when dynamical regimes recur;
3. one learned structure supports several fast state-dependent effective operators without retraining;
4. recurrent structure grown from delayed consequence discovers a compact useful distributed state machine rather than merely implementing a supplied memory loop;
5. selective output/expression control preserves useful internal state while routing only context-relevant traffic, and has a measurable advantage over boring normalization/attention under the same constraints;
6. on a real continuous stream, the architecture buys a measurable tradeoff in adaptation, compute, energy, memory, robustness, or hardware simplicity.

## Current sentence

> **Dynamic AI, in the Twensday sense, is a continuously running machine in which history lives in fast state, state changes the effect of present traffic, points keep broadcasting into one another, delayed consequence assigns credit to earlier local activity, and slower finite structure changes the dynamics through which future signals will move.**

DYN0–DYN2 show that individual pieces of this grammar can be made operational. They have not shown that the full grammar is preferable to conventional adaptive/recurrent machinery. The next experiment must demand distributed state plus fast context-dependent expression; otherwise we are just rebuilding latches and filters with biological names.
