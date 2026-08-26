# Dynamic AI — the Twensday working question

This is a working research direction, not a novelty claim and not a definition of intelligence.

The phrase is useful only if it means something narrower than "a neural network with hidden state." RNNs, reservoirs, state-space models, spiking networks, adaptive filters and control systems are already dynamical machines.

Twensday's specific candidate object is:

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
(q(t+1), y(t)) = F(q(t), u(t), peer(t); M(t))

M(t+1) = G(M(t), eligibility(t), consequence(t))
```

where `q` is fast computational state and `M` is slower structural allocation.

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

The useful biological inspiration is modest:

```text
a real neuron is already in a state when a signal arrives
signals alter that state
that state changes the effect of later signals
neurons continue broadcasting into other continuously evolving neurons
some traffic returns
plasticity changes slower structure while all of this is happening
```

The software abstraction does not need literal dendrites, phases, Koopman modes, AMUSE, AIS geometry, or any particular biological mechanism.

The signal-train viewpoint matters because it changes the grammar from

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

## DYN0 — first useful slice

[`results/DYN0.md`](results/DYN0.md) gives the first concrete job.

The hidden condition is not identifiable from current sensor values; it is identifiable from temporal correlation. Sensor points turn recent temporal relation into fast state and broadcast it. A decision point receives only delayed binary success/failure and reallocates finite structural mass toward whichever physical broadcasts are currently useful.

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

The important result is not that Twensday wins. It does not beat the mature Fixed Share attacker.

What DYN0 earns is the decomposition:

```text
fast temporal state
        !=
slow structural trust
```

Both are necessary in the constructed world.

## What DYN0 cheated

The temporal primitive was supplied explicitly:

```text
x(t) * x(t-1)
```

So the machine was handed the relevant local relation. It learned which continuous broadcasts to trust, not which temporal computation should exist.

That is the immediate wall.

## DYN1 — discover the temporal listener

Remove the explicit lag-product feature.

Each point should receive raw traffic and own only a generic repertoire such as:

```text
several leaky states / time constants
several resonant states
small fixed signed mixtures
small generic local nonlinear conjunction bank
peer broadcasts
```

Slow consequence-driven allocation must discover which local dynamical coordinates are useful.

Required attackers:

```text
explicit delay-line features
RLS / adaptive filtering
reservoir / echo-state network
small RNN / GRU
matched hand-written autocorrelation detector
same-information multiplicative learners
```

If the generic Twensday point only rediscovers a worse autocorrelation detector, record exactly that.

## DYN2 — let state live between points again

DYN0 points broadcast evidence but do not need returning traffic to preserve their state.

The next network-level test should make a useful state impossible to maintain inside any one point alone:

```text
point A sees one partial stream
point B sees another
A -> B -> A circulation becomes useful
```

Then ask whether finite growth discovers/selects the recurrent subgraph that makes prediction or action possible.

Controls:

```text
cut return traffic after growth
shuffle broadcast addresses
remove persistence requirement
fixed random recurrent graph
ordinary RNN / reservoir attacker
```

## DYN3 — close the perception/action loop

So far the action is scored by the world but does not change the world's future dynamics.

A stronger machine must act on a plant:

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

Now bad inference changes the future evidence the machine receives. Credit assignment and state become genuinely coupled to control.

Good candidate tasks are boring:

```text
switching linear plant control
adaptive noise cancellation
changing vibration path
sensor fault / drift compensation
online beam / filter selection
small robot or simulated body with changing actuator/sensor geometry
```

## What would count as interesting

Not "it has dynamics." That is old.

Not "it can remember a bit." RNNs do that.

Not "the matrix looks brain-like."

A useful result would be one of these:

1. under one-sided/local structural update constraints, the fast-state + finite-allocation machine approaches a stronger centrally optimized adaptive method;
2. slow structure gives robustness or rapid reacquisition when dynamical regimes recur;
3. one learned structure supports several fast state-dependent effective operators without retraining;
4. recurrent structure grown from delayed consequence discovers a compact useful state machine rather than merely memorizing a supplied graph;
5. on a real continuous stream, the architecture buys a measurable tradeoff in adaptation, compute, energy, memory, robustness, or hardware simplicity.

## Current sentence

> **Dynamic AI, in the Twensday sense, is a continuously running machine in which history lives in fast state, state changes the effect of present traffic, points keep broadcasting into one another, delayed consequence assigns credit to earlier local activity, and slower finite structure changes the dynamics through which future signals will move.**

That is the question now. The experiments decide whether it is merely another coordinate system for known adaptive machinery or whether one of its constraints produces something worth keeping.
