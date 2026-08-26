# DYN2 — circulating memory works; simple memory does not justify it

Development receipt, not confirmatory evidence. The task, recurrent bank, AIS-like rule, and attacker parameters were explored while constructing the gate.

## Why this gate existed

DYN0 showed that fast temporal state and slow structural trust can cooperate.

DYN1 removed the hand-written lag statistic and showed that an active temporal repertoire plus delayed structural selection can discover some useful temporal coordinates.

DYN2 asks the next network-level question:

> **Can useful state live in traffic circulating between points rather than inside one long-lived local state?**

A second question came from the axon-initial-segment / chandelier-cell papers:

> **Does an independent output-excitability state earn itself once recurrent traffic must remain stable?**

The AIS abstraction here is intentionally tiny. It is not a biological AIS model.

## Biological prompt, kept narrow

Christophe Leterrier's *The Axon Initial Segment: An Updated Viewpoint* (J Neurosci, 2018) emphasizes two separable AIS roles relevant to the software abstraction:

1. concentrated ion-channel machinery initiates and shapes action potentials, while AIS channel state, morphology, length, and position can adapt excitability on several timescales;
2. the AIS also helps maintain axonal identity and participates in selective trafficking / retrieval at the soma-to-axon boundary.

That suggested a computational distinction:

```text
internal state exists
        !=
it is emitted into the recurrent network
```

DYN2 tests only the first, simplest consequence: an independent emission/excitability gain. It does not model molecular trafficking or chandelier circuitry.

## World

One continuous 9000-step stream.

A hidden binary state is announced by a noisy four-step cue. Then the useful instantaneous cue disappears for roughly 250–450 steps. Cue amplitude changes between regimes over a wide range.

The machine must continuously report the last announced sign.

Reward arrives 20 steps after the action.

## Recurrent machine

There are nine candidate two-point recurrent motifs with different return gains.

Each motif contains fast states `A` and `B`:

```text
external cue -> A -> B
                ^    |
                |____|
```

Each local state has only leak `0.15`. Without the return path, neither can preserve useful state for hundreds of steps.

Slow positive/conserved structural mass allocates trust over the stable motif addresses from delayed consequence.

### AIS-like output gate

The internal states are distinct from the emitted traffic:

```text
internal A/B state -> slowly adapted local excitability gain -> broadcast
```

The local gain homeostatically targets a moderate emitted activity level. This is only an AIS-inspired abstraction of independently adjustable output excitability.

## First GitHub run

GitHub Actions independently ran the committed experiment on Ubuntu 24.04 / Python 3.12.

Twelve seeds, 9000 steps, reward delay 20:

| method / lesion | accuracy |
|---|---:|
| hand-written digital latch | **`0.9891 +/- 0.0148`** |
| **DYN2 recurrent bank + local AIS-like gate** | **`0.9822 +/- 0.0144`** |
| same recurrent bank + ordinary per-loop RMS AGC | `0.9747 +/- 0.0120` |
| same recurrent bank + fixed emission gate | `0.9626 +/- 0.0134` |
| one shared global homeostatic gate | `0.9512 +/- 0.0153` |
| local AIS-like gates but structural learning frozen | `0.9248 +/- 0.0421` |
| one long linear local leak | `0.8033 +/- 0.0352` |
| return traffic removed from the start | **`0.5052 +/- 0.0074`** |

The hand-written digital latch wins. DYN2 is not a better memory algorithm.

## The decisive return-path lesion

A second DYN2 trace is allowed to operate and learn normally. At step 5000 only the `B -> A` return traffic is severed. Internal variables and structural mass are not reset.

```text
before cut    0.9765 +/- 0.0250
post cut      0.5015 +/- 0.0125
```

So in this implementation the long-lived information really is carried by recurrent circulation. It is not secretly held by the local leak.

This establishes **where DYN2 stores its state**, not that circulation is necessary for the task.

## Attack 1 — one nonlinear point kills the necessity claim

A one-dimensional bistable attacker is given the same noisy cue:

```text
q(t+1) = tanh(1.5 q(t) + 1.5 x(t))
```

It has one nonlinear self-state and no inter-point circulation.

Independent GitHub result:

```text
single bistable point    0.9891 +/- 0.0168
```

This matches the digital latch and slightly beats DYN2.

Therefore:

> **A task requiring one persistent binary state does not justify distributed recurrent circulation.**

The return-path lesion remains a clean mechanistic result, but it is not a computational advantage.

## Attack 2 — remove amplitude drift

The first world deliberately changes cue amplitude between regimes. That is exactly the sort of nuisance an adaptive output-gain mechanism should help with.

So the attacker repeats the world with unit cue scale throughout.

Independent GitHub result:

| fixed-scale method | accuracy |
|---|---:|
| ordinary per-loop RMS AGC | **`0.9938 +/- 0.0005`** |
| AIS-like per-loop homeostasis | `0.9935 +/- 0.0027` |
| fixed output gate | `0.9921 +/- 0.0011` |
| one global homeostatic gate | `0.9722 +/- 0.0049` |

The local AIS-like gate loses its special advantage. Ordinary AGC is fractionally best and even a fixed gate is essentially tied.

Therefore the current evidence is:

> **The DYN2 AIS-like variable mainly performs ordinary channel-wise scale stabilization in this task.**

That may still matter in a physical substrate, but it is not yet a distinctive software mechanism.

A single global homeostatic scalar is consistently worse than channel-wise regulation. This is compatible with the idea that output excitability can be local/selective, but it does not establish any special chandelier mechanism.

## What survives

### Survives: state can literally live in traffic

Under deliberately short local time constants, a two-point loop can carry memory for hundreds of steps. Cutting the return destroys that memory immediately.

### Survives: slow structural trust still matters

With the same recurrent bank and local gates, freezing structural allocation reduces accuracy from `0.9822` to `0.9248`.

So there remains a useful distinction between:

```text
fast recurrent state
slow structural trust over recurrent motifs
```

### Does not survive: recurrence is needed for simple memory

A one-state nonlinear attractor reaches `0.9891`.

### Does not survive: AIS-like homeostasis is uniquely useful here

RMS AGC does essentially the same job, and without amplitude drift all local gate variants converge near the ceiling.

## What the Leterrier paper changes conceptually

The more interesting AIS hint is no longer merely gain control.

The paper treats the AIS as both:

```text
electrogenic output machinery
+
compartment boundary / trafficking filter
```

For Dynamic AI, the stronger abstraction is therefore:

> **internal computation and exported network traffic are distinct objects.**

A future gate should not merely ask how strongly a point emits. It should ask whether different internal states or traffic classes should be selectively exposed to different recurrent partners while the internal state continues to exist.

That is closer to routing / expression than to AGC.

## Next wall

Do not build another one-bit memory loop.

The next task must make the distributed network do something a single matched local state cannot do.

Promising attack:

```text
several partial states live at different points
context/query changes quickly
same slow structure must support several fast effective routings
output must depend on which distributed state is currently allowed to become traffic
```

Attack that with:

```text
single/vector recurrent unit with matched total state
ordinary GRU/RNN
attention / explicit multiplexer
fixed reservoir + trained readout
context-indexed experts
```

The interesting Twensday criterion is no longer raw accuracy. It is whether one slowly learned physical/resource structure can preserve a repertoire while fast state changes which effective operator is expressed.

## Current DYN2 sentence

> **Twensday can make memory live in recurrent traffic, but simple memory gives no reason to do so. AIS-like local gain helps under scale drift but is currently just an AGC-class mechanism. The next useful test must require distributed state and fast selective expression, not merely persistence.**
