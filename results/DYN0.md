# DYN0 — hidden dynamics can drive a continuous adaptive machine

Development receipt, not confirmatory evidence. The world and hyperparameters were designed while building the gate.

## Question

Twensday has so far mostly inspected static effective matrices and slow allocation. DYN0 asks whether the larger object can do a mundane but genuinely dynamical job:

> **Can a continuously running population act correctly when the present sensor values are ambiguous, the useful information exists in temporal dynamics, reward is delayed, and which physical sensor deserves trust changes over time?**

This is the first Twensday gate explicitly aimed at the working phrase **dynamic AI**. It is not a claim that the architecture is novel or intelligent. It is a test of whether the decomposition itself can make a competent machine.

## World

Each run is one uninterrupted stream:

```text
9000 time steps
8 sensor processes
hidden binary mode +/-1
mode changes roughly every 240 steps
which sensor is useful changes roughly every 1800 steps
reward arrives 30 steps after the action
no trial reset
```

The hidden mode changes the **sign of temporal correlation** in several sensor streams. The instantaneous sensor marginal is deliberately symmetric enough that a current-value learner has almost no mode information.

For the strongest useful sensor, schematically:

```text
mode +1:  x(t) ~= +0.94 x(t-1) + noise
mode -1:  x(t) ~= -0.94 x(t-1) + noise
```

The current scalar value can look the same in both worlds. The recent relation to its own past carries the clue.

Two channels provide useful evidence, two provide misleading opposite evidence, and the rest are weak/noisy. Their physical identities rotate on the slower reliability clock.

The action is simply which of two controllers/modes to select. Reward is `+1` for the correct choice and `-1` for the wrong one, delivered 30 steps later.

## Machine

Each sensor is treated as a continuously running point with fast state:

```text
q_i(t) = beta q_i(t-1) + (1-beta) x_i(t) x_i(t-1)

b_i(t) = tanh(3 q_i(t))
```

`b_i` is its broadcast. This is intentionally a very simple AMUSE-like temporal primitive: history has become current state.

A decision point owns finite structural mass over those stable broadcast addresses:

```text
m_i >= reserve
sum_i m_i = 1

y(t) = sum_i m_i b_i(t)
action(t) = sign(y(t))
```

Thirty steps later the scalar consequence returns. Eligibility means it is attached to the broadcasts that actually produced that old action:

```text
fitness_i = reward * old_action * old_broadcast_i
impulse_i = max(m_i * fitness_i, 0)

m <- conserve_and_normalize(m + eta * impulse)
```

If the old action was correct, broadcasts that supported it receive positive structural evidence. If it was wrong, broadcasts that opposed it receive positive evidence.

There is still **no explicit negative structural update**. Conservation supplies relative shrinkage.

## Reproduced result

Twelve deterministic seeds, 9000 steps each, reward delay 30. The committed GitHub Actions workflow independently reproduced the same run.

| method / lesion | accuracy |
|---|---:|
| oracle currently-best sensor | `0.9753 +/- 0.0042` |
| Fixed Share attacker | **`0.9674 +/- 0.0064`** |
| **Twensday dynamic allocator** | **`0.9646 +/- 0.0047`** |
| global signed learner | `0.9534 +/- 0.0058` |
| one-lag relation only, no accumulated fast state | `0.8658 +/- 0.0071` |
| shuffled delayed credit | `0.7636 +/- 0.0247` |
| stateless current-value learner | `0.5077 +/- 0.0126` |
| uniform use of the same dynamic broadcasts | `0.5052 +/- 0.0141` |
| scramble structural broadcast addresses every step | `0.5043 +/- 0.0095` |

The Twensday allocator places on average:

```text
0.8806 +/- 0.0113
```

of its finite structural mass on the two currently useful physical channels.

When sensor usefulness moves to new physical addresses, Twensday accuracy is:

```text
first 100 steps   0.8815 +/- 0.0475
first 400 steps   0.9491 +/- 0.0133
```

Fixed Share is almost identical over those windows (`0.8790`, `0.9497`).

## What DYN0 earns

### 1. The task is genuinely dynamical

The stateless learner is at chance. The hidden condition is available in the process's relation to its past, not reliably in the present sample.

So DYN0 finally gives the sentence

```text
history -> state -> present action
```

a useful job rather than using temporal state only as a memory demonstration.

### 2. Accumulated fast state matters

Replacing the continuous correlation state with only the single latest lag product drops accuracy from about `0.965` to `0.866`.

So merely having one previous sample is not the whole result. Accumulating noisy temporal evidence into ongoing state buys substantial robustness.

### 3. Dynamics alone are not enough

Uniformly averaging the exact same useful fast broadcasts gives chance performance because useful and misleading sensor processes coexist.

The slow structural layer has a different job:

> **fast state estimates what each stream is doing now; slow allocation estimates which stream is worth trusting.**

### 4. Stable addressability is essential again

Randomly reassign the broadcasts to structural coordinates every timestep and performance collapses to chance (`0.5043`).

The information still exists. What disappears is the ability of delayed consequence to accumulate on a reusable structural place.

This reprises Gate 14/15 in a more useful task.

### 5. Delayed causal credit matters

Attach reward to a random historical broadcast/action instead of the event that caused it and accuracy falls to `0.7636`.

It does not fall completely to chance because the environment has slow regularities, so random old activity is sometimes accidentally relevant. But the large loss shows that causal address through time is doing real work.

## The important attacker result

Fixed Share slightly wins:

```text
Fixed Share   0.9674
Twensday      0.9646
```

The difference is small in this development world, but the conclusion should be strict:

> **DYN0 is not evidence that positive conserved structural growth is a superior online tracking algorithm.**

Fixed Share is a mature method designed for changing experts and remains the cleaner software solution.

The signed global learner also performs strongly. Twensday remains interesting only insofar as its update constraints matter:

```text
local temporal state
+ stable broadcast address
+ one delayed scalar consequence
+ positive local structural reinforcement
+ finite conserved capacity
```

## The biggest cheat

The fast temporal feature was supplied by hand:

```text
x(t) * x(t-1)
```

That is almost the exact statistic the synthetic world was designed to reveal.

So DYN0 does **not** show that a generic dynamical point discovers the right state coordinates. It shows what can happen **once a useful local temporal state exists**.

That is now the wall.

## DYN1

Do not make the world more elaborate yet. Remove the gifted temporal detector.

Give each point only something like:

```text
raw incoming stream
+ generic leaky / resonant local states
+ a small fixed nonlinear feature bank
+ peer broadcasts
```

and ask whether consequence plus finite allocation can find a useful temporal listener without an explicit `x(t)x(t-1)` coordinate.

Attack with:

```text
explicit lag features + logistic/RLS
reservoir / echo-state machine
small RNN / GRU
Fixed Share over task-matched experts
same-information local multiplicative rules
```

Only after that should the action alter the world itself, closing the perception-action loop.

## Current working definition

DYN0 suggests a useful, deliberately modest meaning of **dynamic AI** for Twensday:

> **A continuously running machine in which recent history lives in fast internal state, state changes the meaning of present signals, points broadcast into the ongoing computation, delayed consequence changes slower finite structure, and that changed structure alters how future trajectories are processed.**

Existing RNNs, reservoirs, state-space models, adaptive filters, online-learning algorithms and control systems already occupy large parts of this territory. Twensday's question is not whether dynamics are new. It is whether this particular separation of **fast state / ongoing traffic / delayed local credit / slow resource allocation** buys anything useful under the constraints that motivated it.
