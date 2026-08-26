# Dynamic AI — the Twensday working question

This is a working research direction, not a novelty claim and not a definition of intelligence.

The phrase is useful only if it means something narrower than "a neural network with hidden state." RNNs, reservoirs, state-space models, spiking networks, adaptive filters and control systems are already dynamical machines.

The Twensday question is:

> **Can a useful AI be built around an ongoing dynamical process whose active state changes the meaning of present traffic, whose finite longer-lived memories can reinstate old states, and whose slower structure/knowledge changes what future dynamics are easy to express?**

The biological inspiration is modest. A real neuron is already in a state when a signal arrives; signals alter that state; state changes what later signals do; neurons continuously perturb other evolving neurons; some traffic returns; and plasticity changes slower structure while all of this continues.

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

## The state variables have multiplied

The original Twensday abstraction was roughly `fast q + slow M`. The hippocampal detour made that too small.

The current useful separation is:

```text
q   active fast computational state
    what is happening now / what recent traffic currently means

e   short eligibility / causal trace
    which earlier local event can still receive delayed consequence

H   fast-write, potentially long-retained dormant state
    an episode or previously learned fast state that can disappear from active dynamics
    and later be reinstated

M   slow reusable knowledge / structure
    what the system has learned well enough to regenerate, predict, or make easy

theta (optional)
    output/expression state kept distinct from internal computation
```

These are not simply four different leak constants.

In particular:

```text
write speed != retention time
retain != replay
remember exactly != generalize from
active state != dormant state
internal state != exported traffic
```

A compact sketch is now closer to:

```text
q(t+1), y(t) = F(q(t), u(t), peer(t), retrieve(H,t); M(t), theta(t))

e(t+1)      = E(e(t), local activity, consequence timing)
H(t+1)      = W(H(t), q(t), u(t); M(t))
M(t+1)      = G(M(t), e(t), consequence(t), selected experience)
```

The interesting reverse arrow is that `M` can help decide what deserves scarce `H` capacity.

---

## DYN0 — fast temporal state and slow structural trust

[`results/DYN0.md`](results/DYN0.md)

A hidden condition was identifiable from temporal correlation but not current sensor values. Sensor points turned recent temporal relation into fast state; delayed binary success/failure reallocated finite structural mass toward useful broadcasts.

Twelve-seed result:

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

DYN0 did not beat mature Fixed Share. It earned only:

```text
fast temporal state != slow structural trust
```

Its cheat was that the useful statistic `x(t) * x(t-1)` was supplied by hand.

## DYN1 — active temporal listeners from raw streams

[`results/DYN1.md`](results/DYN1.md)

The hand-written lag product was removed. Each point received raw traffic and owned a mixed bank of continuously occupied temporal states with compressive, near-linear, and expansive local transfer shapes. Slow consequence-driven allocation chose which internal dynamics deserved capacity.

```text
task-matched hand lag detector     0.9699
basket-inspired active bank        0.8042
matched random nonlinear bank      0.7717
active bank, local growth frozen   0.5069
passive leaky bank                 0.5065
```

What survives:

```text
passive multiscale memory alone was not enough
active local temporal nonlinearity created useful coordinates
slow local structural selection was necessary to exploit them
```

The exact digital statistic still wins badly. Biology supplied a useful repertoire prior, not magic.

## AIS / axon lesson — internal state is not automatically traffic

The AIS work suggested separating internal computation from emission/expression. DYN2 tested the simplest version as an adaptive gain and mostly killed the special story.

The stronger abstraction that remains is:

```text
internal state exists
        !=
it must be exported
        !=
every internal state / traffic class must be exported identically
```

That may matter later for selective recurrent routing. Ordinary normalization, attention and standard gating remain mandatory attackers.

## DYN2 — circulating memory works, but simple memory does not justify it

[`results/DYN2.md`](results/DYN2.md)

A two-point `A -> B -> A` loop with short local leaks could preserve a binary state for hundreds of steps. Cutting only the return path after normal operation gave:

```text
before cut    0.9765
post cut      0.5015
```

So memory genuinely lived in circulating traffic in that implementation.

But a single nonlinear bistable state reached `0.9891`, matching the digital latch and slightly beating the distributed loop. Therefore DYN2 established a possible *location* for memory, not a computational reason to distribute one-bit memory.

The AIS-like gain also failed to become special. With cue-amplitude drift removed:

```text
ordinary RMS AGC                  0.9938
AIS-like local homeostasis        0.9935
fixed output gate                 0.9921
global homeostasis                0.9722
```

The current AIS-like rule is an AGC-class mechanism on this task.

---

# Hippocampal detour — memory lifetime is not learning timescale

The hippocampal/cortical comparison forced a useful correction.

A memory can be **written quickly and retained for a long time**. Slow-write and long-lived are not synonyms.

That suggests a memory economy rather than a ladder of larger time constants:

```text
active fast state
short causal trace
fast-write dormant episodic/context state
slow reusable knowledge
```

The practical questions become:

```text
what stays active?
what deserves fast long retention?
what can be reconstructed and therefore forgotten?
what should influence slow knowledge?
what should remain an exception rather than become a rule?
what dormant state should be reinstated when a situation returns?
```

## DYN3 — complementary timescales

[`results/DYN3.md`](results/DYN3.md)

The world mixed repeatable semantic regularities with rare item-specific exceptions. A slow linear "cortex" generalized; a finite fast store could remember exact events after one exposure.

The cleanest useful variant was not replay. It was a **surprise-selected finite episodic store** guided by the slow model.

Representative results:

```text
slow cortex:
new-item accuracy             0.8243
long regular recall          0.9328
long exception recall        0.0754

surprise-selected H, no replay:
new-item accuracy             0.8243
short exception recall       0.9951
long regular recall          0.9630
long exception recall        0.8690

after erasing H:
regular recall               0.9347
exception recall             0.0745
```

The lesion is the important part:

```text
what became regular knowledge survives H erasure
what remained an idiosyncratic episode dies with H
```

So `H` and `M` are doing different jobs.

## DYN3B — retain is not consolidate

[`results/DYN3B.md`](results/DYN3B.md)

The next attack asked whether replay should have its own gate. This failed.

```text
no replay                  new-item accuracy  0.8243
moderately congruent replay                   0.7954
congruent replay                              0.7932
soft congruent replay                         0.7723
conflicting replay                            0.6922
uniform replay                                0.6882
```

The fast store correctly concentrated on surprising exceptions. Replaying those exceptions into the slow statistical model distorted the very regularity the slow model was supposed to learn.

Current conclusion:

> **Worth remembering is not the same as worth training the slow model on repeatedly.**

Naive replay has not earned a place in Twensday. Do not rescue it by parameter fiddling on this stationary stream.

## DYN3C — forgetting becomes useful allocation

[`results/DYN3C.md`](results/DYN3C.md)

The slow model was allowed to guide a finite cache: retain what the slow model still cannot regenerate; let predictable material lose scarce episodic space.

At an 800-event delay:

```text
capacity 64:
FIFO exception recall             0.0747
random eviction                   0.0747
slow-model residual cache         0.4415

capacity 128:
FIFO                              0.0747
random                            0.0760
residual cache                    0.7389
exception-flag oracle             1.0000

capacity 256:
FIFO                              0.0747
random                            0.1110
residual cache                    0.9498
```

The cache was never told which events were exceptions, yet its final contents became strongly exception-enriched:

```text
capacity 32      ~100% exceptions
capacity 64      ~99.5%
capacity 128     ~93.2%
capacity 256     ~76.9%
```

So the slow model acted as a **compression oracle for fast memory**:

> if slow knowledge can regenerate this, scarce exact-memory space can be spent elsewhere.

This is familiar model-aware caching / residual-memory territory, not a novelty claim. But it is a useful architectural result.

Forgetting is no longer only decay. It can be an active consequence of redundancy with slower knowledge.

---

## DYN4 — H stores an old fast state, not merely an example

[`results/DYN4.md`](results/DYN4.md)

DYN4 moved beyond exact-item storage.

There are 48 recurring contexts. During a short visit, a fast state `q` adapts to the context. The context then disappears for roughly 1,200 events. A finite `H` can hold only 12 dormant context states.

When a context returns, `H` can reinstate its previously learned `q` before the new visit has time to relearn it.

First-event reacquisition:

```text
no H                            0.8065
FIFO 12-slot cache              0.8065
random eviction                 0.8090
model-residual state cache      0.8607
true-offset oracle cache        0.8360
unbounded context table         0.9471
scrambled residual addresses    0.8034
```

For high-residual contexts:

```text
no H                            0.7209
model-residual H                0.8293
```

The address scramble is crucial. The same extra stored scalar states attached to the wrong returning contexts lose the gain.

So the current useful operation is:

```text
learn q quickly while context is active
        ↓
write q into dormant finite H
        ↓
context disappears; q is gone from active dynamics
        ↓
context returns much later
        ↓
retrieve the matching dormant state
        ↓
reinstate q and continue from prior adaptation
```

That is a real three-timescale machine:

```text
q   fast within-context adaptation
H   fast-write / long-retained dormant context state
M   slow global regularity
```

The unbounded explicit context table remains the software upper attacker and wins, as it should.

## DYN4B — what should H retain?

The obvious attack was that the slow model might be unnecessary: perhaps just keep contexts with large `|q|`.

With 12 slots:

```text
priority                              first-event reacquisition
slow-model residual improvement              0.8607
slow-model raw prediction error              0.8544
mean |q|                                      0.8418
final |q|                                     0.8378
late fast-state error                         0.8306
no H                                          0.8065
```

So the exact residual-improvement formula is not sacred; much of the gain is already obtained from simple slow-model error. But state magnitude alone is weaker.

The robust architectural result is:

> **slow knowledge can help decide which previously learned fast states deserve long retention.**

---

# What survives now

Twensday has not produced a superior generic optimizer or memory algorithm. Mature digital mechanisms repeatedly win when they are allowed to use unconstrained explicit state.

The pieces that have survived falsification so far are narrower:

1. fast dynamical state and slow structural trust can do different jobs;
2. active local temporal nonlinearities can create useful features that passive leaks do not;
3. state can physically/computationally live in recurrent traffic, though simple tasks do not require it;
4. internal state and exported traffic are worth keeping conceptually separate, though AIS-like gain itself collapsed to AGC;
5. fast-write memory and slow learning are different axes from retention lifetime;
6. retaining an episode and training slow knowledge on it are different decisions;
7. a slow model can guide scarce fast-memory allocation by identifying what it cannot regenerate;
8. a dormant stored state can later reinstate an old fast dynamical state and accelerate reacquisition;
9. stable addressability continues to matter: scramble the mapping between dormant state and returning context and the benefit disappears.

# DYN5 — remove the exact context ID

DYN4 still cheats badly.

When context 37 returns, the machine is handed an exact stable key equivalent to:

```text
context = 37
```

That turns `H` into a dictionary.

The next gate should remove that key.

A returning situation should provide only a partial/noisy cue. The machine must decide whether the current trajectory resembles something previously experienced strongly enough to retrieve a dormant state.

Target grammar:

```text
partial present sensory/dynamical state
        ↓
recognize similarity to an old context
        ↓
retrieve dormant H candidate
        ↓
reinstate old q
        ↓
subsequent incoming signals test/correct that reinstatement
        ↓
continue dynamics
```

Mandatory attackers:

```text
nearest-neighbour lookup
vector database / exact embedding retrieval
k-NN with the same memory budget
prototype cache
Bayesian context filter / HMM where appropriate
small GRU / RNN with matched state
unbounded context table with noisy-key classifier
```

Kill conditions:

- if nearest-neighbour lookup gives the same reacquisition at the same memory/compute budget, call the mechanism associative retrieval and move on;
- if wrong retrievals poison the fast dynamics more than starting from scratch, the reinstatement mechanism needs a confidence/rejection state;
- if `M` does not improve retrieval/allocation beyond raw cue similarity, drop the cortex-to-H feedback claim;
- if dormant state cannot be useful after the exact address is removed, DYN4 was only a keyed cache.

If DYN5 survives, then the hippocampal detour finally rejoins the original signal-train thought:

> **the present trajectory itself can awaken an old dormant dynamical state, which then changes what subsequent signals mean.**

Only after that should we return to distributed selective routing and then close the perception/action loop.

# Current sentence

> **Dynamic AI, in the Twensday sense, is a continuously running machine in which information can occupy different temporal roles: active fast state, short causal trace, dormant fast-write memory, and slow reusable knowledge. Present traffic changes active state; delayed consequence changes slower allocation; slow knowledge helps decide what deserves scarce long retention; and returning situations may reinstate old dynamical states so that the same incoming signal is interpreted by a system carrying a different history.**

The experiments decide whether this is merely a verbose decomposition of known adaptive memory machinery or whether one of these constrained interactions buys something useful in real continuous systems.