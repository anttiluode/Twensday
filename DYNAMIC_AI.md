# Dynamic AI — Twensday compass

This is a working research direction, not a novelty claim and not a definition of intelligence.

The question is narrower than "can an AI have hidden state?" RNNs, reservoirs, SSMs, spiking nets, adaptive filters and control systems already do that.

> **Can a useful AI be built around an ongoing dynamical process whose active state changes the meaning of present traffic, whose finite longer-lived memories can preserve/reinstate selected states, and whose slower knowledge changes what future dynamics are easy to express and what deserves further learning?**

The biological inspiration is modest:

```text
a system is already in a state when a signal arrives
signals change that state
state changes what later signals do
other evolving systems keep sending traffic
some traffic returns
consequence arrives later
slower structure/knowledge changes while all of this continues
```

The signal-train grammar is therefore not

```text
input -> function -> output
```

but

```text
signal perturbs an already-moving system
        ↓
changed system emits new perturbations
        ↓
those perturb other moving systems
        ↓
some return to a system that is no longer in the same state
```

## Current state variables

The original `fast q + slow M` picture became too small.

```text
q   active fast computational state
    what is happening now / what recent traffic currently means

e   short eligibility / causal trace
    which earlier local event can still receive delayed consequence

H   fast-write, potentially long-retained dormant state
    selected information that can disappear from active dynamics
    and later be available again

M   slow reusable knowledge / structure
    what the system has learned well enough to regenerate, predict,
    compress, or make easy

theta (optional)
    output / expression state kept distinct from internal computation
```

These are not merely different leak constants.

```text
write speed != retention time
retain != replay
remember exactly != generalize from
active state != dormant state
internal computation != exported traffic
prediction error != useful curiosity
```

A deliberately loose sketch is:

```text
q(t+1), y(t) = F(q(t), u(t), peer(t), H(t); M(t), theta(t))
e(t+1)      = E(e(t), local activity, consequence timing)
H(t+1)      = W(H(t), q(t), u(t); M(t))
M(t+1)      = G(M(t), e(t), consequence(t), selected experience)
```

The most interesting reverse arrow so far is:

```text
M -> what deserves scarce H / learning capacity?
```

---

# Receipts

## DYN0 — temporal state != structural trust

[`results/DYN0.md`](results/DYN0.md)

A hidden condition was identifiable from temporal correlation but not current sensor values. Fast temporal state plus delayed structural allocation reached `0.9646`; Fixed Share reached `0.9674`; stateless and scrambled-address controls sat near chance.

What survived:

```text
fast temporal state and slow structural trust can do different jobs
stable addressability matters for delayed structural credit
```

What did not survive:

```text
new optimizer
```

## DYN1 — active temporal listeners

[`results/DYN1.md`](results/DYN1.md)

Removing the supplied lag-product feature gave:

```text
hand-written lag detector          0.9699
basket-inspired active bank        0.8042
matched random nonlinear bank      0.7717
active bank, local growth frozen   0.5069
passive leaky bank                 0.5065
```

Passive multiscale memory alone was not enough. Active local temporal nonlinearities plus slow selection created useful coordinates, but the exact digital statistic still won badly.

## DYN2 — circulating state is possible, not necessary

[`results/DYN2.md`](results/DYN2.md)

A two-point `A -> B -> A` loop genuinely stored a binary state. Cutting only the return path gave:

```text
before cut    0.9765
post cut      0.5015
```

But a single nonlinear bistable scalar reached `0.9891`. Therefore recurrence showed **where** state can live, not why a one-bit task needs distributed circulation.

The AIS-like gain also collapsed into ordinary AGC-class behavior.

The remaining AIS/axon abstraction is only:

```text
internal state exists
        !=
it must be exported
        !=
every internal state / traffic class must be exported identically
```

## DYN3 — complementary timescales

[`results/DYN3.md`](results/DYN3.md)

A slow model learned regular structure while a finite fast-write store retained rare item-specific exceptions.

Representative result:

```text
slow cortex:
new-item accuracy             0.8243
long regular recall          0.9328
long exception recall        0.0754

surprise-selected H:
new-item accuracy             0.8243
short exception recall       0.9951
long regular recall          0.9630
long exception recall        0.8690

after erasing H:
regular recall               0.9347
exception recall             0.0745
```

The useful lesion:

```text
what became regular knowledge survives H erasure
what remained an idiosyncratic episode dies with H
```

## DYN3B — retain != consolidate

[`results/DYN3B.md`](results/DYN3B.md)

Naive replay lost.

```text
no replay                    0.8243 new-item accuracy
moderately congruent replay  0.7954
congruent replay             0.7932
uniform replay               0.6882
```

The store correctly concentrated on exceptions; replaying those exceptions into the slow statistical model distorted the regularity it was supposed to learn.

> **Worth remembering is not the same as worth training the slow model on repeatedly.**

Replay is demoted until a task actually needs it.

## DYN3C — forgetting becomes allocation

[`results/DYN3C.md`](results/DYN3C.md)

The slow model guided finite memory toward what it still could not regenerate.

At an 800-event delay:

```text
capacity 64:   FIFO exception recall 0.0747   residual cache 0.4415
capacity 128:  FIFO                  0.0747   residual cache 0.7389
capacity 256:  FIFO                  0.0747   residual cache 0.9498
```

The world contained roughly 12% exceptions, yet the residual-guided cache became heavily exception-enriched without receiving an exception label.

This is familiar model-aware residual caching / compression territory, not a novelty claim. But the architectural result is useful:

> **If slow knowledge can regenerate something, scarce exact/dormant memory can be spent elsewhere.**

Forgetting is therefore not only decay. It can be an active consequence of redundancy with slower knowledge.

## DYN4 — dormant H can store an old fast q

[`results/DYN4.md`](results/DYN4.md)

Forty-eight contexts recur after roughly 1,200 events. Fast `q` adapts during a short visit; finite `H` can hold only 12 old context states.

First-event reacquisition:

```text
no H                            0.8065
FIFO                            0.8065
random                          0.8090
slow-model residual H           0.8607
unbounded context table         0.9471
scrambled residual addresses    0.8034
```

So a previously learned fast state can disappear from active dynamics, remain dormant, and later accelerate reacquisition when the same stable context returns.

DYN4B attacked the retention priority:

```text
slow-model residual improvement   0.8607
slow-model raw error               0.8544
mean |q|                           0.8418
final |q|                          0.8378
no H                               0.8065
```

The exact residual formula is not sacred. The robust part is:

> **slow knowledge can help decide which previously learned fast states deserve long retention.**

## DYN5 — remove the exact context ID

[`results/DYN5.md`](results/DYN5.md)

DYN4 used an exact stable context key. DYN5 replaced it with partial/noisy cues shared by families of contexts.

Eager nearest-neighbour reinstatement was harmful:

```text
no H, first five return events       0.7772
eager nearest                        0.6890
cue-only soft retrieval              0.7362
```

Allowing the current consequence to reweight candidate dormant states looked much better:

```text
confirm residual                     0.8368
unbounded confirm                    0.8463
exact-ID diagnostic table            0.9588
```

But the decisive warning was that scrambling dormant states away from their cue keys did not collapse confirmation (`0.8395`).

Therefore DYN5 did **not** earn associative pattern-completion language.

## DYN5B — associative recall killed; rapid mode inference survives

[`results/DYN5B.md`](results/DYN5B.md)

The next attackers removed the allegedly hippocampal machinery:

```text
DYN5 cue+memory confirmation          0.8368
same finite old states, NO cues       0.8414
fixed 9-state grid, NO memory         0.8513
plain fast q adaptation, eta=.60      0.8474
exact-ID table                        0.9588
```

So the useful DYN5 operation is best described as:

> **rapid online mode/state inference from consequence**

not:

> partial sensory cue pattern-completes a specific episodic state.

The large exact-ID upper bound says that genuine recurring-context recognition would be valuable. This mechanism simply did not achieve it.

Do not tune DYN5 until it wins.

---

# What survives now

Twensday has repeatedly been beaten by mature or boring digital mechanisms. That is useful.

The surviving pieces are narrower:

1. fast dynamical state and slow structural trust can do different jobs;
2. active local temporal nonlinearities can create useful coordinates that passive leaks do not;
3. state can live in recurrent traffic, though simple memory does not justify distributed recurrence;
4. internal computation and exported traffic are worth keeping conceptually separate;
5. write speed, retention lifetime and learning speed are different axes;
6. retaining an experience and consolidating/generalizing from it are different decisions;
7. slow knowledge can guide scarce dormant-memory allocation by identifying what it cannot regenerate;
8. a dormant old fast state can accelerate reacquisition when a stable context is genuinely identifiable;
9. the current noisy partial-cue mechanism does not yet identify that context; outcome-conditioned mode inference explains the gain more simply.

The strongest new conceptual sentence is therefore not "we built a hippocampus."

It is:

> **A finite dynamic learner may benefit from spending memory and learning capacity on the residual between what is happening and what its slower knowledge can already explain.**

That sounds like curiosity, but prediction error alone is not enough.

---

# DYN6 — curiosity must survive noisy TV

This is the next active-loop gate.

The agent will choose what part of a world to sample next. Some regions are initially unexplained but learnable. At least one region is deliberately stochastic / irreducible: it can remain surprising forever without teaching the slow model anything useful.

The loop is:

```text
M predicts world
   ↓
residual says what M does not explain
   ↓
agent chooses where to look / interact
   ↓
new signal changes q / M
   ↓
future residual changes
   ↺
```

The raw-error curiosity policy should be expected to fail on the stochastic distractor. That is a feature of the test, not a surprise.

The candidate signal is closer to:

```text
unexplained
AND
my interaction with it is producing learning progress
```

rather than simply:

```text
surprising
```

Mandatory attackers:

```text
uniform/random exploration
count-based exploration
raw prediction-error curiosity
learning-progress curiosity
uncertainty / information-gain style selection where practical
oracle reducible-error selector
```

Kill conditions:

- if residual curiosity camps on irreducible noise, residual alone is rejected;
- if a standard learning-progress / uncertainty policy does everything better, use the standard policy;
- if active selection gives no sample-efficiency or adaptation gain over random/count-balanced sampling, curiosity has not earned a mechanism;
- if a useful policy emerges, then add recurrence / dormant-state memory only when the task demands them.

This is the first gate where the system's own ignorance begins to alter the trajectory of experience that will train it.

# Current sentence

> **Dynamic AI, in the Twensday sense, is a continuously running learner in which information can occupy different temporal roles: active state, short causal trace, dormant fast-write memory, and slow reusable knowledge. The slow system not only predicts; its failures can help allocate finite memory and, if DYN6 survives, may help choose what the system experiences next.**
