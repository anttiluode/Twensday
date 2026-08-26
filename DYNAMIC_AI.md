# Dynamic AI — Twensday compass

This is a working research direction, not a novelty claim and not a definition of intelligence.

The question is narrower than "can an AI have hidden state?" RNNs, reservoirs, SSMs, spiking nets, adaptive filters and control systems already do that.

> **Can a useful AI be built around an ongoing dynamical process whose active state changes the meaning of present traffic, whose finite longer-lived memories preserve only selected information, whose slower explanatory structure can itself change, and whose current knowledge changes what the system chooses to experience next?**

The biological inspiration remains modest:

```text
a system is already in a state when a signal arrives
signals change that state
state changes what later signals do
other evolving systems keep sending traffic
some traffic returns
consequence arrives later
slower structure changes while all of this continues
```

The signal-train grammar is therefore not merely:

```text
input -> function -> output
```

but:

```text
signal perturbs an already-moving system
        ↓
changed system emits new perturbations
        ↓
those perturb other moving systems
        ↓
some return to a system that is no longer in the same state
        ↓
consequence changes what will be easy / remembered / sought next
```

## Current state variables

The original `fast q + slow M` picture became too small.

```text
q   active fast computational / belief state
    what is happening now / what recent traffic currently means

e   short eligibility / causal trace
    which earlier local event can still receive delayed consequence

H   fast-write, selectively retained dormant state / evidence
    information allowed to disappear from q but kept because M
    cannot yet regenerate or explain it

M   slow reusable knowledge / explanatory structure
    what the system currently knows how to predict, compress,
    regenerate, or explain

theta (optional)
    expression / output state distinct from internal computation
```

These are not merely different leak constants.

```text
write speed != retention time
retain != replay
remember exactly != generalize from
active state != dormant state
internal computation != exported traffic
prediction error != useful curiosity
surprise != learning progress
learning progress != experiment value
latent model disagreement != observable information
information gain inside M != evidence that M is adequate
```

A deliberately loose sketch is:

```text
q(t+1), y(t) = F(q(t), u(t), peer(t), H(t); M(t), theta(t))
e(t+1)      = E(e(t), local activity, consequence timing)
H(t+1)      = W(H(t), q(t), u(t); M(t))
M(t+1)      = G(M(t), e(t), consequence(t), H(t), selected experience)
```

The reverse arrows now matter as much as the forward computation:

```text
M -> what deserves scarce H?
M -> where is interaction still producing learnable progress?
M -> which experiment best discriminates live explanations?
H -> when is current M inadequate enough that a new explanation should exist?
```

---

# Receipts

## DYN0 — temporal state != structural trust

[`results/DYN0.md`](results/DYN0.md)

A hidden condition was identifiable from temporal correlation but not current sensor values. Fast temporal state plus delayed structural allocation reached `0.9646`; Fixed Share reached `0.9674`; stateless and scrambled-address controls sat near chance.

Survived:

```text
fast temporal state and slow structural trust can do different jobs
stable addressability matters for delayed structural credit
```

Did not survive:

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

Passive multiscale traces alone were not enough. Active local temporal nonlinearities plus slow selection created useful coordinates, but the exact statistic still won badly.

## DYN2 — circulating state is possible, not necessary

[`results/DYN2.md`](results/DYN2.md)

A two-point `A -> B -> A` loop genuinely stored a binary state. Cutting only the return path gave:

```text
before cut    0.9765
post cut      0.5015
```

But a single nonlinear bistable scalar reached `0.9891`. Recurrence showed one place state can live, not why a one-bit task needs distributed circulation.

The remaining AIS/axon abstraction is only:

```text
internal state exists
        !=
it must be exported
        !=
every internal state / traffic class must be exported identically
```

## DYN3 / DYN3B / DYN3C — memory becomes an economy

[`results/DYN3.md`](results/DYN3.md) · [`results/DYN3B.md`](results/DYN3B.md) · [`results/DYN3C.md`](results/DYN3C.md)

A slow model learned regular structure while finite fast-write memory retained item-specific exceptions.

DYN3B killed naive replay:

```text
no replay                    0.8243 new-item accuracy
moderately congruent replay  0.7954
congruent replay             0.7932
uniform replay               0.6882
```

> **Worth remembering is not the same as worth training the slow model on repeatedly.**

DYN3C made forgetting an allocation decision. At an 800-event delay:

```text
capacity 64:   FIFO exception recall 0.0747   residual cache 0.4415
capacity 128:  FIFO                  0.0747   residual cache 0.7389
capacity 256:  FIFO                  0.0747   residual cache 0.9498
```

The slow model became a compression oracle for finite memory:

> **If M can regenerate something, scarce exact H can be spent elsewhere.**

## DYN4 / DYN4B — dormant H can preserve an old fast q

[`results/DYN4.md`](results/DYN4.md) · [`results/DYN4B.md`](results/DYN4B.md)

Forty-eight contexts recur after roughly 1,200 events. A fast state learned during one visit can disappear from active dynamics, remain dormant, and accelerate reacquisition later.

```text
no H                            0.8065
FIFO                            0.8065
random                          0.8090
slow-model residual H           0.8607
unbounded context table         0.9471
scrambled residual addresses    0.8034
```

Slow-model residual value beat simple `|q|` retention heuristics modestly.

## DYN5 / DYN5B — associative recall claim killed

[`results/DYN5.md`](results/DYN5.md) · [`results/DYN5B.md`](results/DYN5B.md)

Partial/noisy cues appeared to reinstate a useful dormant state, but attackers removed the alleged associative mechanism:

```text
DYN5 cue+memory confirmation          0.8368
same finite old states, NO cues       0.8414
fixed 9-state grid, NO memory         0.8513
plain fast q adaptation, eta=.60      0.8474
exact-ID table                        0.9588
```

So the useful operation was **rapid online mode inference from consequence**, not earned hippocampal pattern completion.

## DYN6 — active curiosity survives; raw residual does not

[`results/DYN6.md`](results/DYN6.md)

The agent chose among six learnable regions and six irreducibly stochastic noisy-TV regions.

Raw prediction-error curiosity became addicted to noise:

```text
noise fraction at 360 steps     0.867
noise fraction at 600 steps     0.920
noise fraction at 1200 steps    0.960
```

A fresh-sample prequential learning-progress signal improved sample allocation:

```text
mean interactions to MSE < .01

random                513.1
count-balanced        504.4
uncertainty           487.5
learning progress     350.0
oracle reducible      305.6
```

Survived:

> **The learner's current knowledge can beneficially change which future observations will train it.**

## DYN7 — research needs experiment value, not merely curiosity

[`results/DYN7.md`](results/DYN7.md)

DYN7 gave the learner 12 competing smooth hypotheses, 21 experiments, five bad instruments, and an exact Bayesian posterior.

Learning progress failed as a research selector:

```text
coverage                     20.79 experiments to .95 posterior(true)
random                       21.67
learning progress            33.77
```

Raw predictive variance selected bad instruments on `100%` of trials. Naive model disagreement also failed because it ignored whether the measurement channel could resolve the disagreement.

Standard expected information gain won:

```text
expected information gain       9.88
truth-aware oracle              9.02
coverage                       20.79
random                         21.67
learning progress              33.77
model disagreement             33.90
raw predictive variance        54.90
```

Survived:

> **When the goal is explanation, create observations expected to distinguish the explanations still plausible, accounting for observation reliability.**

That is mature Bayesian experimental design, not a Twensday invention.

## DYN8 — the hypothesis set itself can change, but the attackers still matter

[`results/DYN8.md`](results/DYN8.md)

DYN8 removes DYN7's biggest cheat: the correct explanation is deliberately **absent** from every finite initial population.

A small grammar of primitives is still supplied. The hidden law is a three-term combination, but the combination/catalog is not given to the six-slot learner.

`H` has only eight stubborn-residual slots and is used only to **propose new explanatory structure**. All finite attackers receive the same compact accumulated evidence for scoring candidates.

### DYN8A — 364 possible explanations, six live slots

32 seeds:

```text
method                   structure hit   MSE hit   final truth present
residual_info                 15.59        15.59          .906
residual_hybrid               12.78        12.78          .969
residual_random               14.28        14.28          .969
beam_info                      9.62         9.66         1.000
smc_info                      15.94        15.94          .938
smc_hybrid                    14.12        14.12         1.000
fixed_info                    41.00        41.00          .000
catalog_info                   4.50         3.75         1.000
```

The important failure is:

```text
residual_info       15.59
residual_random     14.28
```

Pure information-aware sampling became **worse than random** while the live explanation population was misspecified.

A small coverage reserve rescued it:

```text
residual_hybrid     12.78
```

So:

> **Information gain over M is only information gain inside the explanatory space M currently contains. If M may be wrong, some observation budget must remain capable of producing evidence from outside that focus.**

This matches the known active-learning / adaptive-design misspecification problem rather than constituting a new one.

### DYN8B — exact DYN7 EIG, 120 possible explanations, six live slots

24 seeds:

```text
method                   structure hit   MSE hit   final truth present
residual_exact                 7.25         7.25         1.000
residual_random                8.12         8.12         1.000
beam_exact                     5.75         5.75         1.000
smc_exact                      8.29         8.38          .958
fixed_exact                   31.00        31.00          .000
catalog_exact                  4.12         3.04         1.000
```

This gives a narrow positive result:

```text
correct model absent initially
6 live model slots
120 possible structures
residual birth + exact EIG finds the exact hidden structure in all 24 seeds
```

It is about 11% faster than the same residual birth with random observations and modestly faster than the equal-slot SMC attacker here.

But the stronger claim dies:

```text
predictive target
catalog exact      3.04 observations
beam exact         5.75
residual exact     7.25
smc exact          8.38
```

The finite Twensday machine does **not** preserve most of full enumerated Bayesian-design efficiency in this gate. Exhaustive local beam search also beats stubborn-H proposal search.

What `H` earned is only:

> **A tiny set of stubborn residuals can be a useful directed proposal mechanism for missing model structure under a slot budget; it is not yet a superior inference method.**

---

# What survives now

Twensday has repeatedly been beaten by mature or boring mechanisms. That remains the useful part of the process.

The surviving pieces are narrower:

1. fast dynamical state and slow structural trust can do different jobs;
2. active local temporal nonlinearities can create useful coordinates that passive leaks do not;
3. state can live in recurrent traffic, though simple memory does not justify distributed recurrence;
4. internal computation and exported traffic are worth keeping conceptually separate;
5. write speed, retention lifetime and learning speed are different axes;
6. retaining an experience and generalizing from it are different decisions;
7. slow knowledge can guide scarce dormant-memory allocation by identifying what it cannot regenerate;
8. dormant fast state can accelerate reacquisition when a stable context is genuinely identifiable;
9. the current partial-cue mechanism did not earn associative recall;
10. raw prediction error is a bad exploration objective in stochastic worlds;
11. prequential learning progress can improve where scarce learning observations are spent;
12. the knowledge state can close an active loop by changing the future data distribution it experiences;
13. learning progress and scientific experiment value are different;
14. expected information gain is the correct mature answer in DYN7 when the hypothesis set is fixed and adequate;
15. a finite explanation population can grow a previously absent correct model inside a supplied grammar;
16. information gain can become myopic when the current explanation population is misspecified;
17. stubborn residuals can guide model birth modestly, but exhaustive beam search remains stronger when full historical evidence is cheaply available.

The research loop now has two competing obligations:

```text
EXPLOIT CURRENT M
choose measurements that discriminate live explanations

AND

ALLOW M TO BE WRONG
preserve enough off-model evidence / exploration
that a missing explanation can be born
```

That is a substantially better formulation than "curiosity machine."

---

# DYN9 — can old evidence survive for theories that do not exist yet?

DYN8 still cheats in exactly the place where DYN3C may become useful again.

Every new explanation born late can immediately be scored against **all historical evidence** through compact per-action sufficient statistics.

In a more realistic open-ended model space that may be impossible.

A theory born today faces a temporal problem:

```text
this explanation did not exist when yesterday's evidence arrived
I did not know which statistics it would need
I cannot keep all raw history forever

so:
which old observations were important enough to preserve
for explanations that had not yet been invented?
```

That is the first clean place where the DYN3C memory result and DYN8 explanatory growth truly meet.

The DYN9 gate should remove the all-history sufficient-statistics cheat.

Give each learner the same hard memory budget and let a missing explanation appear only after enough contradictory evidence accumulates.

Attackers must include:

```text
full-history beam search                 diagnostic upper bound
large replay buffer
uniform reservoir sampling
recent-window replay
coreset / leverage / influence-style selection where applicable
SMC / resample-move without special H
stubborn residual H
H + diversity / coverage rather than residual alone
```

The decisive question is:

> **Can a small memory chosen before a hypothesis exists retain the evidence that a late-born hypothesis will need to be evaluated?**

Kill conditions:

- if ordinary reservoir/coreset memory evaluates late-born models as well as residual H, use the ordinary method;
- if residual H fills with noisy-TV-like exceptions again, kill raw residual retention in this role;
- if full-history beam remains necessary, then finite forgotten-history model birth is still unsolved;
- if small H preserves most of the useful historical discrimination, then DYN3C's forgetting rule has finally earned a concrete research-machine job.

# Current sentence

> **Dynamic AI, in the Twensday sense, is becoming less a special neuron and more a finite temporal economy: active state, causal traces, selectively retained evidence, evolving explanatory structure, and actions chosen by what is currently known. DYN8 shows that explanations can be born online and then guide experiment choice, but also that information gain becomes dangerous when its current model class is incomplete. The next wall is whether finite memory can preserve the right old evidence for explanations that do not exist yet.**
