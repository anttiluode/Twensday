# Dynamic AI — Twensday compass

This is a working research direction, not a novelty claim and not a definition of intelligence.

The question is narrower than "can an AI have hidden state?" RNNs, reservoirs, SSMs, spiking nets, adaptive filters and control systems already do that.

> **Can a useful AI be built around an ongoing dynamical process whose active state changes the meaning of present traffic, whose finite memory preserves selected information across incompatible timescales, whose explanatory structure can itself change, and whose current knowledge changes what the system chooses to experience next?**

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

The useful abstraction is therefore not merely:

```text
input -> function -> output
```

but:

```text
signal perturbs an already-moving system
        ↓
changed system emits / predicts / acts
        ↓
consequence arrives
        ↓
fast state changes
memory allocation changes
slow explanatory structure may change
future observations/actions change
        ↺
```

## Current state variables

```text
q   active fast computational / belief state
    what is happening now / what recent traffic currently means

e   short eligibility / causal trace
    which earlier event can still receive delayed consequence

H   finite fast-write retained state / evidence
    information allowed to leave q but kept because it may matter later

M   slow reusable knowledge / explanatory structure
    what the system currently predicts, compresses, regenerates or explains

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
latent disagreement != observable information
information gain inside M != evidence that M is adequate
large residual != evidence worth retaining
model failure != measurement failure
```

A deliberately loose sketch is:

```text
q(t+1), y(t) = F(q(t), u(t), peer(t), H(t); M(t), theta(t))
e(t+1)      = E(e(t), local activity, consequence timing)
H(t+1)      = W(H(t), q(t), u(t), reliability; M(t))
M(t+1)      = G(M(t), e(t), consequence(t), H(t), selected experience)
```

The reverse arrows matter as much as the forward computation:

```text
M -> what deserves scarce H?
M -> where is interaction still producing learnable progress?
M -> which experiment best discriminates live explanations?
H -> when is current M inadequate enough that a new explanation should exist?
reliability -> is a stubborn residual about the world or about the sensor?
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

Did not survive: a claim to a new optimizer.

## DYN1 — active temporal listeners

[`results/DYN1.md`](results/DYN1.md)

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

A two-point `A -> B -> A` loop genuinely stored a binary state; cutting only the return path dropped `0.9765 -> 0.5015`. But a single nonlinear bistable scalar reached `0.9891`.

Recurrence showed one place state can live, not why simple memory needs distributed circulation.

The remaining AIS/axon abstraction is only:

```text
internal state exists
        !=
it must be exported
        !=
every internal state must be exported identically
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

DYN3C made forgetting an allocation decision. At an 800-event delay, residual-guided finite memory preserved rare exceptions much better than FIFO.

What survived:

> **If M can regenerate something, scarce exact H can be spent elsewhere.**

## DYN4 / DYN4B — dormant H can preserve an old fast q

[`results/DYN4.md`](results/DYN4.md) · [`results/DYN4B.md`](results/DYN4B.md)

A fast state learned during one context visit could disappear from active dynamics, remain dormant, and later accelerate reacquisition.

```text
no H                            0.8065
FIFO                            0.8065
random                          0.8090
slow-model residual H           0.8607
unbounded context table         0.9471
scrambled residual addresses    0.8034
```

## DYN5 / DYN5B — associative recall claim killed

[`results/DYN5.md`](results/DYN5.md) · [`results/DYN5B.md`](results/DYN5B.md)

Attackers removed the alleged hippocampal mechanism:

```text
DYN5 cue+memory confirmation          0.8368
same finite old states, NO cues       0.8414
fixed 9-state grid, NO memory         0.8513
plain fast q adaptation, eta=.60      0.8474
exact-ID table                        0.9588
```

The useful operation was **rapid online mode inference from consequence**, not earned associative pattern completion.

## DYN6 — active curiosity survives; raw residual does not

[`results/DYN6.md`](results/DYN6.md)

Raw prediction-error curiosity became addicted to stochastic noisy-TV regions. Fresh-sample prequential learning progress improved where observations were spent:

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

Learning progress failed as a research selector. Raw variance and naive model disagreement were fooled by noisy instruments. Standard expected information gain won decisively:

```text
expected information gain       9.88 experiments to .95 posterior(true)
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

## DYN8 — explanations can be born, but EIG can become myopic

[`results/DYN8.md`](results/DYN8.md)

The correct explanation was deliberately absent from every finite initial population. A grammar of primitives was supplied, but the correct combination was not.

In the exact-EIG 120-model world with only six live model slots:

```text
catalog exact      3.04 observations to predictive target
beam exact         5.75
residual exact     7.25
SMC exact          8.38
```

Residual-guided birth + exact EIG nevertheless constructed the exact hidden model in all 24 seeds.

The stronger claim died: finite Twensday did not preserve most of full enumerated Bayesian-design efficiency, and exhaustive beam search remained stronger when cheap.

The harder 364-model world exposed a separate failure:

```text
residual + information gain    15.59
residual + random              14.28
residual + info + coverage     12.78
```

Information gain over a wrong live model class can become myopic: it asks brilliant questions inside an inadequate explanatory space.

Survived:

> **An open explanation population needs both epistemic exploitation of current M and enough off-model traffic for M to be shown wrong.**

## DYN9 — future theories need old evidence, but raw residual memory loses

[`results/DYN9.md`](results/DYN9.md)

DYN9 removed DYN8's all-history sufficient-statistics cheat. During the historical phase the true theory does not exist and no future hypothesis can collect custom statistics. Only a tiny raw memory survives. Later, candidate explanations are born and must be evaluated from that retained past.

In the hard gate:

```text
16 grammar primitives
hidden order 4
1820 possible future theories
6 future live slots
150 historical observations
only 6 raw memories retained
```

### Clean forgotten history

```text
method                 truth@6   top1   pre-EIG MSE   post-EIG MSE
reservoir                .969     .875      .2694          .0372
recent                  1.000     .750      .3916          .0000
leverage                1.000     .844      .2211          .0000
raw residual             .750     .406     1.0154          .5233
stubborn + diverse      1.000     .906      .1283          .0000
```

Under extreme clean memory pressure, persistent + diverse mismatch earns a narrow positive result: it preserves the future true theory slightly better than generic reservoir and gives the best pre-follow-up predictive quality among the generic finite memories.

### Contaminated forgotten history

With 10% corrupted historical observations:

```text
method                 truth@6   top1   post-EIG MSE   corruption in memory
reservoir                .938     .750      .1242              .104
leverage                 .906     .625      .1415              .089
raw residual             .625     .281      .7064              .234
stubborn + diverse       .906     .719      .2030              .172
```

The ranking flips. Representative reservoir memory is safer than stubborn memory.

Raw residual is decisively dead in this role: it turns a 10% corruption stream into roughly 23% of retained memory.

### DYN9D — static split memory also loses

A fixed half-representative / half-stubborn memory sounded sensible. It did not survive.

In the hard six-slot contaminated world, all tested split thresholds were worse than reservoir/leverage; the best split still produced lower future-theory inclusion and worse post-EIG error.

So:

```text
all representative memory      robust default
all stubborn memory            specialist, clean misspecification only
raw surprise memory            bad
static 50/50 split             bad
```

What survives is narrower:

> **Evidence can be valuable to an explanation that did not exist when the evidence arrived, but selecting such evidence requires distinguishing persistent model inadequacy from bad observations.**

---

# What survives now

Twensday has repeatedly been beaten by mature or boring mechanisms. That remains the useful part of the process.

1. Fast dynamical state and slow structural trust can do different jobs.
2. Active local temporal nonlinearities can create useful coordinates that passive leaks do not.
3. State can live in recurrent traffic, though simple memory does not justify distributed recurrence.
4. Internal computation and exported traffic are worth keeping conceptually separate.
5. Write speed, retention lifetime and learning speed are different axes.
6. Retaining an experience and generalizing from it are different decisions.
7. Slow knowledge can help allocate scarce dormant memory, but raw residual magnitude is not a general retention rule.
8. Dormant fast state can accelerate reacquisition when context is genuinely identifiable.
9. The current partial-cue mechanism did not earn associative recall.
10. Raw prediction error is a bad exploration objective in stochastic worlds.
11. Prequential learning progress can improve where scarce learning observations are spent.
12. Knowledge can close an active loop by changing the future data distribution it experiences.
13. Learning progress and scientific experiment value are different.
14. Expected information gain is the mature answer when the hypothesis set is fixed and adequate.
15. A finite explanation population can grow a previously absent correct model inside a supplied grammar.
16. Information gain can become myopic when the current explanation population is misspecified.
17. Stubborn residuals can guide model birth modestly, but beam search remains stronger when full historical evidence is cheap.
18. Tiny retained memories can support hypotheses invented later.
19. Persistent + diverse residual memory can help under extreme clean model misspecification, but representative memory is more robust to corruption.
20. A static representative/anomaly split does not solve the tradeoff.

The research loop now has three competing obligations:

```text
EXPLOIT CURRENT M
choose measurements that discriminate live explanations

ALLOW M TO BE WRONG
keep enough off-model evidence / exploration
for missing explanations to appear

DO NOT CONFUSE BAD DATA WITH NEW PHYSICS
learn which observation channels deserve trust
```

---

# DYN10 — is the model wrong, or is the source wrong?

DYN9 gives the learner known nominal instrument noise and then injects corruption behind its back. A useful open-ended learner cannot assume every stubborn residual deserves structural change.

A persistent mismatch can mean:

```text
A. current explanatory M is inadequate
B. the measurement/source is unreliable
C. the world changed regime
```

The next gate should make source reliability latent.

A candidate machine keeps separate slow variables for:

```text
M          explanatory structure
r_j        trust / reliability of observation source j
H          finite evidence whose interpretation is still unresolved
```

Then evidence should be able to change either:

```text
residual -> lower trust in source
```

or:

```text
residual -> modify / grow explanation
```

without deciding in advance which one is responsible.

Mandatory attackers:

```text
Gaussian model with fixed known variance
Student-t / heavy-tailed robust likelihood
explicit contamination-mixture likelihood
robust regression / Huber-style loss
RANSAC-style fitting where appropriate
Bayesian per-source reliability estimation
representative reservoir / coreset memory
current persistent-residual memory
```

Kill conditions:

- if standard robust statistics solve the ambiguity and special H adds nothing, use robust statistics;
- if source-trust learning merely becomes ordinary heteroscedastic regression, call it that;
- if the machine changes its theory whenever one bad source misbehaves, it fails;
- if it distrusts a reliable source that is exposing a genuine missing law, it also fails.

The decisive question is:

> **Can a finite evolving learner decide whether stubborn evidence means “change my explanation” or “trust this observation channel less”?**

# Current sentence

> **Dynamic AI, in the Twensday sense, is becoming a finite temporal economy rather than a special neuron: active state, causal traces, selectively retained evidence, evolving explanatory structure, learned observation trust, and actions chosen by what is currently known. DYN9 shows that old evidence can support theories invented later, but also kills raw surprise as a memory rule. The next wall is whether the machine can separate failure of its model from failure of its evidence source.**
