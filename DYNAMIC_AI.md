# Dynamic AI — Twensday compass

This is a working research direction, not a novelty claim and not a definition of intelligence.

The question is narrower than "can an AI have hidden state?" RNNs, reservoirs, SSMs, spiking nets, adaptive filters and control systems already do that.

> **Can a useful AI be built around an ongoing dynamical process whose active state changes the meaning of present traffic, whose finite longer-lived memories can preserve/reinstate selected states, and whose slower knowledge changes what future dynamics are easy to express, what deserves further learning, and what the system chooses to experience next?**

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
surprise != learning progress
learning progress != information gain between hypotheses
latent disagreement != observable information when instruments are noisy
```

A deliberately loose sketch is:

```text
q(t+1), y(t) = F(q(t), u(t), peer(t), H(t); M(t), theta(t))
e(t+1)      = E(e(t), local activity, consequence timing)
H(t+1)      = W(H(t), q(t), u(t); M(t))
M(t+1)      = G(M(t), e(t), consequence(t), selected experience)
```

Three reverse arrows are now distinct:

```text
M -> what deserves scarce H / learning capacity?
M -> where is interaction still producing learnable progress?
M -> which experiment should be created to discriminate live explanations?
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

## DYN6 — active curiosity survives; raw residual does not

[`results/DYN6.md`](results/DYN6.md)

The agent chose among six learnable regions and six irreducibly stochastic noisy-TV regions.

Raw prediction-error curiosity failed catastrophically:

```text
noise fraction at 360 steps     0.867
noise fraction at 600 steps     0.920
noise fraction at 1200 steps    0.960
final learnable MSE              0.10898
```

A prequential learning-progress signal compared the current predictor to a lagged copy **on a fresh observation before updating from it**. That standard-family heuristic improved active allocation:

```text
mean interactions to MSE < .01

random                513.1
count-balanced        504.4
uncertainty           487.5
learning progress     350.0
oracle reducible      305.6
```

This is established learning-progress / intrinsic-motivation territory, not a new curiosity algorithm. What matters for Twensday is the architectural result:

> **the learner's current state of knowledge can beneficially change which future observations will train it.**

The limitation matters too. Once the learnable regions are mostly mastered, progress becomes small everywhere and the weak revisit pressure sends the forced-choice agent back toward broad exploration; its noisy-TV fraction rises to `0.411` by 1,200 steps. DYN6 therefore earns scarce-effort allocation, not a permanent oracle of irreducibility.

## DYN7 — research needs experiment value, not merely curiosity

[`results/DYN7.md`](results/DYN7.md)

DYN7 gave the learner 12 competing smooth hypotheses, 21 possible experiments, and five deliberately bad high-noise instruments. The learner maintained an exact Bayesian posterior and chose its next experiment.

The DYN6 learning-progress rule avoided bad instruments but failed as a research selector:

```text
mean experiments to posterior(true) >= .95

coverage                     20.79
random                       21.67
learning progress            33.77
```

Raw predictive variance recreated noisy TV at the experiment level: it selected a bad instrument on `100%` of trials and reached the confidence target in only `25%` of seeds.

Naive model disagreement also failed because it ignored measurement reliability:

```text
final bad-instrument fraction    0.817
confidence-target hit rate       0.708
```

Standard expected information gain won decisively:

```text
expected information gain       9.88 mean experiments
truth-aware oracle              9.02
coverage                       20.79
random                         21.67
learning progress              33.77
model disagreement             33.90
raw predictive variance        54.90
```

So DYN7 kills two tempting slogans:

```text
research = learning-progress curiosity
research = ask where models disagree most
```

The useful object is closer to:

> **Choose an experiment whose possible observations are expected to reduce uncertainty among the explanations still alive, accounting for the reliability of the observation channel.**

That is mature Bayesian experimental-design territory and should be called that.

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
9. the current noisy partial-cue mechanism does not yet identify that context; outcome-conditioned mode inference explains the gain more simply;
10. raw prediction error is a bad exploration objective in stochastic worlds, while prequential learning progress can improve where scarce observations are spent;
11. the knowledge state can close an active loop by changing the future data distribution it experiences;
12. learning progress and scientific experiment value are different; the latter requires reasoning over competing explanations and observation reliability;
13. exact expected information gain beats the current Twensday-style heuristics when the hypothesis set and likelihoods are handed to the machine.

DYN6 gave:

> **Spend scarce interaction where the boundary of current knowledge is moving — where observation is becoming understanding rather than remaining surprise.**

DYN7 adds:

> **When the goal is explanation, do not merely seek progress or disagreement. Create observations expected to distinguish the explanations that remain plausible.**

This is closer to one primitive of research, but DYN7 also cheats heavily.

---

# DYN8 — the hypothesis set must itself become dynamic

DYN7 is easy in one crucial sense:

```text
all candidate explanations are supplied in advance
all likelihoods are known
instrument reliability is known
exact posterior updates are cheap
expected information gain is cheap
```

Real research often begins precisely where that setup fails.

Candidate explanations can:

```text
appear
split
merge
die
become relevant only after an anomaly
need old observations reconsidered
```

and the reliability of the measurement process may itself be uncertain or drifting.

That is where Twensday's original finite evolving-structure idea may finally have a non-decorative job.

A possible DYN8 machine is:

```text
q   fast evidence / current belief state
H   finite anomalous episodes not explained well by current models
M   finite active population of explanatory structures

M predicts observations
        ↓
reliable residual persists
        ↓
H preserves the stubborn case
        ↓
current explanation population cannot absorb it
        ↓
spawn / split / modify a candidate explanation
        ↓
choose discriminative experiment against surviving candidates
        ↓
consequence kills, supports, merges or reshapes candidates
        ↺
```

The key word is **finite**. The machine cannot retain every hypothesis or every anomalous observation forever.

Mandatory attackers:

```text
exact finite-hypothesis expected information gain whenever available
particle filters / sequential Monte Carlo
Bayesian online change-point detection
online mixture / clustering methods
ensemble active learning / BALD-style acquisition
Thompson-style sampling where applicable
simple fixed-size ensembles with ordinary replacement heuristics
```

Kill conditions:

- if a standard particle/ensemble method handles hypothesis birth/death and experiment choice better, use it;
- if `H` anomaly retention does not improve discovery of a missing explanation, remove it;
- if structural growth merely recreates a mixture model badly, say so;
- if dynamic finite structure earns anything, it must do so under a condition exact enumerated Bayesian design cannot cheaply represent: drifting worlds, open-ended candidate models, limited memory, or expensive evaluation.

The concrete DYN8 question is:

> **Can a finite continuously evolving population of candidate explanations preserve most of the experiment-selection value of Bayesian information gain when the hypothesis set itself is not fixed in advance?**

That question reconnects the research loop to Twensday instead of merely renaming Bayesian experimental design.

# Current sentence

> **Dynamic AI, in the Twensday sense, is a continuously running learner in which information can occupy different temporal roles: active state, short causal trace, dormant fast-write memory, and slow reusable knowledge. Slow knowledge can guide scarce memory and exploration. DYN7 shows that research-like experiment choice requires a stronger object than curiosity: explicit competition among explanations plus the expected epistemic value of possible observations. The next wall is whether a finite evolving dynamical system can maintain and revise those explanations when they are not handed to it in advance.**
