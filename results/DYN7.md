# DYN7 — curiosity becomes experiment choice

DYN6 showed that a prequential learning-progress signal can allocate observations toward learnable parts of a world while raw prediction error becomes addicted to irreducible noise.

That is not yet research.

A research-like learner often faces several explanations that all fit what has been seen so far. The useful question is not simply:

```text
where am I still learning?
```

but:

```text
which measurement / intervention would make the live explanations disagree in a way I can actually observe reliably?
```

DYN7 attacks that distinction directly.

This is **not** a novelty claim. Bayesian optimal experimental design, expected information gain, BALD-style acquisition, active learning, and query-by-committee are established methods. The point is to determine whether DYN6-style learning progress is enough for the next Twensday step, or whether research-like experiment choice needs a different object.

## World

Each seed creates:

```text
12 competing smooth hypotheses / laws
21 possible experiments / measurement locations
```

The hypotheses come from one nearby smooth-function family, so many ordinary observations leave several explanations plausible.

Five of the 21 experiments are deliberately bad instruments:

```text
ordinary instrument sigma = 0.28
bad instrument sigma      = 1.40
```

The bad measurements can look highly variable even when they carry little useful epistemic information.

As in DYN6, each policy receives the same **nth observation from each action** through deterministic action-specific random streams.

The learner maintains an exact Bayesian posterior over the 12 candidate hypotheses.

## Policies / attackers

- `random`: uniform experiment choice.
- `coverage`: choose a least-sampled experiment.
- `raw_predictive_variance`: model disagreement + known instrument variance. This deliberately conflates epistemic and aleatoric uncertainty.
- `learning_progress`: DYN6-style prequential progress attached to experiment locations.
- `model_disagreement`: posterior-weighted variance of candidate-hypothesis predictions, a simple query-by-committee-like score that does **not** discount instrument noise.
- `expected_information_gain`: expected posterior entropy reduction under the current predictive mixture, computed with Gauss-Hermite quadrature.
- `oracle_true_information`: truth-aware diagnostic upper bound using the actual hidden hypothesis.

The primary sample-efficiency target is:

```text
posterior probability of the true hypothesis >= 0.95
```

## GitHub Actions result

48 seeds, 12 hypotheses, 21 possible experiments, maximum 60 experiments.

### After 5 experiments

```text
policy                     accuracy   P(true)   entropy   noisy fraction
random                       0.458      0.376     1.332       0.246
coverage                     0.500      0.407     1.270       0.242
raw predictive variance      0.167      0.138     2.238       1.000
learning progress            0.562      0.410     1.267       0.133
model disagreement           0.396      0.321     1.586       0.708
expected information gain    0.896      0.749     0.579       0.000
oracle true information      0.792      0.715     0.676       0.000
```

### After 20 experiments

```text
policy                     accuracy   P(true)   entropy   noisy fraction
random                       0.917      0.818     0.364       0.252
coverage                     0.854      0.839     0.309       0.237
raw predictive variance      0.396      0.284     1.736       1.000
learning progress            0.729      0.641     0.703       0.106
model disagreement           0.792      0.627     0.860       0.783
expected information gain    0.958      0.943     0.087       0.000
oracle true information      1.000      0.978     0.072       0.052
```

### After 60 experiments

```text
policy                     accuracy   P(true)   entropy   noisy fraction
random                       1.000      0.981     0.053       0.220
coverage                     0.979      0.965     0.049       0.237
raw predictive variance      0.667      0.542     0.971       1.000
learning progress            1.000      0.970     0.071       0.096
model disagreement           0.896      0.850     0.303       0.817
expected information gain    1.000      0.994     0.018       0.000
oracle true information      1.000      1.000     0.001       0.270
```

### Sample efficiency to posterior(true) >= 0.95

```text
policy                     mean hit   median   hit rate   final accuracy
random                       21.67      16       0.958        1.000
coverage                     20.79      15       0.958        0.979
raw predictive variance      54.90      61       0.250        0.667
learning progress            33.77      36       0.896        1.000
model disagreement           33.90      34       0.708        0.896
expected information gain     9.88       6       1.000        1.000
oracle true information       9.02       6.5     1.000        1.000
```

A hit time of `61` is the censored value for a run that did not reach the threshold inside 60 experiments.

## Kill 1 — learning progress is not experiment design

DYN6's learning-progress signal transfers badly to this task.

It does something valuable: it strongly avoids the bad high-noise instruments (`9.6%` noisy experiments over the full run). But it reaches 95% posterior confidence in the true explanation much more slowly than passive baselines:

```text
coverage             20.79
random               21.67
learning progress    33.77
```

Why?

Because these are different questions:

```text
learning progress:
where has recent interaction been making my predictor better?

experiment design:
where will the competing explanations make observably different predictions now?
```

A location can still be learnable without being the most discriminative experiment for the hypotheses that remain alive.

So this claim is rejected:

> a learning-progress curiosity mechanism is sufficient to make a research-like experiment selector

## Kill 2 — variance is not epistemic value

Raw predictive variance selects a bad instrument on **100%** of experiments.

It is the DYN6 noisy-TV failure again at research scale:

```text
high observed variance
!=
useful information about which explanation is true
```

The policy reaches the confidence target in only 25% of seeds and finishes at only 66.7% hypothesis-identification accuracy.

So predictive variance must be decomposed. Observation noise / aleatoric uncertainty is not automatically epistemic uncertainty.

## Kill 3 — disagreement alone is also insufficient

Naive posterior model disagreement sounds much closer to science:

```text
ask where the models disagree
```

But here it spends `81.7%` of its final experiment budget on the deliberately noisy instruments and reaches the confidence target in only `70.8%` of seeds.

The problem is subtle but important:

> **models disagreeing about the latent outcome is not enough if the available measurement is too noisy to resolve that disagreement.**

Research needs the expected information in the *observation channel*, not merely disagreement behind it.

## What wins

Standard expected information gain is the clear winner:

```text
mean experiments to 95% confidence

expected information gain      9.88
truth-aware oracle             9.02
coverage                      20.79
random                        21.67
learning progress             33.77
model disagreement            33.90
raw predictive variance       54.90
```

It comes within `0.86` experiments of the truth-aware diagnostic while never selecting a deliberately bad noisy instrument in this constructed run.

The winning object is essentially:

```text
current posterior over explanations
        ↓
for each possible experiment:
    imagine possible observations
    estimate how much each would change posterior uncertainty
        ↓
choose maximum expected information gain
```

That is mature Bayesian experimental design, and it should be called that.

## What DYN7 changes in Twensday

The phrase "curiosity machine" is still useful, but it now splits into at least two mechanisms:

```text
EXPLORATION / SKILL ACQUISITION
Where is interaction still producing learning progress?

RESEARCH / EXPLANATION DISCRIMINATION
Which experiment is expected to reduce uncertainty among live explanations?
```

DYN6 earned the first operation in its toy world.

DYN7 says the second operation is different, and a standard expected-information-gain rule wins decisively.

That is a useful boundary rather than a disappointment.

## The dynamic loop survives

The experiment does preserve one part of the broader Twensday picture:

```text
slow explanatory state M
        ↓
changes which intervention is selected
        ↓
selected intervention changes incoming traffic
        ↓
consequence changes fast evidence / belief state
        ↓
slow explanatory state M changes
        ↺
```

The system is no longer passively consuming a dataset. Its current knowledge changes what evidence it causes itself to encounter next.

But the useful acquisition function here is not uniquely Twensday. It is standard Bayesian information gain.

## The next place Twensday might actually matter

DYN7 cheats in a much more interesting way than DYN6 did:

```text
all 12 candidate explanations are handed to the learner
all likelihoods are known
all instrument noise levels are known
exact posterior updates are cheap
exact expected information gain is cheap
```

Real research rarely has that luxury.

So the next wall should not be "invent another curiosity bonus."

It should ask whether the other Twensday machinery becomes useful when exact Bayesian experimental design becomes structurally awkward:

```text
explanations appear, split, merge, and die online
world dynamics drift
measurement reliability must itself be learned
only a finite active set of explanatory states can be maintained
old anomalous episodes H may need to be reconsidered when a new model appears
```

A plausible DYN8 question is therefore:

> **Can a finite continuously evolving population of candidate explanations approximate the useful behavior of Bayesian information gain when the hypothesis set itself is not fixed?**

The boring attackers are obvious and mandatory:

```text
particle filters / sequential Monte Carlo
Bayesian online change-point detection
ensemble active learning / BALD-style acquisition
Thompson sampling where applicable
online mixture / clustering methods
exact finite-hypothesis EIG whenever it remains computable
```

That reconnects experiment choice to the original Twensday concern with finite evolving structure instead of merely renaming Bayesian design.
