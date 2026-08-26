# DYN6 — curiosity must survive noisy TV

DYN3C/DYN4 suggested a useful reverse arrow:

```text
slow knowledge M
    -> what still deserves scarce memory / learning capacity?
```

DYN6 makes that arrow active. The learner is no longer handed a fixed stream. It chooses which region of the world to sample next.

This is **not** a novelty claim. Prediction-error curiosity, learning-progress intrinsic motivation, uncertainty/information-gain exploration, and the noisy-TV failure mode are established territory. The purpose of DYN6 is narrower: determine which of those signals, if any, earns a place in the Twensday loop.

## World

Each seed contains:

```text
6 learnable regions
    x -> fixed low-dimensional function + small observation noise

6 stochastic regions
    x -> fresh high-variance random output
    (the noisy TVs)
```

The slow model in every region is the same recursive least-squares predictor.

Each policy receives the same **nth observation from each region**. Region streams are independently seeded, so changing the exploration schedule does not secretly change the data available from a region.

The diagnostic objective is generalization error on the six learnable regions. Samples spent on stochastic regions are counted separately.

## The important progress signal

DYN6 does **not** call this learning progress:

```text
loss before training on sample
-
loss after training on the same sample
```

That would reward memorizing noise.

Instead every region carries a slow lagged copy of its predictor. On a fresh observation, before either predictor learns from it:

```text
progress = loss(lagged model, fresh sample)
         - loss(current model, fresh sample)
```

If recent interaction has produced a genuinely better predictor, the current model should beat its older self on new data.

The selection score is positive EMA-smoothed prequential progress plus a weak revisit term.

## Attackers

- `random`: uniform random region.
- `count_balanced`: always sample a least-visited region.
- `uncertainty`: average posterior/predictor geometry; residual magnitude is ignored.
- `raw_error`: choose the region with largest recent prediction error.
- `learning_progress`: choose where the current predictor is beating its lagged self on fresh observations.
- `oracle_reducible`: diagnostic upper bound that knows which regions are learnable and where their true remaining model error is largest.

## GitHub Actions result

48 seeds, 6 learnable + 6 stochastic regions, maximum 1,200 interactions.

At 360 interactions:

```text
policy                learnable MSE    noise fraction
random                   0.01643          0.498
count_balanced           0.01547          0.500
uncertainty              0.01458          0.500
raw_error                0.10925          0.867
learning_progress        0.01339          0.142
oracle_reducible         0.00750          0.133
```

At 600 interactions:

```text
policy                learnable MSE    noise fraction
random                   0.00823          0.501
count_balanced           0.00798          0.500
uncertainty              0.00791          0.500
raw_error                0.10925          0.920
learning_progress        0.00495          0.178
oracle_reducible         0.00379          0.080
```

At 1,200 interactions:

```text
policy                learnable MSE    noise fraction
random                   0.00400          0.502
count_balanced           0.00392          0.500
uncertainty              0.00381          0.500
raw_error                0.10898          0.960
learning_progress        0.00251          0.411
oracle_reducible         0.00144          0.040
```

Sample efficiency to reach mean learnable-region MSE `< 0.01`:

```text
policy                  mean hit    median hit    hit rate
random                     513.1        420          0.938
count_balanced             504.4        435          0.958
uncertainty                487.5        420          0.958
raw_error                 1230.0       1230          0.000
learning_progress          350.0        300          0.979
oracle_reducible           305.6        240          1.000
```

`1230` is the censored value for a policy that did not hit the target inside the 1,200-step run.

## Kill: residual alone is dead

The raw residual policy is not merely a little worse.

It becomes a noisy-TV addict:

```text
noise fraction
360 steps     0.867
600 steps     0.920
1200 steps    0.960
```

Its learnable-region MSE remains around `0.109` while every non-residual attacker keeps improving.

So this Twensday rule is rejected:

> spend attention where prediction error is largest

DYN3C's residual was useful for retaining **finite exact exceptions** in a world where those exceptions were memorizable. That does not license residual magnitude as an exploration objective.

## What earned itself

The fresh-sample learning-progress heuristic reaches the target error roughly:

```text
32% sooner than random
31% sooner than count-balanced
28% sooner than uncertainty
```

and comes within about 44 interactions of the oracle mean hit time (`350.0` versus `305.6`).

That is enough to preserve this arrow:

```text
M changes
   -> estimated learning progress changes
       -> sampling policy changes
           -> future training data changes
               -> M changes again
```

For the first time in this line, the learner's current state of knowledge changes the trajectory of experience that will train it, and that active choice improves sample efficiency in the constructed world.

## Important limitation: this is not permanent boredom

At 360 steps the learning-progress policy spends only `14.2%` of samples on stochastic regions. By 1,200 steps that rises to `41.1%`.

That is not renewed fascination with prediction error. Once the learnable regions are mostly mastered, estimated progress becomes small everywhere. The deliberately weak revisit pressure then pushes the forced-choice agent back toward broad exploration.

So DYN6 has earned:

> **allocate scarce learning effort toward recent reducible improvement**

It has **not** earned:

> permanently classify every source as learnable or irreducible and never look again

A world that can change should probably revisit old conclusions anyway. The rate and trigger for that revisit remain open.

## Literature boundary

The result is intentionally placed inside existing intrinsic-motivation work rather than presented as a new curiosity algorithm.

- Oudeyer & Kaplan (2007), *What is intrinsic motivation? A typology of computational approaches* — learning progress as a central intrinsic-motivation family.
- Pathak et al. (2017), *Curiosity-driven Exploration by Self-supervised Prediction* — prediction-error curiosity with representation choices intended to suppress uncontrollable factors.
- Jarrett et al. (2023), *Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments* — explicit treatment of stochastic/noisy-TV failure of predictive-error bonuses.
- Poli et al. (2022) and Ten et al. (2021) — human exploration results in which learning progress helps explain allocation of attention across learnable and noisy/difficult activities.
- Dubey & Griffiths (2024), *Curiosity and the dynamics of optimal exploration* — review integrating uncertainty, information gain, and learning progress.

## What this says about the larger machine

DYN6 does not make Twensday a research AI.

It does add one operation that a research system would need:

```text
not merely:
    learn from what I was shown

but:
    estimate where interaction is still improving my model
    choose to spend more observations there
    notice when surprise is not becoming knowledge
```

The next useful jump should therefore not be another synthetic noisy-TV variant merely to make this curve prettier.

A stronger gate is to give the machine **competing explanatory models and actions that discriminate between them**. Then curiosity becomes experiment choice rather than region sampling:

```text
several hypotheses explain what I have seen
        ↓
which intervention / measurement would make them disagree most?
        ↓
choose it
        ↓
consequence changes belief
        ↓
choose the next experiment
```

That is much closer to the "research AI" possibility, and it has a brutal boring attacker: standard Bayesian / active experimental design.
