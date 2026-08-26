# DYN9 — can forgotten history support theories that do not exist yet?

DYN8 let every late-born explanation score itself against compact sufficient statistics of **all** old evidence. DYN9 removes that privilege.

The question is:

> **Can a tiny raw memory, selected before a hypothesis exists, preserve enough evidence for that future hypothesis to be evaluated later?**

This is a memory-selection / coreset problem, not a novelty claim. Reservoir sampling, coreset selection, replay buffers, leverage/influence methods, robust statistics and reducible-loss selection are mature attackers.

## Gate

During the historical phase:

```text
six wrong explanations exist
true explanation is absent
new theories are forbidden from being born
150-ish observations arrive
only K raw observations may survive
```

The existing wrong explanations may update their own belief because they exist at the time. Future explanations may not accumulate model-specific sufficient statistics.

After history is over, the grammar is opened and **all possible late-born structures are scored only from the retained raw observations**. The best six become the new live population, then exact DYN7 expected information gain gets a short clean follow-up phase.

Every retention method sees the identical historical stream and identical wrong live population.

Retention methods:

- `full_history`: diagnostic upper bound.
- `reservoir`: uniform reservoir sampling.
- `recent`: most recent K observations.
- `leverage`: grammar-aware but outcome-agnostic weighted reservoir based on primitive energy / instrument variance.
- `raw_residual`: keep largest instantaneous standardized errors under current explanations.
- `stubborn_diverse`: keep persistent standardized mismatch, with repeated storage at the same measurement location discounted.
- `split_*`: DYN9D attempted to divide one hard budget between representative reservoir memory and stubborn memory.

The late-theory evaluator is shared by all methods and clips extreme standardized losses so one giant outlier cannot single-handedly decide the future model ranking.

---

# DYN9A/B — first world

48 seeds, 12 grammar primitives, hidden order 3, 220 possible structures, 21 measurement locations, history length 126.

With a memory of eight observations in the clean world:

```text
method                 truth in top 6   truth top 1   pre-EIG MSE   post-EIG MSE
full history                1.000           1.000        0.0000        0.0000
reservoir                   1.000           0.896        0.0628        0.0000
recent                      1.000           0.979        0.0356        0.0000
leverage                    1.000           0.938        0.0096        0.0000
raw residual                0.979           0.667        0.2640        0.0114
stubborn + diverse          1.000           0.938        0.0260        0.0000
```

At 16 raw memories the ordinary baselines are already essentially perfect. At 32 all useful methods saturate.

So the easy world kills any strong claim that future-theory support requires a special residual memory. Ordinary representative memory is enough.

## Contamination attack

DYN9B corrupts 8% of historical observations with large extra noise.

At memory 8:

```text
method                 truth in top 6   truth top 1   pre-EIG MSE   post-EIG MSE   corruption in memory
reservoir                   1.000           0.792        0.1832        0.0000            0.073
recent                      1.000           0.938        0.1217        0.0000            0.081
leverage                    1.000           0.854        0.0732        0.0000            0.057
raw residual                0.938           0.500        0.5417        0.0466            0.198
stubborn + diverse          1.000           0.854        0.1533        0.0000            0.164
```

The raw residual rule is decisively killed.

An 8% corruption stream becomes almost 20% of its retained memory. Surprise is exactly what an outlier looks like.

> **Large unexplained error is not, by itself, evidence worth preserving.**

Persistent/diverse residual retention is better than raw residual, but it still over-enriches corruption relative to representative memory.

---

# DYN9C — hard future-theory gate

The easy world was too forgiving, so the hypothesis space was enlarged:

```text
16 grammar primitives
hidden order 4
C(16,4) = 1820 possible future theories
6 future live slots
150 historical observations
memory only 6 / 10 / 16 raw observations
```

## Clean history, memory = 6

```text
method                 truth in top 6   truth top 1   pre-EIG MSE   post-EIG MSE
full history                1.000           1.000        0.0000        0.0000
reservoir                   0.969           0.875        0.2694        0.0372
recent                      1.000           0.750        0.3916        0.0000
leverage                    1.000           0.844        0.2211        0.0000
raw residual                0.750           0.406        1.0154        0.5233
stubborn + diverse          1.000           0.906        0.1283        0.0000
```

This is the narrow positive result.

With only six observations retained from 150, persistent + diverse mismatch preserves the future true theory in all 32 seeds, puts it first in 90.6%, and gives the best pre-follow-up MSE among the generic finite memories.

Reservoir is close but not identical: 96.9% truth inclusion, 87.5% top-1, and one failure class survives even after the short EIG follow-up.

So under **extreme clean memory pressure + a misspecified present model**, stubborn evidence can be useful for future structure discovery.

At memory 10 and 16 the advantage mostly disappears: ordinary recent/leverage/reservoir memories become strong enough that the special mechanism is no longer needed.

## Contaminated history, memory = 6

Now 10% of history is corrupted with large outliers:

```text
method                 truth in top 6   truth top 1   pre-EIG MSE   post-EIG MSE   corruption in memory
reservoir                   0.938           0.750        0.4044        0.1242            0.104
recent                      0.906           0.594        0.7499        0.2528            0.089
leverage                    0.906           0.625        0.4390        0.1415            0.089
raw residual                0.625           0.281        1.2735        0.7064            0.234
stubborn + diverse          0.906           0.719        0.5549        0.2030            0.172
```

The ranking flips.

Reservoir is now safer than stubborn memory. The persistent residual store still over-selects corrupted evidence.

At memory 10, stubborn memory recovers useful predictive quality (`0.0566` pre-EIG MSE) but no longer has a structure-recall advantage over representative methods. At memory 16 all sensible methods largely saturate.

---

# DYN9D — split memory does not rescue the story

A tempting response is:

```text
half representative memory
+
half stubborn anomaly memory
```

DYN9D tested exactly that under the hard six-slot budget, routing observations into a representative reservoir lane or a stubborn lane according to several persistent-residual thresholds.

It did not win.

48-seed clean result:

```text
method                 truth in top 6   truth top 1   pre-EIG MSE   post-EIG MSE
reservoir                   0.979           0.854        0.3012        0.0248
leverage                    1.000           0.875        0.1944        0.0000
stubborn + diverse          1.000           0.875        0.1646        0.0000
split threshold .75         0.917           0.750        0.3649        0.0841
split threshold 1.5         0.958           0.646        0.4993        0.0925
split threshold 3.0         0.979           0.750        0.3157        0.0155
```

Contaminated result:

```text
method                 truth in top 6   truth top 1   pre-EIG MSE   post-EIG MSE
reservoir                   0.917           0.708        0.4745        0.1877
leverage                    0.917           0.708        0.4157        0.1644
stubborn + diverse          0.833           0.646        0.6587        0.3030
split threshold .75         0.792           0.458        0.9144        0.3389
split threshold 1.5         0.729           0.500        0.8612        0.4631
split threshold 3.0         0.771           0.479        0.8064        0.4324
```

So a fixed half-and-half memory split is rejected too. Scarce capacity spent on the wrong lane is expensive.

---

# What survives

DYN9 does **not** establish a special hippocampal-style memory system.

It gives a narrower memory-selection result:

1. **Raw surprise retention is bad.** It preferentially stores corruption and irreducible difficulty.
2. **Representative memory is a very strong boring default.** Reservoir/recent/leverage methods solve most of these worlds.
3. **Persistent + diverse model mismatch can add value under an extreme clean memory bottleneck when the current model class is wrong.**
4. **That advantage is fragile to corrupted observations.** The machine needs a way to distinguish `model wrong` from `measurement/source wrong`.
5. **A static 50/50 representative/anomaly split is not the answer.**

The important temporal point still survives:

> **Evidence can be useful to an explanation that did not exist when the evidence arrived.**

But deciding which such evidence to keep is not equivalent to keeping the largest residual.

# Next wall — source reliability versus model failure

DYN9 exposes the next ambiguity directly:

```text
persistent unexplained evidence
        ↓
which is it?

A. my explanatory model is inadequate
B. the measurement/source is unreliable
C. the world changed regimes
```

DYN9 gives the learner the instrument sigma and then injects contamination behind its back. A useful research machine should not receive a truth label saying which channel is trustworthy.

The next gate should therefore make **source reliability itself latent and learnable**.

Required attackers should include robust regression / Student-t likelihoods, explicit contamination-mixture models, RANSAC-style ideas where appropriate, Bayesian source-reliability models, representative coresets, and the current residual memories.

Kill condition:

> If ordinary robust statistics can distinguish bad evidence from missing structure and make special H unnecessary, use ordinary robust statistics.

The architectural question becomes:

> **Can a finite evolving learner decide whether a stubborn residual is evidence that its theory should change, or evidence that its observation channel should be trusted less?**
