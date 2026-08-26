# DYN8 — can the hypothesis set itself evolve?

DYN7 made research-like experiment choice easy in one decisive way: the correct explanation was already somewhere in a fixed finite catalog. DYN8 removes that cheat.

The question is:

> **Can a finite continuously evolving population of explanations retain useful experiment-selection power when the correct explanation is absent from the initial population?**

This is not a novelty claim. Sequential Monte Carlo, online structure learning, beam/model search, Bayesian experimental design and active learning already cover large parts of this territory. DYN8 asks whether Twensday's particular separation

```text
H = stubborn unexplained evidence
M = finite active explanation population
```

adds anything useful under a hard slot budget.

## Important concession: a grammar is still supplied

DYN8 does **not** create scientific concepts from nothing.

The learner receives a small library of reusable basis primitives. A hidden law is a sparse combination of three primitives. What is *not* supplied to the finite learner is the catalog of possible three-term combinations.

So:

```text
grammar supplied
hypothesis catalog not supplied
correct combination absent initially
```

This is structure discovery inside a known grammar, not open-ended theory invention.

## World

There are 21 possible measurement locations. Five are deliberately poor instruments:

```text
ordinary sigma = 0.22
bad sigma      = 0.90
```

Each policy gets the same nth observation from each action-specific random stream.

Every finite-population method receives the same compact per-action sufficient statistics. This matters: `H` is **not** allowed to win merely by possessing old evidence other methods cannot access.

`H` has only eight slots and is used as a *proposal mechanism*: it preferentially keeps observations with persistent excess standardized residual under the current explanation population.

The finite active explanation population has only six slots.

## Structure birth mechanisms

### `residual`

Twensday proposal.

Take the currently best explanation and ask which one-feature structural mutation most reduces loss on stubborn `H`. Insert that child and kill the worst current explanation.

### `beam`

Boring strong attacker.

Exhaustively score every one-feature mutation of every current explanation against **all accumulated sufficient statistics** and insert the globally best improving child.

This uses more proposal computation than `H` and no stubborn-memory heuristic.

### `smc`

Equal-slot resample-move Sequential-Monte-Carlo-style attacker.

Resample explanations by evidence weight, randomly mutate their sparse structures, and Metropolis-accept/reject using accumulated evidence.

### `fixed`

No model birth. The correct explanation is deliberately absent initially.

## Experiment choice

DYN8A uses a cheap Gaussian mutual-information approximation:

```text
0.5 log(1 + epistemic prediction variance / observation variance)
```

It discounts known instrument noise and allows a 364-model full catalog to remain computationally cheap.

`hybrid` adds a small coverage pressure because a missing hypothesis creates a dangerous circularity:

```text
choose observations informative under current M
        ↓
current M is wrong
        ↓
never look where evidence for missing structure lives
```

DYN8B then repeats a smaller world using the **exact DYN7 expected-information-gain calculation** for every method, including the complete catalog.

---

# DYN8A — hard combinatorial world

32 seeds.

```text
grammar primitives     14
three-term catalog     C(14,3) = 364 explanations
finite live slots      6
stubborn H slots       8
```

GitHub Actions result:

```text
method                   structure hit   MSE hit   final truth present   final MSE
residual_info                 15.59        15.59          .906             .13148
residual_hybrid               12.78        12.78          .969             .02566
residual_random               14.28        14.28          .969             .04601
beam_info                      9.62         9.66         1.000             .00000
smc_info                      15.94        15.94          .938             .05577
smc_hybrid                    14.12        14.12         1.000             .00000
fixed_info                    41.00        41.00          .000            1.89500
catalog_info                   4.50         3.75         1.000             .00000
catalog_random                 7.50         5.28         1.000             .00000
```

`41` is the censored value for failure to hit inside 40 observations.

## Kill 1 — fixed EIG cannot discover a missing explanation

This is obvious but important to measure.

The six-slot fixed population has information-aware experiment choice but the true explanation is absent forever:

```text
fixed_info
truth present       0 / 32
final MSE           1.895
```

Experiment design cannot rescue an inadequate hypothesis set if nothing can change that set.

## Kill 2 — information gain over the wrong M can become myopic

The surprising result is:

```text
residual_info       15.59
residual_random     14.28
```

Pure information-aware sampling is **worse than random** for the residual-birth machine in this harder world.

Why?

Because the acquisition function asks:

```text
which experiment discriminates the explanations currently alive?
```

while the structure learner needs evidence for an explanation that is **not alive yet**.

The current population can therefore become very efficient at asking questions inside the wrong model class.

A tiny coverage reserve repairs much of this:

```text
residual_hybrid     12.78
residual_random     14.28
residual_info       15.59
```

This is not a new discovery in active learning. It is directly related to known active-learning / adaptive-design bias under model misspecification. Sloman et al. show that Bayesian adaptive experimental design can be harmed by model misspecification and sampling bias:

https://arxiv.org/abs/2205.13698

For Twensday the architectural lesson is concrete:

> **EIG over M is only safe to exploit aggressively when M is believed to contain an adequate explanation. An open model population needs some budget for evidence from outside its current explanatory focus.**

## Does stubborn H earn anything?

A little, but not dominance.

Against the equal-slot SMC-style attacker in the same hard world:

```text
residual_hybrid      12.78
smc_hybrid           14.12
```

So the eight stubborn residuals provide a useful directed proposal signal here.

But exhaustive beam search is better:

```text
beam_info              9.62
residual_hybrid        12.78
```

And the full 364-model catalog is vastly better:

```text
catalog_info MSE hit    3.75
residual_hybrid        12.78
```

Therefore DYN8A does **not** establish stubborn memory as a superior structure-learning method. It behaves like a cheap proposal heuristic that can outperform blind particle mutation but loses to stronger search when stronger search is affordable.

---

# DYN8B — exact expected information gain

The hard world used a Gaussian information approximation. DYN8B removes that ambiguity by shrinking the grammar enough to run the exact DYN7 EIG calculation.

24 seeds.

```text
grammar primitives     10
three-term catalog     C(10,3) = 120 explanations
finite live slots      6
stubborn H slots       8
```

Result:

```text
method                   structure hit   MSE hit   final truth present   final MSE
residual_exact                 7.25         7.25         1.000             .00000
residual_random                8.12         8.12         1.000             .00000
beam_exact                     5.75         5.75         1.000             .00000
smc_exact                      8.29         8.38          .958             .00314
fixed_exact                   31.00        31.00          .000            1.32634
catalog_exact                  4.12         3.04         1.000             .00000
catalog_random                10.12         5.25         1.000             .00000
```

`31` is the censored value for failure inside 30 observations.

## The narrow positive result

The correct explanation begins absent from all six finite slots.

Residual-guided birth plus exact EIG nevertheless ends with the exact hidden structure in **all 24 seeds**:

```text
residual_exact structure hit   7.25 observations
residual_random                8.12
smc_exact                      8.29
```

So in this constructed world:

- exact experiment choice helps the evolving residual population by about 11% versus random observation choice;
- stubborn residual proposals are about 13% faster than the equal-slot resample-move SMC attacker on structure-hit time;
- six live explanations are enough to search a 120-hypothesis combinatorial space reliably when model birth is allowed.

That is enough to say the combination is computationally coherent.

It is **not** enough to say it retains most of full Bayesian-design efficiency.

## The main negative result

The complete catalog still crushes the finite evolving population:

```text
predictive MSE hit
catalog exact          3.04 observations
residual exact         7.25
beam exact             5.75
smc exact              8.38
```

The six-slot residual machine needs about 2.4x as many observations as the full 120-hypothesis catalog.

And the strongest finite attacker is not H. It is ordinary exhaustive local beam search:

```text
beam exact             5.75
residual exact         7.25
```

So this stronger claim is rejected:

> stubborn H + finite evolving M preserves most of exact Bayesian experiment-design power

Not in DYN8.

---

# What actually survives

DYN8 gives a more useful decomposition:

```text
FIXED MODEL POPULATION
information gain is powerful
but cannot invent an absent explanation

FINITE EVOLVING POPULATION
the missing explanation can be born

STUBBORN RESIDUAL H
can bias births toward useful structural changes
beats blind equal-slot particle mutation modestly here
but loses to exhaustive evidence-based beam search

PURE EIG WITH A MISSPECIFIED LIVE POPULATION
can become myopic

EIG + SMALL OUT-OF-MODEL EXPLORATION RESERVE
can be safer while the hypothesis set itself is still changing
```

That last distinction is the most Twensday-shaped result of the gate.

The research loop is no longer simply:

```text
M -> choose experiment -> update M
```

It needs another path:

```text
                    current M
                    /       \
                   /         \
       discriminate M       admit M may be wrong
              |                    |
             EIG              exploratory traffic
              |                    |
              +-------- world -----+
                         |
                     consequence
                         |
                  stubborn residual H
                         |
                  propose new structure
                         |
                    expanded M
                         ↺
```

A research machine needs both exploitation of its current explanatory space **and a mechanism by which evidence can force that explanatory space to change**.

## Literature boundary

This territory is heavily occupied.

- Sloman et al., *Characterizing the robustness of Bayesian adaptive experimental designs to active learning bias* — adaptive design under model misspecification can amplify bias: https://arxiv.org/abs/2205.13698
- Saad et al. (ICML 2023), *Sequential Monte Carlo Learning for Time Series Structure Discovery* — SMC over symbolic model structures can perform online structure discovery: https://proceedings.mlr.press/v202/saad23a.html
- Iollo et al. (ICML 2024), *PASOA — Particle Based Bayesian Optimal Adaptive Design* — particle inference and expected-information-gain experimental design already have mature combinations: https://proceedings.mlr.press/v235/iollo24a.html

So DYN8 is not a claim that particle-based model discovery plus active design is new.

The narrower Twensday question is whether its particular economy of finite live models plus selective stubborn residuals becomes useful when exhaustive history/model search is too costly.

## Next wall

DYN8 still has an enormous cheat:

```text
the correct theory is exactly three items from a supplied grammar
feature amplitudes are fixed
instrument noise is known
the world is stationary
one local feature swap can reach the answer
```

The next gate should attack the place where a finite dynamic research machine might actually differ from ordinary beam search:

> **Can stubborn residual memory pay for itself when proposing/evaluating a new explanation against all historical evidence is expensive or impossible?**

That means removing the compact sufficient-statistics cheat.

A candidate new explanation should be born late and face a real historical problem:

```text
I did not exist when the old evidence arrived.
I cannot replay everything.
Which old evidence was important enough to keep so that I can be evaluated now?
```

Now `H` has a non-decorative potential job.

If a small residual memory lets late-born explanations be evaluated nearly as well as a large replay buffer / full-history beam search, that would connect the DYN3C forgetting result directly to DYN8 model birth.

If not, keep beam/SMC and drop the special memory story.
