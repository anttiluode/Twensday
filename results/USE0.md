# USE0 — changing sensor fusion: useful neighborhood, wrong winner

Development receipt, not confirmatory evidence. Parameters were explored before the committed run.

## Question

Can the Twensday positive-only conserved allocation do a mundane useful job?

Use the structural mass directly as an online convex mixture over candidate sensor channels:

```text
sensor outputs h_i(t)
structural mass m_i >= reserve
sum_i m_i = 1

prediction y_hat(t) = sum_i m_i h_i(t)
```

The trusted sensor changes repeatedly and sensors that were useful earlier later become useful again. This is deliberately close to the stability/plasticity situation that motivated the reserve in the parent repo.

## World

Twenty deterministic seeds. Each run has:

```text
4800 time steps
8 sensor channels
400-step regimes
4 sensors take turns being the accurate sensor, then recur later
```

The hidden scalar signal is an AR(1) process. In each regime one sensor measures it with noise sigma=0.08. The other channels have wrong gains, substantially larger noise, and a slow bias term.

The learner receives the true scalar target after each prediction so this is online supervised sensor fusion, not blind source separation.

## Learners

### Twensday growth

The update is the Atlas-0 / Gate-15 linear local consequence rule applied to sensor features:

```text
error      = target - prediction
impulse_i  = max(m_i * h_i * error, 0)
m           <- normalize_with_reserve(m + eta * impulse)
```

The committed development parameters are:

```text
reserve = 0.002
growth rate = 12
no score smoothing
```

These were chosen after exploratory sweeps and therefore are not preregistered.

### Signed simplex SGD

A boring signed-gradient mixture with a reserve-constrained simplex projection.

### Fixed Share

Exponentially weighted experts plus a small share term that continually returns probability mass to all experts. This is a mature algorithm specifically designed to track a changing best expert.

### Uniform and oracle

Uniform averages all sensors. Oracle always uses the currently correct sensor and is not implementable; it is the lower reference.

## Results

Mean +/- standard deviation across 20 seeds:

| learner | overall MSE | first 50 steps after switch | first 200 steps after switch | mass on new best sensor, first 200 |
|---|---:|---:|---:|---:|
| oracle | `0.00636 +/- 0.00013` | — | — | `1.0` |
| fixed share | **`0.00903 +/- 0.00071`** | **`0.02871 +/- 0.00603`** | **`0.01203 +/- 0.00152`** | **`0.9293 +/- 0.0014`** |
| Twensday | `0.01956 +/- 0.00133` | `0.09394 +/- 0.01174` | `0.03044 +/- 0.00317` | `0.7963 +/- 0.0167` |
| signed simplex SGD | `0.02165 +/- 0.00066` | `0.08889 +/- 0.00533` | `0.03185 +/- 0.00148` | `0.7359 +/- 0.0096` |
| uniform | `0.19449 +/- 0.00423` | — | — | `0.125` |

Twensday therefore does real adaptive work. It reduces error by about an order of magnitude relative to uniform fusion and, in this development world, slightly beats the tuned signed-simplex baseline on overall error.

But Fixed Share is the decisive attacker:

```text
Twensday overall MSE / Fixed Share MSE      ~= 2.17
Twensday first-50 switch MSE / Fixed Share ~= 3.27
```

So the current growth law is not a competitive answer to ordinary full-information changing-expert fusion.

## The more useful discovery: the growth law has a known algorithmic relative

Ignore the score smoothing and reserve for a moment. The core Twensday update has the shape

```text
m_i' proportional to m_i + eta * m_i * positive_fitness_i
```

or, for a small step,

```text
m_i' approximately proportional to m_i * (1 + eta * positive_fitness_i)
```

followed by conservation / renormalization.

That is replicator / multiplicative-weights-like dynamics: existing mass is multiplied by local fitness and the common budget makes relatively worse candidates shrink without issuing each one an explicit negative structural command.

The protected reserve plays the same broad role as mutation, exploration, or the weight-sharing term in tracking-expert algorithms: do not let an alternative vanish so completely that a changing world can never reveal its value again.

This does **not** make Twensday identical to Hedge, Fixed Share, or a standard replicator-mutator equation. Its fitness signal is different, it clips negative local evidence rather than exponentiating a loss, and its reserve is implemented as a floor. But this is now the right mathematical neighborhood in which to attack it.

## Practical interpretation

For ordinary software sensor fusion, use the mature algorithm. Fixed Share wins and comes with theory.

Twensday becomes interesting only if the implementation constraint is itself important:

```text
candidate i knows its own local activity
all candidates receive one scalar consequence/error signal
local structure can be reinforced/grown
precise signed local retraction is unavailable or expensive
one conserved physical/resource budget supplies indirect shrinkage
```

That is much closer to the original morphology / material-allocation idea than unconstrained software regression is.

So USE0 changes the practical question from:

> can Twensday be a better online mixture algorithm?

(no evidence for that)

to:

> can this one-sided, locally credited, resource-conserving update solve useful adaptive problems **when the substrate really has one-sided/local update constraints**?

That is a sharper and more defensible niche.

## Next useful attacks

1. **Same-information attacker.** Give every candidate only `local activity + one broadcast scalar consequence` and compare against the strongest local signed and multiplicative rules that use exactly that information.
2. **One-sided substrate attacker.** Explicitly prohibit direct negative updates and measure which conservation/share rule tracks changing operators fastest for a fixed exploratory budget.
3. **Filter-bank version.** Replace scalar sensors with fixed FIR/resonant candidate filters. Ask whether conserved allocation can track a changing physical transfer path or noise source.
4. **Real stream.** Only after the synthetic gate is understood, use a real cheap-sensor problem (audio/vibration/PC telemetry) with an externally measurable target.

The important result from USE0 is not a win. It is that Twensday has crossed from an unnamed mechanism into a well-developed practical field where its odd constraints can be stated and attacked precisely.

## Known relatives worth keeping beside this repo

- Sanjeev Arora, Elad Hazan, Satyen Kale. **The Multiplicative Weights Update Method: a Meta-Algorithm and Applications.** Theory of Computing 8 (2012), 121–164. DOI: https://doi.org/10.4086/toc.2012.v008a006
- Mark Herbster, Manfred K. Warmuth. **Tracking the Best Expert.** Machine Learning 32 (1998), 151–178. DOI: https://doi.org/10.1023/A:1007424614876
- Jerónimo Arenas-García et al. **Combinations of Adaptive Filters: Performance and convergence properties.** IEEE Signal Processing Magazine 33(1) (2016), 120–140. DOI: https://doi.org/10.1109/MSP.2015.2481746
