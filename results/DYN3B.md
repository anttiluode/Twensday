# DYN3B — retention and consolidation are different, but replay still loses

Development receipt, not confirmatory evidence.

DYN3 showed that a surprise-prioritized fast store can become enriched for episodes the slow model cannot explain. Uniform replay from that store badly damages reusable generalization.

DYN3B asks whether replay earns itself if it is gated by the slow model rather than applied uniformly.

## Same-information replay gates

No synthetic exception flag is available.

The slow model computes its current signed margin on a stored episode. Policies:

```text
no_replay             retain only; never replay
uniform               replay sampled episodes indiscriminately
congruent             replay only if cortex currently predicts the stored label
moderate_congruent    replay only positive but not-yet-certain margins
soft_congruent        stochastic preference for positive uncertain margins
conflicting           replay only episodes cortex currently disagrees with
```

Fast-memory retention remains the same surprise-prioritized policy in every condition.

## Independent GitHub run

Twelve seeds, 9000 items:

| replay policy | new-item accuracy | short exception recall | long regular recall | long exception recall | regular after H erase | replay acceptance |
|---|---:|---:|---:|---:|---:|---:|
| **no replay** | **0.8243** | 0.9951 | **0.9630** | **0.8690** | **0.9347** | 0.0000 |
| uniform | 0.6882 | 0.9774 | 0.7477 | 0.6339 | 0.8795 | 1.0000 |
| congruent | 0.7932 | 0.9950 | 0.8989 | 0.7942 | 0.9141 | 0.2147 |
| moderate congruent | 0.7954 | 0.9954 | 0.9018 | 0.7932 | 0.9132 | 0.2020 |
| soft congruent | 0.7723 | 0.9949 | 0.8663 | 0.7665 | 0.9023 | 0.2489 |
| conflicting | 0.6922 | 0.9532 | 0.7524 | 0.2476 | 0.7913 | 0.3691 |

## Result

The simple consolidation gates partially rescue uniform replay, but **none beats no replay**.

So replay has not earned a place in Twensday from this benchmark.

The reason is structural: the slow learner already receives every experience online and the reusable schema is stationary. Replay has no missing information to restore. It only changes the sampling distribution, and a residual-oriented episodic store is intentionally a biased distribution.

The failure still earns one architectural distinction:

```text
WRITE
  !=
RETAIN
  !=
CONSOLIDATE INTO SLOW KNOWLEDGE
```

But DYN3B does not provide a successful consolidation rule.

A future replay test needs a world in which old reusable information actually becomes unavailable or interfered with; otherwise replay is an answer to a problem the benchmark does not contain.
