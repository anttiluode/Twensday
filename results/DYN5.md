# DYN5 — partial cue -> dormant state reinstatement

DYN5 removes the exact context identity that DYN4 gave to the dormant-state store.

## Question

Can a finite longer-lived memory `H` recover a previously learned fast state `q` from a partial/noisy cue, with ongoing consequence helping to confirm or reject the provisional retrieval?

This is a mechanism test, not a hippocampal model and not a novelty claim.

## World

- 48 recurring hidden contexts.
- 25 events per visit; a context returns after roughly 1,200 events.
- Contexts are grouped into 12 cue families of four.
- Members of a family share most sensory cue structure but have different hidden offsets.
- Only the first five events expose partial/noisy cue fragments.
- The algorithm never receives the hidden context ID.
- A slow linear model learns the global rule.
- Fast state `q` adapts within a visit.
- Finite `H` can hold 24 content-addressed dormant states.
- Slow-model residual improvement is used as the default finite-memory replacement priority.

The intended DYN5 policy treats cue similarity only as a prior. On event 1 it reinstates weakly. After the actual outcome is observed, stored candidate states are scored by how well they would have predicted that outcome; the resulting candidate distribution can influence event 2 onward.

## GitHub Actions development result

12 seeds, 12,000 events, 24 memory slots:

```text
policy                 first1   first5   pos2-5   hard pos2-5   late    exact-key debug
none                    .7515    .7772    .7836       .6922      .9016      .0000
eager nearest           .6694    .6890    .6939       .6644      .8099      .2264
cue-only soft            .7506    .7362    .7326       .6658      .8415      .2235
confirm FIFO             .7454    .8305    .8518       .8573      .9234      .0000
confirm random           .7494    .8380    .8601       .8655      .9272      .1041
confirm residual         .7525    .8368    .8578       .8955      .9095      .2553
scrambled confirm        .7508    .8395    .8617       .8935      .9171      .1385
unbounded confirm        .7531    .8463    .8696       .8945      .9234      .4491
exact-ID table           .9572    .9588    .9592       .9593      .9633      .0000
```

`exact-key debug` uses hidden labels only after the fact to ask whether the highest-weight stored item happened to originate from the same world context. Those labels are not available to the algorithm.

## What initially looked interesting

Eager vector retrieval is actively harmful. A noisy partial cue often identifies only a family, and immediately loading the nearest old state makes performance worse than no memory.

Outcome-confirmed candidate states are much better:

```text
no H, events 2-5             0.7836
confirm residual, events 2-5 0.8578
```

On the largest-offset contexts:

```text
no H                         0.6922
confirm residual             0.8955
```

So the useful operation is not `cue -> load memory`. It is closer to:

```text
partial cue -> candidate prior
current consequence -> candidate reweighting
reweighted state -> influence next event
```

## The lesion that prevents a pattern-completion claim

Scrambling dormant scalar states away from their content keys does **not** collapse the result:

```text
confirm residual       first5 = 0.8368
scrambled confirm      first5 = 0.8395
```

That means stable key->state association is not essential in this task. Consequence can select a useful scalar state from the available pool even when its original cue address has been corrupted.

Therefore DYN5 does **not** earn:

> partial sensory cue pattern-completes a specific dormant episodic state.

The exact-ID upper bound shows that correct context recognition would be valuable, but the present content-addressed mechanism has not demonstrated it.

## Next required attackers

DYN5B asks whether DYN5 is merely rapid mode estimation:

1. remove cues entirely but retain a finite bag of previously useful scalar states;
2. remove episodic memory entirely and use a fixed small grid of candidate states;
3. let ordinary fast `q` simply adapt more aggressively.

If those match or beat DYN5, associative-memory language is dropped.
