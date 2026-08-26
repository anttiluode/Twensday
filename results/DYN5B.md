# DYN5B — does associative memory earn itself?

DYN5 looked as if partial cues plus outcome could reinstate a useful dormant state. The scrambled-key lesion already suggested that interpretation was too generous.

DYN5B removes the allegedly hippocampal ingredients one at a time.

## Attackers

All methods face the same 48-context continuous world used by DYN5.

- `confirm_residual`: DYN5 finite cue-keyed memory, slow-residual replacement, cue prior + outcome confirmation.
- `stored_state_bag`: same finite capacity, but **no cue keys at all**. It only keeps a bag of previously useful scalar fast states; current outcomes select among them.
- `fixed_state_grid`: **no episodic memory at all**. Nine fixed candidate states from -1.2 to +1.2 are reweighted from current outcomes.
- `adapt_eta_*`: no dormant memory and no candidate bank; ordinary scalar `q` simply adapts at different rates.
- `exact_id_table`: diagnostic upper bound explicitly given the hidden context ID.

## GitHub Actions result

12 seeds, 12,000 events:

```text
method                    first1   first5   pos2-5   hard pos2-5   late
confirm_residual           .7525    .8368    .8578       .8955      .9095
exact_id_table             .9572    .9588    .9592       .9593      .9633
stored_state_bag           .7527    .8414    .8636       .9058      .9093
fixed_state_grid           .7529    .8513    .8759       .8680      .9370
adapt eta=.06              .7515    .7772    .7836       .6922      .9016
adapt eta=.15              .7523    .7995    .8113       .7339      .9456
adapt eta=.30              .7529    .8247    .8426       .7844      .9453
adapt eta=.60              .7541    .8474    .8708       .8556      .9214
```

## Kill

The cue-keyed episodic mechanism does not win.

A finite bag of old scalar states with **no sensory cues** slightly beats it over the first five return events:

```text
DYN5 confirm residual   0.8368
stored-state bag        0.8414
```

A fixed nine-state grid with **no episodic memory** does better again:

```text
fixed state grid        0.8513
```

And simply increasing the fast-state adaptation rate to `eta=0.60` reaches:

```text
0.8474
```

So DYN5's useful operation is best described as **rapid online mode/state inference from consequence**, not associative episodic reinstatement.

The exact-ID table remains much better (`0.9588` first-five accuracy), so there is genuine value available if a recurring context can be recognized correctly. This gate only says that the present partial-cue content memory did not solve that recognition problem.

## What survives

DYN3C/DYN4 still support a different result:

> slow-model residual can be a useful signal for deciding which finite dormant states deserve retention.

DYN5B does **not** kill that memory-allocation result. It kills the stronger claim that the current noisy cue mechanism retrieves a specific dormant state by content.

The remaining useful separation is therefore:

```text
M -> H retention/allocation      survived earlier attacks
partial cue -> specific H recall not yet earned
outcome -> rapid q inference     works, but boring attackers match/beat it
```

## Next direction

Do not tune DYN5 until it wins.

The more promising residual signal should instead be attacked in an **active** loop:

> Can the system use what its slow model fails to explain to decide where to spend sensing/learning effort, while avoiding irreducible-noise traps?

That is an intrinsic-motivation / active-learning neighborhood, so required attackers include random exploration, prediction-error curiosity, learning-progress curiosity, uncertainty/information-gain methods, and a deliberately stochastic `noisy TV` distractor.
