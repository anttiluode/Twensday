# DYN3 — fast-write episodic memory and slow reusable knowledge separate cleanly

Development receipt, not confirmatory evidence. The world, memory policies, learning rates, delays, and attackers were designed while constructing the gate.

## Question

The hippocampal detour suggests that `write speed` and `memory lifetime` should not be collapsed into one time constant.

DYN3 asks a deliberately small complementary-learning question:

> **Can one continuously running machine preserve one-shot events immediately while a slower system learns reusable regularities, and can the slow system in turn decide which episodes still deserve scarce fast-memory capacity?**

This is not a hippocampus model. The fast store is intentionally embarrassingly digital and RAG-like.

## World

One continuous 9000-item stream.

Each new item has a 16-D semantic vector. A hidden linear schema gives the ordinary +/- label. Twelve percent of items are one-shot exceptions whose label is flipped relative to the schema.

For every new item the machine must predict *before* seeing the answer. That measures reusable generalization.

The exact same item is queried again after:

```text
40 events     short one-shot recall
800 events    long recall
```

Fast episodic capacity is only 256 items, so FIFO memory cannot carry an ordinary item across the 800-event delay.

At step 6500 the fast store is erased in a dedicated lesion probe. Regular and exception recall are then measured separately.

## Components

### Slow cortex

A deliberately boring online linear predictor over semantic features.

### FIFO episodic memory

Exact-address one-shot store. New event in, oldest event out.

### Slow-model-guided surprise memory

Each episode receives priority from the slow cortex's current error on it. While the slow model changes, stored priorities are refreshed.

So an initially surprising but learnable regular item becomes cheaper to evict once cortex can explain it, while a persistent idiosyncratic exception remains expensive.

This is the first tiny `M -> H` feedback test:

```text
slow knowledge changes what fast memory still needs to retain
```

It is also obviously related to error-prioritized caches / replay buffers and should not be treated as novel.

## GitHub reproduction

GitHub Actions independently ran the committed NumPy experiment on Ubuntu 24.04 / Python 3.12.

Twelve seeds, 9000 items:

| method | new-item accuracy | short exception recall | long regular recall | long exception recall | regular after H erase | exception after H erase |
|---|---:|---:|---:|---:|---:|---:|
| slow cortex only | 0.8243 | 0.0978 | 0.9328 | 0.0754 | 0.9347 | 0.0745 |
| fast cortex only | 0.7425 | 0.3103 | 0.8195 | 0.1756 | 0.8195 | 0.1797 |
| same-compute repeat-current attacker | 0.7960 | 0.1875 | 0.8912 | 0.1087 | 0.8932 | 0.1113 |
| episodic store only | 0.5003 | 0.9975 | 0.5014 | 0.4851 | 0.5015 | 0.4930 |
| FIFO H + slow cortex, no replay | 0.8243 | 0.9953 | 0.9328 | 0.0754 | 0.9347 | 0.0745 |
| FIFO H + uniform replay | 0.8097 | 0.9956 | 0.9089 | 0.0918 | 0.9046 | 0.0967 |
| **surprise H + slow cortex, no replay** | **0.8243** | **0.9951** | **0.9630** | **0.8690** | **0.9347** | **0.0745** |
| surprise H + uniform replay | 0.6901 | 0.9776 | 0.7474 | 0.6388 | 0.8779 | 0.1328 |

## What survives

### 1. Fast write and slow learning solve different parts of the same stream

The slow learner generalizes well but cannot remember one-shot exceptions:

```text
new item                  0.8243
short exception recall    0.0978
```

The fast episodic store gives the inverse profile:

```text
new item                  0.5003
short exception recall    0.9975
```

Neither alone does both jobs.

### 2. Fast-write does not imply short-lived

The surprise-prioritized 256-slot store retains many exceptions across an 800-item delay:

```text
long exception recall     0.8690
```

So the useful axes are already at least:

```text
write speed
retention policy / lifetime
```

rather than one scalar `tau`.

### 3. Slow knowledge can change fast-memory allocation

FIFO has no useful exception trace after 800 items:

```text
0.0754
```

Slow-model-guided retention reaches:

```text
0.8690
```

The store does not know which items are synthetic exceptions. It only knows which events the slowly changing cortex continues to fail to explain.

That earns the architectural relation:

```text
fast memory teaches / supports slow processing
        AND
slow knowledge can decide what fast memory still needs to hold
```

The second arrow is the more interesting result here.

### 4. Erasing fast memory separates consolidated regularity from episode-specific residue

For the best no-replay variant, after the fast store is erased:

```text
regular recall      0.9347
exception recall    0.0745
```

The regular relation remains because it lives in the slow semantic model. The item-specific deviation disappears because it existed only as an episodic trace.

This is a synthetic architectural lesion, not evidence about biological systems, but it is exactly the distinction DYN3 was built to expose.

## What fails

### Naive replay is harmful

FIFO replay modestly hurts the slow learner.

More importantly, replaying uniformly from the surprise-prioritized store is disastrous:

```text
new-item accuracy
0.8243 -> 0.6901
```

Why? The memory policy is doing its job: it preferentially retains items the cortex cannot explain. In this world those are increasingly the idiosyncratic exceptions. Uniformly replaying them into cortex repeatedly teaches the slow system to fit episodic residue rather than the reusable schema.

So:

> **`worth remembering` is not the same predicate as `worth consolidating`.**

This is the strongest DYN3 lesson so far.

### Extra compute is not replay

Repeating updates on the current example with the same approximate update budget reaches only `0.7960` new-item accuracy and worsens slow regular knowledge relative to the conservative slow learner.

That does not rescue replay. It simply shows that aggressively fitting whatever is currently available is also a bad stability/plasticity policy in this noisy stream.

## What DYN3 does NOT establish

- Surprise-prioritized memory is not a novel algorithm; it is cache / prioritized-memory territory.
- The exact-address store is much closer to an external key-value memory than a biological hippocampus.
- We have not shown replay to be useful. The first replay implementation fails.
- We have not shown any advantage over well-designed rehearsal, prioritized replay, elastic/continual-learning methods, or explicit RAG systems.
- The slow learner is intentionally tiny and the world is synthetic.

## Next attack: separate retention from consolidation

The failure points to a cleaner architecture than the one we started with.

An episode can be:

```text
important to retain exactly
but actively harmful to absorb into the slow model
```

So DYN3B should give replay its own gate.

A first same-information rule is simple:

```text
retain episode if cortex still finds it surprising
consolidate/replay episode only if its relation is becoming compatible with cortex
```

No synthetic exception flag is allowed.

Required controls:

```text
uniform replay
replay only cortex-congruent episodes
replay only cortex-conflicting episodes
no replay
same-compute repeat-current updates
random reservoir rehearsal
```

If a consolidation gate restores slow generalization while keeping long one-shot recall, we earn another separation:

```text
WRITE != RETAIN != REPLAY/CONSOLIDATE
```

That would be a much better reason for multiple memory timescales than simply adding another leaky state.
