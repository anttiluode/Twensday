# DYN4 — fast state can be written long, then reinstated when context returns

Development receipt, not confirmatory evidence.

## Question

DYN3C showed that a slow model can allocate scarce fast episodic storage toward observations it cannot regenerate.

DYN4 moves the memory target one level upward:

> **Can fast memory store a previously learned *dynamical state*, let that state disappear from active computation, and reinstate it rapidly when the old context returns?**

This is a synthetic three-timescale abstraction, not a hippocampal model.

## World

```text
48 recurring contexts
25 events per context visit
~1200 events before the same context returns
12000 total events
```

All contexts share one slow global linear rule, but each has a stable random local offset.

The global rule is learnable across contexts. The context offset is random and cannot be inferred algebraically from the context identity.

## Three clocks

### `M` — slow global model

Learns the reusable global semantic rule.

### `q` — fast active context state

A scalar correction adapts from prediction error during the current 25-event context visit.

Without retrieval it starts from zero every time a context appears.

### `H` — fast-write / long-retained context-state store

At the end of a visit, the learned `q` can be written to a stable context address. On a later recurrence that old fast state can be restored immediately.

Finite H capacity is only 12 states for 48 recurring contexts.

## Finite-memory policies

```text
none          no persistent context state
FIFO          last 12 contexts
random        random replacement
residual      prioritize contexts where q improved loss over M alone
oracle        unfair priority from true synthetic |context offset|
full_table    unbounded explicit context table; boring ceiling
```

A scrambled-address lesion retains the same stored scalar states but destroys the stable context-to-slot relation.

## Independent GitHub result

Twelve seeds, 12000 events.

| policy | first event on recurrence | first 5 | late in visit | first event, large-residual contexts | cache hit rate |
|---|---:|---:|---:|---:|---:|
| no H | 0.8065 | 0.8371 | 0.9425 | 0.7209 | 0.0000 |
| FIFO | 0.8065 | 0.8371 | 0.9425 | 0.7209 | 0.0000 |
| random eviction | 0.8090 | 0.8389 | 0.9424 | 0.7266 | 0.0191 |
| **model-residual H** | **0.8607** | **0.8791** | 0.9443 | **0.8293** | 0.1979 |
| true-offset oracle priority | 0.8360 | 0.8610 | 0.9437 | 0.7794 | 0.1094 |
| full explicit context table | **0.9471** | **0.9478** | **0.9454** | **0.9536** | 1.0000 |
| scrambled residual addresses | 0.8034 | 0.8345 | 0.9416 | 0.7170 | 0.1715 |

## What survives

### 1. Long memory can store a fast state, not merely an old observation

The stored item is the context correction `q` that was learned during an earlier visit.

On recurrence it is reintroduced into the active predictor before new evidence from that context has arrived.

So the memory grammar is:

```text
experience
   -> fast state q is learned
   -> q leaves active dynamics
   -> H retains q much longer
   -> context returns
   -> old q is reinstated
   -> fast dynamics continue from a better initial condition
```

That is a stronger multi-timescale result than exact item lookup.

### 2. Scarce H gives measurable rapid reacquisition

With only 12 slots for 48 contexts:

```text
first event on recurrence
no H              0.8065
residual H         0.8607
```

For contexts where slow global knowledge leaves a large residual:

```text
no H              0.7209
residual H         0.8293
```

By the end of each 25-event visit all policies converge near `0.94`, because the fast state can be relearned online. H buys **time-to-reacquire**, not a higher asymptotic solution.

### 3. Stable addressability remains essential

Scrambling which stored state belongs to which context collapses first-event performance:

```text
0.8607 -> 0.8034
```

The cache still contains roughly the same number of states and reports a similar raw hit frequency. What is lost is causal identity: the right old state cannot be reinstated into the right returning context.

This repeats the stable-address finding from earlier recurrent gates in a different memory role.

### 4. The unbounded boring solution still wins badly

A full explicit context table gets:

```text
0.9471 first-event recurrence accuracy
```

So DYN4 is not evidence that this is a better software memory system. It is a finite-memory allocation experiment.

## DYN4B — attack the residual priority

Maybe slow-model comparison was unnecessary. Perhaps large fast states are simply the contexts worth keeping.

A second independent GitHub run compares same-capacity priority rules:

| priority | first recurrence | first 5 | large-residual first | hit rate |
|---|---:|---:|---:|---:|
| **slow-model residual improvement** | **0.8607** | **0.8791** | **0.8293** | 0.1979 |
| slow-model baseline error | 0.8544 | 0.8752 | 0.8181 | 0.1863 |
| mean fast-state magnitude | 0.8418 | 0.8652 | 0.7925 | 0.1283 |
| final fast-state magnitude | 0.8378 | 0.8600 | 0.7838 | 0.1127 |
| late fast-state error | 0.8306 | 0.8545 | 0.7546 | 0.1991 |

The precise residual-improvement rule is not sacred. Plain slow-model error is close.

But purely local `|q|` heuristics are materially weaker. Some measure of **what the slow model still fails to explain** is useful for allocating long-lived fast-state memory.

## Current surviving principle

The hippocampal detour has therefore earned a more concrete computational sentence:

> **A fast state need not remain continuously active to remain rapidly reusable. It can be written into a longer-lived finite store, while slower knowledge decides which dormant states are worth preserving; when a stable context returns, the old state can be reinstated to shorten reacquisition.**

Algorithmically this lives near adapter caches, fast weights, contextual memory, residual caches, and explicit key-value memories. No novelty claim follows from the architecture alone.

## Next wall

DYN4 cheats with an exact stable context ID.

The next attack should remove that address and retrieve from a partial/noisy cue:

```text
no exact context index
content-addressed key
similar contexts can interfere
old fast state must be pattern-completed from partial evidence
```

Mandatory attackers:

```text
nearest-neighbor/vector database
explicit associative memory
full context classifier + table
small recurrent model
random-key cache
```

If content-addressed reinstatement survives, the fast/long/slow hierarchy becomes more than a dictionary exercise. If nearest-neighbor lookup solves it perfectly, record that and move on.
