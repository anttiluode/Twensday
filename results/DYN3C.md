# DYN3C — slow knowledge can allocate scarce fast memory to residuals

Development receipt, not confirmatory evidence.

## Question

DYN3's strongest result was the reverse arrow:

```text
slow model -> fast memory allocation
```

A finite episodic store can preferentially retain experiences the slow model still cannot regenerate.

DYN3C attacks that result across memory capacities and against boring cache policies.

## World

Same 9000-item stream as DYN3:

```text
16-D semantic input
stable learnable schema
12% item-specific label exceptions
800-event long-recall probe
slow cortex sees every item once online
```

All non-oracle methods receive the same information.

## Memory policies

### FIFO

Store every event and evict oldest.

### Random eviction

Store every event and replace a random slot when full.

### Cortex-guided surprise

Refresh each stored event's priority from the slow model's current prediction error. Events the slow model learns to regenerate become cheap to evict. Persistently unexplained events remain expensive.

### Exception-only oracle

Unfair control told the synthetic exception flag. It stores only exception events with FIFO replacement among them.

This is an oracle for *which class of event deserves episodic storage*, not an optimal retention policy at every capacity.

## Independent GitHub run

Twelve seeds, 9000 items.

### Capacity 32

| policy | all long recall | regular | exception | final exception fraction in memory |
|---|---:|---:|---:|---:|
| FIFO | 0.8297 | 0.9330 | 0.0747 | 0.1198 |
| random eviction | 0.8297 | 0.9330 | 0.0747 | 0.1016 |
| **cortex-guided surprise** | **0.8366** | 0.9330 | **0.1317** | **1.0000** |
| exception-only oracle | 0.8297 | 0.9330 | 0.0747 | 1.0000 |

### Capacity 64

| policy | all long recall | regular | exception | final exception fraction in memory |
|---|---:|---:|---:|---:|
| FIFO | 0.8297 | 0.9330 | 0.0747 | 0.1237 |
| random eviction | 0.8297 | 0.9330 | 0.0747 | 0.1393 |
| **cortex-guided surprise** | **0.8738** | 0.9330 | **0.4415** | **0.9948** |
| exception-only oracle | 0.8297 | 0.9330 | 0.0747 | 1.0000 |

### Capacity 128

| policy | all long recall | regular | exception | final exception fraction in memory |
|---|---:|---:|---:|---:|
| FIFO | 0.8297 | 0.9330 | 0.0747 | 0.1237 |
| random eviction | 0.8300 | 0.9332 | 0.0760 | 0.1198 |
| **cortex-guided surprise** | **0.9106** | 0.9341 | **0.7389** | 0.9316 |
| exception-only oracle | **0.9411** | 0.9330 | **1.0000** | 1.0000 |

### Capacity 256

| policy | all long recall | regular | exception | final exception fraction in memory |
|---|---:|---:|---:|---:|
| FIFO | 0.8297 | 0.9330 | 0.0747 | 0.1195 |
| random eviction | 0.8365 | 0.9358 | 0.1110 | 0.1195 |
| **cortex-guided surprise** | **0.9594** | **0.9607** | **0.9498** | 0.7689 |
| exception-only oracle | 0.9411 | 0.9330 | **1.0000** | 1.0000 |

## What survives

### 1. Model-aware memory allocation is robust across capacity

At the 800-event delay FIFO is effectively useless for exceptions once capacity is far below the delay.

The slow-model-guided cache progressively recovers them:

```text
capacity 32     0.1317
capacity 64     0.4415
capacity 128    0.7389
capacity 256    0.9498
```

### 2. The slow model becomes a compression oracle for episodic storage

The allocator is never told which examples are synthetic exceptions, yet the final memory becomes strongly enriched for them:

```text
capacity 32      100.0% exceptions
capacity 64       99.5%
capacity 128      93.2%
capacity 256      76.9%
```

At larger capacity it also keeps difficult ordinary examples, which is why capacity 256 beats the exception-only oracle on total and regular recall while remaining near the oracle on exception recall.

### 3. Fast write and retention lifetime are separate axes

Every event can be considered immediately, but its persistence is determined later by whether slower knowledge learns to regenerate it.

That is the important timescale distinction:

```text
fast write
+
slowly revised retention priority
```

not merely `a larger decay constant`.

## What this is algorithmically

This is **residual / error-prioritized cache territory**. It is not a novelty claim and not evidence for a literal hippocampal implementation.

A practical description is:

> **Use slow parametric knowledge for what is compressible; spend scarce fast external memory on the residual the slow model cannot regenerate.**

That is already a useful engineering principle for a hybrid parametric + episodic system.

## What remains missing from the hippocampal-loop story

The successful arrow so far is:

```text
slow model -> fast memory retention
```

The reverse consolidation arrow did not survive DYN3B:

```text
fast memory -X-> replay -> slow model
```

at least not in the stationary world where cortex sees every event online.

A stronger benchmark must contain either forgetting/interference or periods in which slow structure cannot learn directly before replay can earn itself.
