# MambaServe

A small C++/CUDA **benchmark testbed** for studying Mamba serving at the distributed-systems layer: scheduling, state migration, and load placement.

This is not a vLLM replacement. vLLM already serves Mamba and hybrid models in production. MambaServe exists to **isolate and measure** scheduling decisions that production runtimes do not explore — especially **dynamic peer migration** of in-flight sequences between GPU workers.

## What This Project Is

```text
MambaServe = minimal runtime + reproducible benchmarks + explicit policies
```

The runtime is intentionally small. The deliverable is **data and tradeoffs**, not feature parity with vLLM.

| vLLM | MambaServe |
| --- | --- |
| Serve many models in production | Serve one pure Mamba model correctly |
| Admission-time routing to replicas | Mid-flight migration between peer workers |
| P/D disagg (fixed prefill → decode roles) | Opportunistic load balancing within a peer pool |
| Implicit scheduler heuristics | Explicit, swappable placement policies |
| Large Python + PyTorch stack | Small C++/CUDA control plane |

Use vLLM as a **correctness and feature ceiling reference**. Use MambaServe to answer questions vLLM is not designed to optimize for.

## Central Question

Production serving load-balances **requests at admission**. Once a request is bound to a worker, its state stays there until completion. vLLM's NIXL connectors move state along **fixed topologies** (prefill → decode, offload → GPU), not between arbitrary peer workers for utilization.

Mamba's recurrent state is **fixed-size** (conv + SSM, independent of sequence length). That changes the migration cost calculus — but only for **pure Mamba**. Hybrid models still carry growing attention KV cache, which dominates at long context.

> **When does async Mamba state migration between peer GPUs improve utilization and tail latency compared to static placement — and under what load and interconnect conditions does it stop being worth it?**

## Scope

### In scope (v1)

* **Pure Mamba** inference (one model variant, one size to start)
* Explicit `MambaState` as a first-class object (conv + SSM buffers on device)
* Single-GPU serving baseline (prefill + decode, no scheduler complexity required)
* Multi-GPU **full-model replicas** (2–4 GPUs), static placement baseline
* Continuous batching and token-budget scheduling (FCFS first, then one migration-aware policy)
* Async state migration between peer workers (NCCL; NIXL optional later)
* Benchmark harness with reproducible workloads and published metrics

### Out of scope (v1)

* Hybrid Mamba + attention models (deferred — KV migration dominates at long context)
* Competing with vLLM on model coverage, quantization, spec decode, prefix caching, multimodal
* Tensor / pipeline / expert parallelism
* Custom HTTP or RPC stacks
* Training, fine-tuning, or a Python ML framework
* Prefill/decode disaggregation (vLLM already covers this; not the research wedge here)

Hybrid models, checkpoint retention policies, and P/D disagg are **future work** after pure-Mamba migration numbers exist.

## Architecture

```text
                    +------------------+
                    | Benchmark Driver |
                    | (workloads, CSV) |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Cluster Scheduler|
                    | (policy plugin)  |
                    +--------+---------+
                             |
           +-----------------+-----------------+
           |                 |                 |
    +------v------+   +------v------+   +------v------+
    |  Worker 0   |   |  Worker 1   |   |  Worker 2   |
    |   GPU 0     |   |   GPU 1     |   |   GPU 2     |
    +------+------+   +------+------+   +------+------+
           |                 |                 |
    +------v------+   +------v------+   +------v------+
    | Mamba RT    |   | Mamba RT    |   | Mamba RT    |
    | State Pool  |   | State Pool  |   | State Pool  |
    +------+------+   +------+------+   +------+------+
           |                 |                 |
           +-----------------+-----------------+
                             |
                    +--------v---------+
                    |  State Transport |
                    |      NCCL        |
                    +------------------+
```

**Policy vs mechanism:** the scheduler decides *whether* to migrate; the transport layer decides *how*.

```cpp
struct MambaState {
    DeviceBuffer conv_state;
    DeviceBuffer ssm_state;
};

class StateTransport {
public:
    virtual TransferHandle transfer(
        StateHandle state,
        WorkerId destination
    ) = 0;
};
```

## Design Principles

1. **Measure first.** Every stage ships with benchmarks and exit criteria. No feature without a metric.
2. **Keep the runtime small.** Use CUDA, cuBLAS, NCCL for mechanism; own the scheduler and state lifecycle.
3. **State is first-class.** Allocatable, migratable, observable — not buried inside the model forward pass.
4. **Separate policy from mechanism.** Swappable placement policies over a fixed transport API.
5. **Pure Mamba first.** Isolate the fixed-size migration story before adding hybrid KV complexity.
6. **Reproducible experiments.** Fixed seeds, documented workloads, results checked into `results/` or a benchmark report.

## Metrics

Every experiment should report:

* Throughput (tokens/sec)
* Time to first token (TTFT)
* Inter-token latency (ITL), p50 / p95
* GPU utilization and memory utilization per worker
* Load skew across workers (max/min queue depth, idle time)
* Scheduler overhead
* State allocation latency
* State migration latency and effective bandwidth
* Policy decision counts (migrations attempted, migrations that improved latency)

## Development Stages

### Stage 1 — Mamba Runtime ✅

GPU forward path for one pure Mamba model: prefill, decode step, explicit conv + SSM state.

**Exit criteria:** logits match a reference implementation within tolerance for prefill + N decode steps.

### Stage 2 — Single-GPU Serving ✅

Basic serving loop on one GPU without distributed or scheduler complexity.

**Exit criteria:** single-sequence and small-batch generation works end-to-end with stable GPU execution.

### Stage 3 — Multi-GPU Static Placement *(current)*

Full-model replicas on 2–4 GPUs. Requests assigned at admission; no migration.

Add:

* Request representation and worker-local state pool
* Continuous batching and token-budget scheduler (FCFS)
* Benchmark harness with configurable load skew

```text
             Scheduler (FCFS)
           /       |        \
          v        v         v
        GPU 0    GPU 1     GPU 2
     (static assignment — no mid-flight moves)
```

**Exit criteria:**

* Reproducible benchmark showing utilization imbalance under skewed workloads
* Baseline numbers for throughput, p95 latency, and per-worker GPU util documented in `docs/experiments/static_placement.md`

### Stage 4 — Async State Migration

Move an active sequence between peer workers without recomputing from scratch.

```text
GPU 0                         GPU 1
Request A (running)           (idle)
State A          ----->       State A
                              Request A resumes decode
```

Add:

* NCCL-backed `StateTransport`
* Migration correctness tests (token stream matches non-migrated path)
* Migration latency and bandwidth microbenchmarks

**Exit criteria:**

* Correct resume after migration for pure Mamba
* Migration cost curve (bytes vs latency) documented per interconnect

### Stage 5 — Migration-Aware Scheduling

One explicit policy beyond static FCFS. Start simple:

```text
migrate(request, dst) iff queue_delay(dst) > migration_cost(request)
```

Sweep:

* Load skew (uniform vs bursty vs long-tail generation lengths)
* Worker count (2, 4 GPUs)
* Interconnect (NVLink vs PCIe if available)

**Exit criteria:**

* Static vs dynamic placement comparison with graphs
* Clear answer: **when migration wins and when it does not** for pure Mamba
* Short write-up suitable for a blog post or upstream vLLM discussion

## Planned Experiments

### Experiment 1 — Static placement under load skew

Compare per-worker GPU utilization and p95 latency when requests are routed round-robin vs least-loaded at admission only.

**Hypothesis:** admission routing helps new requests but cannot fix in-flight skew when generation lengths differ.

### Experiment 2 — Migration break-even

For pure Mamba, find the `(load_skew, migration_latency, queue_depth)` region where mid-flight migration improves p95 latency vs leaving the request put.

**Hypothesis:** migration wins under moderate skew and fast interconnect; wins shrink as queue depth on the destination grows.

### Experiment 3 — Scaling efficiency

Measure throughput at 1, 2, and 4 GPUs under static placement vs migration-aware placement.

**Hypothesis:** static placement loses efficiency at low request counts or high skew; migration recovers some of the gap.

### Experiment 4 — Hybrid extension (future)

Repeat Experiments 1–2 with a hybrid model. Measure how growing KV cache shifts the break-even point as context length increases.

**Hypothesis:** migration cost dominates above a context-length threshold; pure Mamba results do not transfer directly.

## Relationship to vLLM

MambaServe is complementary to vLLM, not competitive.

```text
vLLM          →  production serving, broad model support, P/D disagg, prefix caching
MambaServe    →  controlled experiments on peer migration and placement policies
```

Potential upstream path: benchmark findings from MambaServe (e.g. migration break-even curves, scheduler metrics for hybrid divergent hits) can inform vLLM RFCs or contributions after the testbed produces data.

Systems to study alongside this work:

* vLLM V1 (hybrid allocator, align-mode prefix caching, NIXL connectors)
* NCCL / NIXL
* llm-d

## Non-Goals

* A complete inference platform or vLLM replacement
* Every Mamba model variant
* Hybrid models in v1
* Prefix caching, speculative decoding, quantization breadth
* Custom implementations of mature communication primitives

## Status

Early development. Stages 1–2 (Mamba runtime and single-GPU serving) are in place. Active work is Stage 3: multi-GPU replicas with static placement and the benchmark harness.

```text
Mamba execution          ✅
      |
Single-GPU serving       ✅
      |
Multi-GPU static         ← current
      |
State migration
      |
Migration-aware policy
      |
Hybrid extension         (future)
```

## Objective

> **Build a small, instrumented Mamba serving testbed and use it to measure when dynamic state migration beats static placement — producing numbers that a production runtime like vLLM has no reason to optimize for in isolation.**
