# MambaServe

A small C++/CUDA **benchmark testbed** for studying hybrid Mamba serving: model execution, cache placement, state migration, and multi-GPU systems.

This is not a vLLM replacement. vLLM already serves Mamba and hybrid models in production. MambaServe is a small, instrumented environment for learning how hybrid state and cache movement affect serving systems.

## What This Project Is

```text
MambaServe = minimal runtime + reproducible benchmarks + explicit policies
```

The runtime is intentionally small. The deliverable is **data and tradeoffs**, not feature parity with vLLM.

| vLLM | MambaServe |
| --- | --- |
| Serve many models in production | Study one model path at a time |
| Broad hardware and feature coverage | Explicit CPU → GPU → multi-GPU ladder |
| Admission-time routing and P/D disagg | Controlled cache placement and migration experiments |
| Implicit scheduler heuristics | Explicit, swappable placement policies |
| Large Python + PyTorch stack | Small C++/CUDA control plane |

Use vLLM as a **correctness and feature ceiling reference**. Use MambaServe to answer questions vLLM is not designed to optimize for.

## Central Question

The project starts with a correct CPU implementation of the Mamba path, then adds hybrid layers and progressively more capable cache and transport backends. The goal is to make each systems transition measurable rather than to reproduce a production serving stack.

Pure Mamba has fixed-size recurrent state: convolution state plus SSM state. Hybrid models add attention KV cache, whose size grows with context. This gives the project a useful progression from fixed-size state movement to realistic hybrid cache movement.

> **How do hybrid model state, cache placement, and transport choices affect serving throughput and latency as the system moves from one CPU to multiple GPUs?**

## Scope

### In scope

* Hybrid Mamba + attention inference, starting with a CPU reference path
* Explicit recurrent state and attention KV cache as first-class cache data
* CPU and GPU execution backends
* Multi-GPU full-model replicas with static placement
* State/cache movement experiments using NCCL, then NIXL
* A small benchmark harness with reproducible workloads and published metrics

### Out of scope

* Competing with vLLM on model coverage, quantization, spec decode, prefix caching, multimodal
* Tensor / pipeline / expert parallelism
* Custom HTTP or RPC stacks
* Training, fine-tuning, or a Python ML framework
* A production-grade API server or general-purpose scheduler

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

Tokenizer owns text↔token conversion. Scheduler owns sampling and stopping. Runner owns forward execution. Each runner has a cache pool for its device; a later engine-level scheduler may own allocation decisions while runners consume handles. `StateTransport` moves cache blobs between pools without exposing runner internals.

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
5. **Hybrid learning ladder.** Establish CPU correctness before GPU execution, static multi-GPU placement, and transport migration.
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

## Development Ladder

### Stage 1 — Hybrid CPU *(current)*

Extend the existing CPU Mamba runtime with the first hybrid model path and a cache representation for recurrent state plus attention KV.

**Exit criteria:**

* Hybrid logits match a reference implementation within tolerance
* Prefill and decode work for one sequence
* Cache ownership and release are explicit and tested

### Stage 2 — Hybrid GPU

Add GPU tensors and kernels for the hybrid forward path and device-local cache allocation.

**Exit criteria:** CPU and GPU token streams agree within tolerance, with measured prefill and decode latency.

### Stage 3 — Multi-GPU Static Placement

Run full-model replicas on 2–4 GPUs. Requests are assigned at admission; no mid-flight migration.

Add:

* Worker-local cache pools
* A token-budget scheduler
* A benchmark harness with configurable load skew

**Exit criteria:** reproducible throughput, p95 latency, cache usage, and per-worker utilization under skewed workloads.

### Stage 4 — NCCL State Migration

Move an active request between peer workers without recomputing from scratch.

Add:

* NCCL-backed `StateTransport`
* Migration correctness tests for recurrent state and KV cache
* Migration latency and effective-bandwidth microbenchmarks

**Exit criteria:**

* Correct resume after migration for hybrid models
* Migration cost curve documented per interconnect

### Stage 5 — NIXL Migration

Compare NIXL with the NCCL transport for the same cache movement operations and workloads.

**Exit criteria:** documented latency, bandwidth, and operational tradeoffs for NCCL versus NIXL.

### Stage 6 — Migration-Aware Scheduling

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
* Clear answer: **when migration wins and when it does not** for hybrid models
* Short write-up suitable for a blog post or upstream systems discussion

## Planned Experiments

### Experiment 1 — Static placement under load skew

Compare per-worker GPU utilization and p95 latency when requests are routed round-robin vs least-loaded at admission only.

**Hypothesis:** admission routing helps new requests but cannot fix in-flight skew when generation lengths differ.

### Experiment 2 — Hybrid cache cost

Measure how context length changes cache size, prefill/decode cost, and the break-even point for moving recurrent state plus KV cache.

**Hypothesis:** migration becomes less attractive as attention KV dominates the moved state.

### Experiment 3 — Scaling efficiency

Measure throughput at 1, 2, and 4 GPUs under static placement vs migration-aware placement.

**Hypothesis:** static placement loses efficiency at low request counts or high skew; migration recovers some of the gap.

### Experiment 4 — Transport comparison

Compare CPU copies, GPU/NCCL transfers, and NIXL transfers for the same cache payloads.

**Hypothesis:** the best transport depends on payload size, topology, and whether migration overlaps useful decode work.

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
* Prefix caching, speculative decoding, quantization breadth
* Custom implementations of mature communication primitives

## Status

Early development. The CPU pure-Mamba runtime, cache pool, synchronous scheduler, and tokenizer path are in place. Active work is Stage 1: hybrid model support and a hybrid cache on CPU.

```text
Pure Mamba CPU           ✅
      |
Hybrid CPU               ← current
      |
Hybrid GPU
      |
Multi-GPU static
      |
NCCL migration
      |
NIXL migration
```

## Objective

> **Build a small, instrumented hybrid Mamba serving testbed and measure how execution backends, cache placement, and state transport affect serving performance from one CPU to multiple GPUs.**
