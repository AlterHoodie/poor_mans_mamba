# MambaServe

A small C++/CUDA **benchmark testbed** for studying hybrid Mamba serving: model execution, cache placement, state migration, and multi-GPU systems.

## What This Project Is

```text
MambaServe = minimal runtime + reproducible benchmarks + explicit policies
```

The runtime is intentionally small. The deliverable is **data and tradeoffs**.

## Central Question

The project starts with a correct CPU implementation of the Mamba path, then adds hybrid layers and progressively more capable cache and transport backends. The goal is to make each systems transition measurable rather than to reproduce a production serving stack.

Pure Mamba has fixed-size recurrent state: convolution state plus SSM state. Hybrid models add attention KV cache, whose size grows with context. This gives the project a useful progression from fixed-size state movement to realistic hybrid cache movement.

> **How do hybrid model state, cache placement, and transport choices affect serving throughput and latency as the system moves from one CPU to multiple GPUs?**

## Scope

* Hybrid Mamba + attention inference, starting with a CPU reference path
* Explicit recurrent state and attention KV cache as first-class cache data
* CPU and GPU execution backends
* Multi-GPU full-model replicas with static placement
* State/cache movement experiments using NCCL, then NIXL
* A small benchmark harness with reproducible workloads and published metrics

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

## Metrics

The generate harness (`benchmarks/generate_bench.cpp`, target `benches`) reports:

* Time to first token (TTFT)
* Prefill throughput (tokens/sec)
* Decode throughput (tokens/sec)
* End-to-end throughput (tokens/sec)
* Inter-token latency (ITL) mean / p50 / p95

Sweeps prompt length `{32, 128, 512, 1024, 2048}` × generation length `{1, 16, 64, 128}` on CPU or GPU (`--device CPU|GPU`). Drive `Runner` directly so EOS cannot truncate a timed run.

```text
cmake -S . -B build-cuda -DMAMBASERVE_WITH_CUDA=ON -DBUILD_BENCHMARKS=ON
cmake --build build-cuda --target benches
./build-cuda/benchmarks/benches --device CPU --benchmark_counters_tabular=true
```

## Status

CPU Mamba2 and Falcon-H1 runners, hybrid cache, scheduler, tokenizer, CUDA backends, and the generate latency/throughput harness are in place. Active work is multi-GPU static placement: full-model replicas with admission-time routing.

```text
Pure Mamba CPU           ✅
      |
Hybrid CPU               ✅
      |
Hybrid GPU               ✅
      |
Multi-GPU static         ← current
      |
NCCL migration
      |
NIXL migration
```

## Objective

> **Build a small, instrumented hybrid Mamba serving testbed and measure how execution backends, cache placement, and state transport affect serving performance from one CPU to multiple GPUs.**
