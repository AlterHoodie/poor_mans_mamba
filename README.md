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

Design notes for the cluster scheduler / workers / transports: [docs/cluster_runtime.md](docs/cluster_runtime.md). Implemented CommAgent backends and migration FSM: [docs/comm_and_migration.md](docs/comm_and_migration.md).

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

## Multi-GPU benchmarking and telemetry

Two plain-`main()` drivers (CSV output) sit next to the google-benchmark `benches` target:

* `transport_bench`: model-free CommAgent microbenchmark (MemcpyPeer / NCCL / NIXL plus a raw `cudaMemcpyPeerAsync` baseline). Size sweep, uni/bidirectional, one-time setup cost reported separately.
* `cluster_bench`: drives `ClusterScheduler`. Scenarios `scale` (workers x sessions), `migrate` (forced migrations, idle vs busy peer GPU, state size via `--max-seq`), `rebalance` (skewed load, policy off/on).

```text
cmake -S . -B build-cuda -DMAMBASERVE_WITH_CUDA=ON -DMAMBASERVE_WITH_NCCL=ON \
      -DMAMBASERVE_WITH_NIXL=ON -DBUILD_BENCHMARKS=ON [-DMAMBASERVE_WITH_NVTX=ON]
cmake --build build-cuda --target transport_bench cluster_bench
scripts/run_matrix.sh            # full matrix on a 2-GPU VM -> results/<timestamp>/report.md
```

`scripts/run_matrix.sh` runs per transport variant (`memcpy`, `nccl`, `nccl_nop2p`, `nixl_cudaipc`, `nixl_hoststaged`), records the NCCL/UCX path actually chosen (`paths` stage), compares `MAMBASERVE_LOG=off` vs `debug` (`logcheck` stage), and calls `scripts/analyze.py` for the report. Cluster stages use each backend's native worker mode (`thread` for memcpy, `process` for nccl/nixl). Use `STAGES=...`, `DEVICE=CPU REPEATS=1` for subsets or a no-GPU smoke test.

**Telemetry.** `MAMBASERVE_LOG=debug|info|warn|error|off` (default `warn`) controls logging. The trace recorder (`include/telemetry/recorder.h`) stamps submit, prefill, every decode, and each migrate phase per thread with no shared lock; `cluster_bench` enables it and derives TTFT, ITL, migrate wait/transfer/ack/stall, and background-session interference from it. Counters (bytes migrated, rebalance decisions, slot rejects, ...) land in `summary.csv`. NVTX ranges for Nsight Systems are compiled in with `-DMAMBASERVE_WITH_NVTX=ON`.

**Reading the numbers.**

* Workers are threads of one process by default (`ClusterConfig::worker_mode = WorkerMode::Thread`). `WorkerMode::Process` (`cluster_bench --worker-mode process`) runs each worker as a child process that re-execs the same binary over a Unix socket; it needs GPU with NCCL or NIXL, because MemcpyPeer passes raw pointers between workers and is thread-mode only. The host creates workers with `create_cluster_workers(cfg)` (`worker_factory.h`) and passes them to `ClusterScheduler::start(cfg, workers)`; the scheduler never spawns anything itself. Any binary that hosts process workers must call `maybe_run_worker_process(argc, argv)` first in `main()`. In process mode, `cluster_bench` syncs recorder epochs and pulls worker TraceBatches over IPC at the end of each run so TTFT/ITL/migrate phases match thread mode. NIXL runs over UCX between local GPUs (`cuda_ipc` or host staged), not network RDMA, so results are intra-node only.
* A migrate always moves the whole slot (`slot_bytes`, set by `--max-seq`), not just the used KV prefix. State size is therefore swept with `--max-seq`.
* The worker polls transfers on a 1 ms tick, so host-side `xfer_us` is only reliable above about 1 ms. Use `transport_bench` for fine-grained transfer latency.
* NCCL and NIXL senders ack at post time; the receive side marks real completion.
* Needs at least `workers + 3` hardware threads (workers, event thread, driver, migrate driver) for clean numbers.

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
