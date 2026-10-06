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
scripts/run_matrix.sh            # full matrix on a 2-GPU VM -> report.md
```

`scripts/run_matrix.sh` runs per transport variant (`memcpy`, `nccl`, `nccl_nop2p`, `nixl_cudaipc`, `nixl_hoststaged`), records the NCCL/UCX path actually chosen (`paths` stage), compares `MAMBASERVE_LOG=off` vs `debug` (`logcheck` stage), and calls `scripts/analyze.py` for the report. Cluster stages use each backend's native worker mode (`thread` for memcpy, `process` for nccl/nixl). Use `STAGES=...`, `DEVICE=CPU REPEATS=1` for subsets or a no-GPU smoke test.

**Telemetry.** `MAMBASERVE_LOG=debug|info|warn|error|off` (default `warn`) controls logging. The trace recorder (`include/telemetry/recorder.h`) stamps submit, prefill, every decode, and each migrate phase per thread with no shared lock; `cluster_bench` enables it and derives TTFT, ITL, migrate wait/transfer/ack/stall, and background-session interference from it. Counters (bytes migrated, rebalance decisions, slot rejects, ...) land in `summary.csv`. NVTX ranges for Nsight Systems are compiled in with `-DMAMBASERVE_WITH_NVTX=ON`.

**Reading the numbers.**

* Workers are threads of one process by default (`ClusterConfig::worker_mode = WorkerMode::Thread`). `WorkerMode::Process` (`cluster_bench --worker-mode process`) runs each worker as a child process that re-execs the same binary over a Unix socket; it needs GPU with NCCL or NIXL, because MemcpyPeer passes raw pointers between workers and is thread-mode only. The host creates workers with `create_cluster_workers(cfg)` (`worker_factory.h`) and passes them to `ClusterScheduler::start(cfg, workers)`; the scheduler never spawns anything itself. Any binary that hosts process workers must call `maybe_run_worker_process(argc, argv)` first in `main()`. In process mode, `cluster_bench` syncs recorder epochs and pulls worker TraceBatches over IPC at the end of each run so TTFT/ITL/migrate phases match thread mode. NIXL runs over UCX between local GPUs (`cuda_ipc` or host staged), not network RDMA, so results are intra-node only.
* A migrate always moves the whole slot (`slot_bytes`, set by `--max-seq`), not just the used KV prefix. State size is therefore swept with `--max-seq`.
* The worker polls transfers on a 1 ms tick, so host-side `xfer_us` is only reliable above about 1 ms. Use `transport_bench` for fine-grained transfer latency.
* NCCL and NIXL senders ack at post time; the receive side marks real completion.
* Needs at least `workers + 3` hardware threads (workers, event thread, driver, migrate driver) for clean numbers.

## Findings: NCCL vs NIXL migration (intra-node)

Two generations of NIXL results are kept here. The first is the original sender-WRITE design with a per-transfer announce (**before**). The second is the receiver-driven design that piggybacks NIXL metadata and control on the cluster transport plane (**after**, see [After: piggybacked NIXL](#after-piggybacked-nixl-on-the-cluster-transport-plane)).

### Before: sender-WRITE NIXL with per-transfer announce

Setup: VM with 2x A100-SXM4-80GB (NV12 between the GPUs), falcon-h1-0.5b-base, process-per-worker for NCCL and NIXL. Early runs had zeros for NCCL/NIXL xfer because process-mode traces were not collected yet; the table below uses the later runs. In the latest of those runs `mamba2-130m-hf` failed to load, so all numbers are falcon-h1 only.

| Scenario | NCCL (P2P) | NIXL (cuda_ipc), before |
|---|---|---|
| Idle xfer, 100 MiB / 172 MiB | 1.2 ms / 1.1 ms | 3.2 ms / 3.2 ms |
| Busy xfer, 2 background sessions | ~18-24 ms | ~20-25 ms |
| Busy xfer, 32 background sessions | 24-29 ms | ~322 ms |
| Rebalance `pinned_on` makespan | baseline | within 1-5% of NCCL |
| `balanced_off` makespan (no migrations) | 13.76 s | 14.6-14.8 s (59.7 s vs 50.7 s in the 32-session run) |

**Takeaways (before)**

* **Idle xfer is not a bandwidth number.** Time is flat across 100 and 172 MiB for both backends, so it measures fixed latency plus the 1 ms poll floor. Only `nccl_nop2p` scales with size (about 14 GB/s). `transport_bench` has not been run for these results (no `paths/` or `transport/` output), and `ucx_version` is not recorded, so `cuda_ipc` use is unconfirmed.
* **NIXL's extra ~2 ms idle is the handshake.** NCCL posts `ncclSend`/`ncclRecv` independently. NIXL does a sender WRITE, so the destination address travels recv worker -> parent (`ingress_loop_`, 1 ms idle sleep) -> src worker. The src worker must then post and poll `getXferStatus` (it flushes the WRITE and sends the notif), and the recv side polls `getNotifs`. Each hop is quantized to ~1 ms.
* **The ~322 ms busy xfer is about one decode round.** With 32 background sessions the ITL is ~305 ms. The announce shares the worker's FIFO IPC channel with queued `Decode` commands, and `Worker::loop_` is single-threaded, so the announce waits behind them. NCCL has no such dependency. `wait_boundary` and `commit_ack` (~300 ms) are the same for every backend, so they do not separate the two.
* **NIXL workers decode slower even with no migration.** ITL outside migration windows is 20.3 ms vs 19.0 ms for NCCL, and `pinned_off`/`balanced_off` are 5-7% slower (up to 18% under the 32-session load). Likely cause: `nixlAgentConfig cfg(true)` in `NixlCommAgent::init_backend_` turns on the agent's progress thread (NIXL's default is off, `kDefaultUseProgThread = false`, and the UCX backend then runs a shared progress thread with `pthrDelay = 0`). That thread, plus `tcp` in `UCX_TLS`, probably competes with launch-bound decode. That the thread is on is confirmed from the NIXL source; that it causes the slowdown is not yet measured. Test with `cfg(false)`.
* **First NIXL migration is slow.** About 20-30 ms vs 3.2 ms steady state, from lazy `loadRemoteMD` and UCX wireup. NCCL does its setup at bootstrap.
* **`nixl_hoststaged` (~0.24 GB/s) is a configuration artifact,** likely UCX falling back to TCP loopback. It is not comparable to `nccl_nop2p` (~14 GB/s over SHM).

**Versus Ray's experimental NIXL transport** (`ray/experimental/rdt/nixl_tensor_transport.py`):

| | MambaServe NIXL (before) | Ray |
|---|---|---|
| Direction | Sender WRITE + notif | Receiver READ (one-sided) |
| Address exchange | Per-transfer announce via parent and src worker | Descriptors and agent metadata ride in object metadata over Ray's control plane |
| Sender in data path | Yes (must post and poll) | No |
| Registration | Whole slab once | Per object (ref-counted), or an optional preregistered pool; receiver registers fresh buffers per fetch |
| Remote agent load | Lazy on first post | LRU cache, invalidated by a metadata version bump |
| Completion polling | 1 ms tick in the worker loop | `check_xfer_state` with `sleep(0.001)` |

Ray avoids the sender-gating problem because the receiver drives the transfer. The preregistered slab here is cheaper per transfer than Ray's per-object registration.

### After: piggybacked NIXL on the cluster transport plane

What changed (`src/comm/nixl_comm_agent.cpp`, `src/comm/nixl_transport_control.cpp`, `src/runtime/worker.cpp`):

* The receiver drives the transfer: `NixlCommAgent::create_xfer_req_` posts a `NIXL_READ` from the source slot, and the source pointer rides in the migrate command (`cmd.src_ptr()`). The sender is no longer in the data path and there is no per-transfer announce hop through the parent.
* Agent metadata is published once at startup over the cluster transport control plane (`publish_md` after Ready, then `peers_finalized`), instead of being exchanged per transfer.
* `loadRemoteMD` is still lazy on the first post, so the first migration still pays a wire-up cost.

Setup: same VM (2x A100-SXM4-80GB, NV12), falcon-h1-0.5b-base, process-per-worker, `UCX_TLS=self,tcp,sm,cuda_copy,cuda_ipc`, git `8ab0518`. Primary migrate numbers: n=10 per cell, 16 background sessions for busy. Rebalance numbers come from an earlier short run (n=3, 8 background sessions). The fuller matrix was cut short during rebalance, so only migrate (all variants) and `rebalance/memcpy` completed there.

Migrate, 99.7 MiB slot (`--max-seq 2048`):

| Metric | NIXL cuda_ipc, before | NIXL cuda_ipc, after | NCCL P2P, after |
|---|---|---|---|
| Idle xfer, steady | 3.2 ms | 1.15 ms | 1.14 ms |
| Idle xfer, first migration | ~67 ms | ~34 ms | ~6 ms |
| Busy xfer | ~322 ms (32 bg sessions) | ~16 ms (16 bg sessions) | ~23.5 ms (16 bg sessions) |
| Busy `wait_boundary` / `commit_ack` | ~322 ms / ~297 ms | ~141 ms / ~121 ms | ~132 ms / ~118 ms |
| Busy `stall_excess` | ~904 ms | ~248 ms | ~241 ms |
| Background ITL during vs outside migration | 0.99x | 0.99x | 1.00x |
| `nixl_hoststaged` idle / busy xfer | ~395 ms / ~862 ms | ~220 ms / ~340 ms | n/a |

Busy "before" is 32 background sessions (decode round about 305 ms) and busy "after" is 16 (decode round about 125 ms), so the busy rows are not a matched load comparison. The idle rows are matched.

Migrate across state size (busy, `stall_excess`; after):

| max_seq | slot | NIXL cuda_ipc xfer / stall | NCCL xfer / stall | memcpy stall |
|---|---|---|---|---|
| 512 | 45.7 MiB | 15.7 ms / 237 ms | 21.6 ms / 248 ms | 312 ms |
| 1024 | 63.7 MiB | 15.8 ms / 240 ms | 22.6 ms / 241 ms | 310 ms |
| 2048 | 99.7 MiB | 16.1 ms / 248 ms | 23.5 ms / 241 ms | 308 ms |

Rebalance (`pinned_on` threshold 2, 16 sessions; short run):

| Backend | Makespan | vs `pinned_off` | Migrations | `stall_excess` median |
|---|---|---|---|---|
| NIXL cuda_ipc | 23.08 s | 1.53x | 12 | 156 ms |
| NCCL P2P | 22.78 s | 1.50x | 10 | 195 ms |
| memcpy | 27.81 s | 1.22x | 12 | 202 ms |

**Takeaways (after)**

* **The busy-xfer gap is gone.** Before, busy NIXL xfer (~322 ms) was about one decode round because the announce queued behind `Decode` commands. Now busy xfer is ~16 ms, far below a decode round (~125 ms), and slightly faster than NCCL (~23 ms). Idle xfer matches NCCL (~1.15 ms vs ~1.14 ms), with the ~2 ms handshake gone.
* **Busy stall is now the same for NIXL and NCCL, and it is not the wire.** `stall_excess` is ~240-250 ms for both and flat across 46-100 MiB. About 140 ms is `wait_boundary` and ~120 ms is `commit_ack`, so xfer is ~16 ms of a ~395 ms total (about 4%). Speeding up the copy further will not move stall.
* **Doubling the background load doubled stall, not xfer.** Going from 8 sessions (stall ~120 ms, ITL ~60 ms) to 16 (stall ~240 ms, ITL ~120 ms) scales stall with ITL for both NIXL and NCCL. Xfer stayed ~16 ms for NIXL and ~21-25 ms for NCCL, and background ITL during migration stayed at ~1.0x. This does not show NCCL being hurt more by load yet; that needs more concurrent decode or bigger models (see next steps).
* **NIXL workers still decode a bit slower than NCCL.** ITL outside migration windows is 124-128 ms vs 118-119 ms (about 5-6%) at 16 sessions, and `balanced_off` is 21.4 s vs 20.3 s in the short run. The agent progress thread (`nixlAgentConfig cfg(true)`) is still the suspect and still unmeasured.
* **First migration still has a wire-up cost.** Idle first xfer is ~34-40 ms for NIXL (steady ~1.2 ms), from the lazy `loadRemoteMD`. NCCL does its setup at bootstrap (first xfer ~4-6 ms).
* **Rebalance is on par with NCCL.** NIXL `pinned_on` reaches 1.53x vs 1.50x for NCCL with a lower median per-migration stall (156 ms vs 195 ms, with 12 vs 10 migrations). Only one short run per backend, so treat the difference as noise-level.
* **`nixl_hoststaged` is still a configuration artifact.** Xfer is better than before (idle ~395 ms to ~220 ms) but still scales with slot size (127 to 220 ms idle, 215 to 340 ms busy over 46-100 MiB) and is the only variant that slows background ITL (1.09-1.13x).
* **nsys companion** (3 sessions, 1 timed run): steady cuda_ipc xfer ~1.1 ms vs ~20 ms NCCL, host-staged ~445 ms. Profiles were recorded with Nsight Systems 2026.5.1, so open them with 2026.5.1 or newer. Numbers include profiler overhead.

**Planned next steps**

1. ~~Receiver-driven READ~~ (done): the src pointer rides in the migrate command, the receiver posts `NIXL_READ`, and agent metadata is published over the cluster transport control plane. Busy xfer dropped from ~322 ms to ~16 ms.
2. Run transport progress (`getXferStatus`, `getNotifs`, control messages) off the decode loop, or prioritize it over `Decode`. Lower priority now that the announce no longer queues behind `Decode`; busy stall is dominated by `wait_boundary` and `commit_ack`, which are the same for NCCL and NIXL.
3. A/B the agent progress thread (`cfg(false)`) on `balanced_off`; NIXL decode is still ~5-6% slower than NCCL.
4. Eagerly `loadRemoteMD` after `peers_finalized` and do one warm-up transfer, to remove the ~34-40 ms first-migration cost.
5. Run the `paths` and `transport` stages, record the UCX version, and fix the `nixl_hoststaged` `UCX_TLS`.
6. Stress the NCCL-vs-NIXL contention question: re-run `migrate` for `nccl` and `nixl_cudaipc` with 32-48 background sessions (and larger `--max-seq` or a bigger model). 8 to 16 sessions only scaled stall with ITL and did not separate the two backends.
7. Re-run `rebalance` to completion for all variants (the fuller matrix was cut short mid-rebalance), with more than one repeat.

## Status

CPU Mamba2 and Falcon-H1 runners, hybrid cache, scheduler, tokenizer, CUDA backends, and the generate latency/throughput harness are in place. Multi-GPU static placement (full-model replicas with admission-time routing) and live state migration over MemcpyPeer, NCCL, and NIXL/UCX are implemented and benchmarked on a 2x A100-SXM VM (see Findings above). NIXL migration now matches NCCL intra-node on cuda_ipc after moving to receiver-driven READ with metadata exchanged over the cluster transport plane (busy xfer ~16 ms vs ~23 ms for NCCL; the old ~322 ms busy xfer is gone). Remaining work: the ~5-6% NIXL decode slowdown, the first-migration wire-up cost, higher-contention comparisons, complete rebalance runs, and clean transport microbenchmarks.

```text
Pure Mamba CPU           ✅
      |
Hybrid CPU               ✅
      |
Hybrid GPU               ✅
      |
Multi-GPU static         ✅
      |
NCCL migration           ✅
      |
NIXL migration           ← current (receiver-driven READ, on par with NCCL intra-node; tuning decode overhead and warmup)
```

## Objective

> **Build a small, instrumented hybrid Mamba serving testbed and measure how execution backends, cache placement, and state transport affect serving performance from one CPU to multiple GPUs.**
