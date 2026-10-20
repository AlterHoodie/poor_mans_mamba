# MambaServe

A small C++/CUDA benchmark testbed for studying hybrid Mamba serving: model execution, cache placement, state migration, and multi-GPU systems.

## What This Project Is for

An exploration of how distributed systems, multi-GPU layouts, cache layouts for different models, CUDA kernels, and different transport or policy choices affect metrics like TTFT, tok/s, and transfer speed.

## High Level Architecture

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
                    | NCCL/NIXL/MemCpy |
                    +------------------+
```

## Metrics

### Generate harness (`benches`)

From `benchmarks/generate_bench.cpp`:

1. Time to first token (TTFT)
2. Prefill throughput (tokens/sec)
3. Decode throughput (tokens/sec)
4. End-to-end throughput (tokens/sec)
5. Inter-token latency (ITL) mean / p50 / p95

Sweeps prompt length `{32, 128, 512, 1024, 2048}` × generation length `{1, 16, 64, 128}` on CPU or GPU (`--device CPU|GPU`). Drive `Runner` directly so EOS cannot truncate a timed run.

Model outputs may not always be meaningful. System prompt templates are not implemented yet.

### Migration harness (`cluster_bench`)

Forced migrate of one decode session from GPU A to GPU B while both GPUs may also run background decode sessions. Reported per migrate:

1. `wait_boundary` (ms): wait until the in-flight decode finishes so the slot is safe to copy
2. `xfer` (ms): host-observed transfer time
3. `commit_ack` (ms): recv completion through commit and source release
4. `stall_excess` (ms): hitch beyond a normal decode step
5. `xfer` GB/s: slot bytes over `xfer` time

Slot size is set by `--max-seq`. Background load is set by `--bg-sessions`.

## How to Run

### 1. Download models

```bash
mkdir -p models
huggingface-cli download tiiuae/Falcon-H1-0.5B-Base --local-dir models/falcon-h1-0.5b-base
huggingface-cli download AntonV/mamba2-130m-hf --local-dir models/mamba2-130m-hf
```

### 2. Build

For CUDA + NCCL + NIXL (pulls UCX, Abseil, NIXL, NCCL into `deps/`, then builds the project):

```bash
./scripts/build.sh
source deps/env.sh
```

Useful knobs:

```bash
CMAKE_CUDA_ARCHITECTURES=80 ./scripts/build.sh   # A100. auto from nvidia-smi if unset
SKIP_APT=1 ./scripts/build.sh                    # skip apt/pip bootstrap
SKIP_SMOKE=1 ./scripts/build.sh                  # skip NIXL smoke at the end
TARGETS="benches transport_bench cluster_bench" ./scripts/build.sh
```

CPU only (no GPU deps, generate benches):

```bash
cmake -S . -B build -DBUILD_BENCHMARKS=ON
cmake --build build --target benches
```

`scripts/build.sh` always enables CUDA, NCCL, and NIXL. Variants whose backend was not built are skipped by `scripts/run_matrix.sh`.

### 3. Run generate benches

```bash
./build/benchmarks/benches --device CPU --benchmark_counters_tabular=true
./build-cuda/benchmarks/benches --device GPU --benchmark_counters_tabular=true
```

### 4. Run migration matrix (2 GPUs)

```bash
STAGES=migrate,analyze \
VARIANTS=memcpy,nccl,nccl_nop2p,nixl_cudaipc,nixl_hoststaged \
MODELS=models/falcon-h1-0.5b-base \
scripts/run_matrix.sh
```

Or call `cluster_bench` directly:

```bash
./build-cuda/benchmarks/cluster_bench \
  --scenario migrate \
  --model-dirs models/falcon-h1-0.5b-base \
  --device GPU \
  --backends nccl,nixl \
  --worker-mode process \
  --max-seq 2048 \
  --loads idle,busy \
  --bg-sessions 16 \
  --out-dir results/migrate_smoke
```

`memcpy` uses thread workers. `nccl` and `nixl` use process workers.

## Results

Migration results measure the time to move one decode session from GPU A to GPU B while both GPUs also run `bg` background decode sessions. Numbers below use falcon-h1-0.5b-base on **2× A100-SXM4-80GB** (NV12), with a **~171.6 MiB** cache slot per session (`max_seq=4096`).

Busy stage waterfall (ms), pinned medians from `results_12` / `results_13`:


| bg  | backend         | wait_boundary | xfer    | commit_ack | stall_excess | xfer GB/s |
| --- | --------------- | ------------- | ------- | ---------- | ------------ | --------- |
| 4   | nccl            | 17.96         | 4.305   | 2.747      | 10.13        | 41.83     |
| 4   | nixl_cudaipc    | 19.46         | 4.594   | 5.406      | 11.45        | 39.2      |
| 4   | nccl_nop2p      | 17.93         | 14.358  | 1.839      | 19.63        | 12.54     |
| 4   | nixl_hoststaged | 15.76         | 724.98  | 1.95       | 724.8        | 0.25      |
| 16  | nccl            | 42.17         | 8.948   | 20.702     | 46.22        | 20.12     |
| 16  | nixl_cudaipc    | 42.33         | 5.506   | 24.272     | 52.08        | 32.71     |
| 16  | nccl_nop2p      | 42.14         | 15.014  | 18.269     | 51.03        | 11.99     |
| 16  | nixl_hoststaged | 42.34         | 686.259 | 30.563     | 757.9        | 0.26      |
| 32  | nccl            | 79.341        | 4.688   | 60.122     | 123.298      | 38.42     |
| 32  | nccl_nop2p      | 79.847        | 14.598  | 55.982     | 128.402      | 12.34     |
| 48  | nccl            | 127.28        | 14.217  | 112.768    | 229.71       | 12.67     |
| 48  | nixl_cudaipc    | 122.88        | 5.352   | 105.501    | 203.18       | 33.65     |
| 48  | nccl_nop2p      | 135.77        | 18.204  | 102.839    | 232.87       | 9.89      |
| 48  | nixl_hoststaged | 134.02        | 745.883 | 114.403    | 969.37       | 0.24      |
| 64  | nccl            | 154.68        | 18.658  | 136.059    | 289.33       | 9.65      |
| 64  | nixl_cudaipc    | 155.1         | 5.32    | 139.389    | 282.87       | 33.85     |
| 64  | nccl_nop2p      | 162.36        | 15.423  | 135.81     | 291.47       | 11.68     |
| 64  | nixl_hoststaged | 162.1         | 671.333 | 149.048    | 969.25       | 0.27      |


Busy `nixl_cudaipc` at bg 32 was not collected.

Idle xfer (ms), same slot:


| bg  | nccl  | nixl_cudaipc | nccl_nop2p | nixl_hoststaged |
| --- | ----- | ------------ | ---------- | --------------- |
| 4   | 1.087 | 1.235        | 12.756     | 703.325         |
| 16  | 1.095 | 1.199        | 9.571      | 596.43          |
| 32  | 1.089 | —            | 9.544      | —               |
| 48  | 1.195 | 1.135        | 12.735     | 692.405         |
| 64  | 1.151 | 1.143        | 9.541      | 624.528         |


## Findings

### 1. Who asks for the data: NIXL sender-WRITE vs receiver-READ

The early design had the sender push the slot. The destination told the parent it wanted a slot, the parent forwarded that to the source worker, and only then did the source post the copy and notify the destination. Every migrate paid for that round trip, and because the announce rode the same single threaded worker loop as `Decode` commands, it queued behind decode work under load. Busy xfer blew up to **~322 ms** at **32** bg sessions in that design.

```text
Before: sender-WRITE, destination announces first

+----------------+        +------------------------+        +----------------+
|   Dst worker   |        |     Parent (relay)     |        |   Src worker   |
+----------------+        +------------------------+        +----------------+
         |                             |                             |
         | ready to receive                                          | waiting to hear
         |                             |                             |   where to send to
         |  "here is my
         |   address, send here"
         +----------------------------->                             |
         |                             | already knows who
         |                             |   this goes to
         |                             +----------------------------->
         |                             |                             | now knows the address
         |                             |                             | starts copying the
         |                             |                             |   slot over
         |                             |                             | keeps checking until
         |                             |                             |   the copy finishes
         | keeps checking for a                                      | copy is done, sends a
         |   "done" signal, sent                                     |   "done" signal straight
         |   straight from src                                       |   to the dst worker
         <-----------------------------------------------------------+
         | done                                                      | done
```

The fix moved two things. First, the `ClusterScheduler` already tracks where every session's slot lives: each worker reports its `slot_ptr` back on `PrefillEvent`/`MigrateEvent`, and the scheduler stores it on the `Session` (`sess.slot_ptr`). So when a migrate starts, `begin_migrate_` sends the destination a `MigrateCmd` that already carries `src_ptr = sess.slot_ptr`. The destination never has to ask the source where the data is, the scheduler's own session metadata answers that half of the handshake before the migrate even begins. Second, each NIXL agent's address info (used to post RDMA style transfers) is published once at worker startup over the control plane (`publish_md`, then `peers_finalized`), not re-sent per transfer.

```text
After: receiver-READ, address already known

                      +-------------------------------------+
                      |          ClusterScheduler            |
                      | Session[7].worker_idx = GPU0         | <- filled in by PrefillEvent,
                      | Session[7].slot_ptr   = 0x7f00...    |    before any migrate starts
                      +-------------------------------------+
                                         |
                 +-----------------------+-----------------------+
                 |                                               |
    MigrateCmd(XFER_RECV,
      src_ptr=0x7f00...)                             MigrateCmd(XFER_SEND)
                 |                                               |
                 v                                               v
       +------------------+                            +------------------+
       |    Dst worker    |                            |    Src worker    |
       |       GPU1       |                            |       GPU0       |
       +------------------+                            +------------------+
            |                                                        |
    NIXL_READ(local=mine,                             never asked where the data is. 
      remote=0x7f00... on GPU0)                      just waits to be told to Release
            |                                                        |
            v                                                        v
      +-----------+       direct copy : Pull from src          +-----------+
      | GPU0 VRAM |------------------------------------------->| GPU1 VRAM | 
      +-----------+                                            +-----------+
```

With the source address already in hand, the destination posts `NIXL_READ` immediately and the source is never in the data path at all, it just waits to be told it can release the slot. Busy xfer dropped from the **~322 ms** announce path into the same **few-to-tens-of-ms** band as NCCL.

### 2. NCCL's default path shares the SM cores with decode

NCCL's usual GPU to GPU P2P path uses a CUDA kernel (`ncclDevKernel_SendRecv`) that runs on the SMs to coordinate communication, rather than relying entirely on a DMA engine. These are the same SMs that execute decode kernels such as `mamba2_ssm_scan`, attention, GEMV, and RoPE, so communication can compete with inference for GPU execution resources. NCCL can also use alternative mechanisms, including CUDA copy engines on supported topologies, but the P2P path measured here uses an SM resident communication kernel.

```text
GPU SM grid (A100, ~108 SMs)

      idle decode                         busy decode (many bg sessions)
  +----+----+----+----+----+----+     +----+----+----+----+----+----+
  |    | NC |    |    |    |    |     | de | de | de | de | de | NC |
  +----+----+----+----+----+----+     +----+----+----+----+----+----+
  |    |    |    |    |    |    |     | de | de | de | de | de | de |
  +----+----+----+----+----+----+     +----+----+----+----+----+----+
   plenty of free SMs for NCCL's        SMs mostly full of decode (de);
   SendRecv kernel (NC)                 NCCL's SendRecv kernel (NC)
                                        has to wait its turn
```

That is why idle NCCL xfer is flat at **~1.1 ms** (mostly a fixed launch cost plus the worker's **1 ms** poll tick), but busy stall climbs steadily with background load, from **~10 ms** at bg **4** to **~289 ms** at bg **64**. The transfer kernel is not slower on the wire, it is waiting for SM occupancy the same way any other launched kernel would. `wait_boundary` (waiting for the in-flight decode step to end before the slot is safe to copy) and `commit_ack` (completion handshake) dominate the hitch far more than the few milliseconds the actual copy takes.

### 3. NIXL's cuda_ipc path uses the copy engine, not the SMs

`nixl_cudaipc` moves bytes with a `PtoP memcpy` on the GPU's copy engine (the DMA/copy engine), a separate piece of silicon from the SMs that CUDA kernels run on. It does not compete with decode for compute cores.

```text
GPU die (simplified)

  +-----------------------------+
  |   SM SM SM SM SM SM SM SM   |  <- decode kernels live here
  |   SM SM SM SM SM SM SM SM   |     (mamba2_ssm_scan, attention, ...)
  +-----------------------------+
  |      Copy / DMA engine      |  <- NIXL cuda_ipc PtoP memcpy lives here
  +-----------------------------+     independent queue, own hardware
```

nsys shows this directly: NCCL's `ncclDevKernel_SendRecv` holds the SMs for **~16 ms** per **~100 MiB** slot in a light-bg capture, while NIXL's `PtoP memcpy` finishes in **~0.5 ms** on the copy engine. Even so, under heavy bg load the two backends' *stall* numbers track closely (e.g. **~290 ms** NCCL vs **~283 ms** `nixl_cudaipc` at bg **64**), because stall is dominated by `wait_boundary` and `commit_ack`, protocol costs shared by both backends, not by which engine does the copy. The copy engine wins on the wire, but the wire was never the bottleneck here.

### 4. NIXL cuda_ipc (NVLink) vs host-staged (D2H/H2D over TCP)

`nixl_cudaipc` sets `UCX_TLS=self,tcp,sm,cuda_copy,cuda_ipc`, so UCX maps the peer GPU's memory directly and the copy engine moves bytes GPU VRAM to GPU VRAM over NVLink in one hop.

`nixl_hoststaged` drops `cuda_ipc` from that list (`UCX_TLS=self,tcp,sm,cuda_copy`), so UCX has no direct GPU to GPU path left and falls back to bouncing through host memory over its TCP transport, even though both GPUs are in the same box:

```text
nixl_cudaipc: one hop, over NVLink

  +---------------+                                        +---------------+
  |   GPU0 VRAM   |-------------- NVLink P2P ------------->|   GPU1 VRAM   |
  +---------------+                                        +---------------+


nixl_hoststaged: four hops, bounced through host memory

  +-----------+     +-----------+     +-----------+     +-----------+
  | GPU0 VRAM |---> | Host RAM  |---> | Host RAM  |---> | GPU1 VRAM |
  +-----------+     +-----------+     +-----------+     +-----------+
 D2H over PCIe     pinned buffer     TCP loopback      H2D over PCIe
                   (extra copy)      (kernel socket)   (extra copy)
```

Each extra hop adds a real cost: a PCIe device to host copy, a pass through the host network stack (even looped back locally), a host to device copy on the other side, and the pinned staging buffers in between. That is the whole gap between **~0.24–0.27 GB/s** for `nixl_hoststaged` and the **10–40 GB/s** class numbers for `nixl_cudaipc` in the table above. It is not a NIXL weakness, it is a UCX configuration that happens to drop the one transport (`cuda_ipc`) that avoids the host round trip. Treat `nixl_hoststaged` as a misconfigured baseline, not a fair NIXL number, and use `nccl_nop2p` instead as the sane no-P2P reference.

## Planned next steps

1. Run transport progress (`getXferStatus`, `getNotifs`, control messages) off the decode loop, or prioritize it over `Decode`. Busy stall is still dominated by `wait_boundary` and `commit_ack`.
2. A/B the NIXL agent progress thread (`cfg(false)`) on `balanced_off`. Earlier runs saw NIXL decode ~5–6% slower than NCCL.
3. Eagerly `loadRemoteMD` after `peers_finalized` and do one warm-up transfer to remove the ~34–40 ms first-migration cost. NCCL pays its equivalent at bootstrap.
4. Record the UCX version and fix the `nixl_hoststaged` `UCX_TLS`. Compare against `nccl_nop2p` as the no-P2P reference.
5. Capture or finish reading nsys at bg 32–48: whether decode bars stretch while NCCL `SendRecv` holds SMs, versus NIXL `PtoP memcpy` on the copy engine. Some `results_13/a100_nsys_bg_*` captures exist.
6. Re-run `rebalance` to completion for all variants, with more than one repeat.

