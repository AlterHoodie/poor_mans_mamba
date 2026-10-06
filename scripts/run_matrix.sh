#!/usr/bin/env bash
# Runs the full benchmark matrix on a 2-GPU VM and writes everything under
# results/<timestamp>/ (override with OUT=...). Afterwards runs analyze.py.
#
#   scripts/run_matrix.sh                       # everything
#   STAGES=transport,migrate scripts/run_matrix.sh
#   DEVICE=CPU STAGES=scale REPEATS=1 scripts/run_matrix.sh   # smoke test, no GPU
#   DEVICE=CPU BIN_DIR=build/benchmarks VARIANTS=memcpy STAGES=perf scripts/run_matrix.sh
#
# Stages: env paths transport scale migrate rebalance logcheck nsys perf analyze
#
# Knobs (environment variables):
#   BIN_DIR              directory with transport_bench / cluster_bench   (build-cuda/benchmarks)
#   MODELS               comma separated model dirs   (models/falcon-h1-0.5b-base,models/mamba2-130m-hf)
#   DEVICE               GPU | CPU                    (GPU)
#   REPEATS              measured repeats per config  (5)
#   MAX_SEQS             slot sizes for the migrate sweep (512,1024,2048,4096)
#   VARIANTS             transport variants to run (see below)
#   OUT                  output root
#   BG_SESSIONS          background sessions for migrate busy load (2)
#   MIGRATIONS           forced migrates of the primary session (3)
#   SESSIONS_REBALANCE   concurrent sessions in rebalance (8)
#   THRESHOLDS           rebalance thresholds when *_on (2,4)
#   REBALANCE_MAX_SEQ    max-seq for rebalance stage (2048)
#   PROMPT_LEN / GEN_LEN prompt / decode length (128 / 64)
#   PERF                 perf binary (auto-resolved; needed on WSL where /usr/bin/perf is a stub)
#   PERF_CALL_GRAPH      dwarf | fp | lbr                  (dwarf)
#   PERF_FREQ            sample frequency Hz               (99)
#   PERF_MAX_SEQ         max-seq for the perf stage        (512)
#   PERF_STAT            1 to also run perf stat (2nd pass) (0)
#
# A100-80GB stress example (falcon-h1-0.5b ~100MiB/slot @2048, ~170MiB @4096,
# ~316MiB @8192; each worker preallocates SESSIONS slots):
#   STAGES=migrate,rebalance,analyze \
#   BG_SESSIONS=48 SESSIONS_REBALANCE=64 THRESHOLDS=1,2 \
#   MAX_SEQS=2048,4096,8192 REBALANCE_MAX_SEQ=8192 \
#   MIGRATIONS=5 GEN_LEN=128 REPEATS=3 \
#   VARIANTS=memcpy,nccl,nccl_nop2p,nixl_cudaipc,nixl_hoststaged \
#   scripts/run_matrix.sh
#
# Transport variants (name -> backend + env):
#   memcpy           MemcpyPeer
#   nccl             NCCL, default path selection
#   nccl_nop2p       NCCL with NCCL_P2P_DISABLE=1 (control: forces SHM/host path)
#   nixl_cudaipc     NIXL/UCX with UCX_TLS=self,tcp,sm,cuda_copy,cuda_ipc (P2P path)
#   nixl_hoststaged  NIXL/UCX with UCX_TLS=self,tcp,sm,cuda_copy (no cuda_ipc => host staged)
# Variants whose backend is not built simply fail to load and are skipped.
#
# cluster_bench worker mode is the native one per backend (not configurable):
#   memcpy -> --worker-mode thread    (raw pointers, one address space)
#   nccl, nixl -> --worker-mode process

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-$ROOT/build-cuda/benchmarks}"
MODELS="${MODELS:-models/falcon-h1-0.5b-base,models/mamba2-130m-hf}"
DEVICE="${DEVICE:-GPU}"
REPEATS="${REPEATS:-5}"
MAX_SEQS="${MAX_SEQS:-512,1024,2048,4096}"
VARIANTS="${VARIANTS:-memcpy,nccl,nccl_nop2p,nixl_cudaipc,nixl_hoststaged}"
STAGES="${STAGES:-env,paths,transport,scale,migrate,rebalance,logcheck,analyze}"
OUT="${OUT:-$ROOT/results/$(date +%Y%m%d_%H%M%S)}"
PROMPT_LEN="${PROMPT_LEN:-128}"
GEN_LEN="${GEN_LEN:-64}"
BG_SESSIONS="${BG_SESSIONS:-2}"
MIGRATIONS="${MIGRATIONS:-3}"
SESSIONS_REBALANCE="${SESSIONS_REBALANCE:-8}"
THRESHOLDS="${THRESHOLDS:-2,4}"
REBALANCE_MAX_SEQ="${REBALANCE_MAX_SEQ:-2048}"
PERF_CALL_GRAPH="${PERF_CALL_GRAPH:-dwarf}"
PERF_FREQ="${PERF_FREQ:-99}"
PERF_MAX_SEQ="${PERF_MAX_SEQ:-512}"
PERF_STAT="${PERF_STAT:-0}"

mkdir -p "$OUT"
cd "$ROOT"

have_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# /usr/bin/perf on WSL is often a stub that looks for a matching microsoft kernel
# tools package. Prefer an explicit PERF=, else a working linux-tools binary.
resolve_perf() {
  local cand
  if [[ -n "${PERF:-}" ]]; then
    echo "$PERF"
    return 0
  fi
  if command -v perf >/dev/null 2>&1 && perf version >/dev/null 2>&1; then
    echo "perf"
    return 0
  fi
  for cand in /usr/lib/linux-tools/*/perf; do
    if [[ -x "$cand" ]] && "$cand" version >/dev/null 2>&1; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

# variant -> "backend|ENV=VAL ENV2=VAL2"
variant_spec() {
  case "$1" in
    memcpy)          echo "memcpy|" ;;
    nccl)            echo "nccl|" ;;
    nccl_nop2p)      echo "nccl|NCCL_P2P_DISABLE=1" ;;
    # self+tcp are required by NIXL for loopback/wireup/notif active messages
    # (cuda_ipc / cuda_copy / sm have no AM-bcopy + peer-failure-handler support).
    # Bulk data still goes over cuda_ipc / cuda_copy.
    nixl_cudaipc)    echo "nixl|UCX_TLS=self,tcp,sm,cuda_copy,cuda_ipc" ;;
    nixl_hoststaged) echo "nixl|UCX_TLS=self,tcp,sm,cuda_copy" ;;
    *) echo "unknown|" ;;
  esac
}

# backend -> native cluster_bench worker mode (thread|process)
worker_mode_for() {
  case "$1" in
    memcpy) echo thread ;;
    nccl|nixl) echo process ;;
    *) echo thread ;;
  esac
}

# run_with_env "<ENV=VAL ...>" cmd args...
run_with_env() {
  local envs="$1"; shift
  if [[ -n "$envs" ]]; then
    env $envs "$@"
  else
    "$@"
  fi
}

log() { echo "[run_matrix] $*" >&2; }

if have_stage env; then
  log "recording environment"
  mkdir -p "$OUT/env"
  {
    date -Is
    uname -a
    nproc
    lscpu | grep -E 'Model name|Socket|Thread|Core' || true
    nvidia-smi || true
    nvidia-smi topo -m || true
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm --format=csv || true
    nvcc --version || true
    ldconfig -p | grep -E 'libnccl|libucp|libnixl' || true
    if command -v ucx_info >/dev/null; then
      ucx_info -v || true
      ucx_info -d | grep -E 'Transport|Device' | head -40 || true
    else
      echo "ucx_info: not installed"
    fi
  } > "$OUT/env/host.txt" 2>&1
  log "tip: for a hardware ceiling also run the CUDA sample p2pBandwidthLatencyTest and save it to $OUT/env/"
fi

IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"

if have_stage paths; then
  # Record which transport NCCL / UCX actually pick for each variant (P2P/NVLink vs SHM vs
  # host staged). Check these before interpreting any NCCL-vs-NIXL numbers.
  mkdir -p "$OUT/paths"
  for v in "${VARIANT_LIST[@]}"; do
    IFS='|' read -r backend envs <<< "$(variant_spec "$v")"
    [[ "$backend" == "memcpy" ]] && continue
    log "paths: $v"
    run_with_env "$envs NCCL_DEBUG=INFO UCX_LOG_LEVEL=info" "$BIN_DIR/transport_bench" \
      --device "$DEVICE" --backends "$backend" --min-mb 1 --max-mb 1 --iters 1 --warmup 1 \
      --modes uni --out-dir "$OUT/paths/$v" > "$OUT/paths/$v.log" 2>&1 || log "  (failed: $v)"
    grep -iE "NCCL INFO (Channel|Connected|Using|P2P|SHM|NET|Ring)|via P2P|via SHM|cuda_ipc|cuda_copy|ucp_|transport" \
      "$OUT/paths/$v.log" | head -60 > "$OUT/paths/$v.summary.txt" || true
  done
fi

if have_stage transport; then
  for v in "${VARIANT_LIST[@]}"; do
    IFS='|' read -r backend envs <<< "$(variant_spec "$v")"
    log "transport_bench: $v ($backend ${envs:-})"
    # cuda_p2p_raw baseline is added once, alongside memcpy.
    backends="$backend"
    [[ "$v" == "memcpy" ]] && backends="memcpy,cuda_p2p_raw"
    run_with_env "$envs" "$BIN_DIR/transport_bench" --device "$DEVICE" --backends "$backends" \
      --min-mb 1 --max-mb 256 --iters 20 --warmup 5 --modes uni,bidir \
      --out-dir "$OUT/transport/$v" 2>&1 | tee "$OUT/transport_$v.log" >/dev/null || log "  (failed: $v)"
  done
fi

if have_stage scale; then
  log "cluster_bench scale (worker-mode $(worker_mode_for memcpy))"
  "$BIN_DIR/cluster_bench" --scenario scale --device "$DEVICE" --model-dirs "$MODELS" \
    --backends memcpy --worker-mode "$(worker_mode_for memcpy)" --workers 1,2 --sessions 1,2,4,8,16 --prompt-len "$PROMPT_LEN" \
    --gen-len "$GEN_LEN" --repeats "$REPEATS" --out-dir "$OUT/scale" \
    2>&1 | tee "$OUT/scale.log" >/dev/null || log "  (scale failed)"
fi

if have_stage migrate; then
  for v in "${VARIANT_LIST[@]}"; do
    IFS='|' read -r backend envs <<< "$(variant_spec "$v")"
    wm="$(worker_mode_for "$backend")"
    log "cluster_bench migrate: $v (worker-mode $wm)"
    run_with_env "$envs" "$BIN_DIR/cluster_bench" --scenario migrate --device "$DEVICE" \
      --model-dirs "$MODELS" --backends "$backend" --worker-mode "$wm" --max-seq "$MAX_SEQS" --loads idle,busy \
      --bg-sessions "$BG_SESSIONS" --migrate-at 8 --migrate-every 8 --migrations "$MIGRATIONS" \
      --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN" --repeats "$REPEATS" \
      --out-dir "$OUT/migrate/$v" 2>&1 | tee "$OUT/migrate_$v.log" >/dev/null || log "  (failed: $v)"
  done
fi

if have_stage rebalance; then
  for v in "${VARIANT_LIST[@]}"; do
    IFS='|' read -r backend envs <<< "$(variant_spec "$v")"
    wm="$(worker_mode_for "$backend")"
    log "cluster_bench rebalance: $v (worker-mode $wm)"
    run_with_env "$envs" "$BIN_DIR/cluster_bench" --scenario rebalance --device "$DEVICE" \
      --model-dirs "$MODELS" --backends "$backend" --worker-mode "$wm" --max-seq "$REBALANCE_MAX_SEQ" \
      --sessions-rebalance "$SESSIONS_REBALANCE" --configs pinned_off,pinned_on,balanced_off \
      --thresholds "$THRESHOLDS" \
      --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN" --repeats "$REPEATS" \
      --out-dir "$OUT/rebalance/$v" 2>&1 | tee "$OUT/rebalance_$v.log" >/dev/null || log "  (failed: $v)"
  done
fi

if have_stage logcheck; then
  # Same config with logging off vs debug; large deltas mean logging distorts results.
  first_model="${MODELS%%,*}"
  for lvl in off debug; do
    log "logcheck: MAMBASERVE_LOG=$lvl"
    MAMBASERVE_LOG="$lvl" "$BIN_DIR/cluster_bench" --scenario migrate --device "$DEVICE" \
      --model-dirs "$first_model" --backends memcpy --worker-mode "$(worker_mode_for memcpy)" \
      --max-seq 2048 --loads busy --bg-sessions "$BG_SESSIONS" \
      --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN" --repeats "$REPEATS" \
      --out-dir "$OUT/logcheck/$lvl" > "$OUT/logcheck_$lvl.log" 2>&1 || log "  (failed: $lvl)"
  done
fi

if have_stage nsys; then
  # Needs a build with -DMAMBASERVE_WITH_NVTX=ON and nsys on PATH. Shows whether the
  # migrate overlaps decode on the same GPU and whether NCCL (SM kernels) vs
  # UCX/NIXL (copy engines) interfere differently.
  if command -v nsys >/dev/null; then
    for v in "${VARIANT_LIST[@]}"; do
      IFS='|' read -r backend envs <<< "$(variant_spec "$v")"
      wm="$(worker_mode_for "$backend")"
      log "nsys: $v (worker-mode $wm)"
      mkdir -p "$OUT/nsys"
      run_with_env "$envs" nsys profile -t cuda,nvtx,osrt -o "$OUT/nsys/$v" --force-overwrite true \
        "$BIN_DIR/cluster_bench" --scenario migrate --device "$DEVICE" --model-dirs "${MODELS%%,*}" \
        --backends "$backend" --worker-mode "$wm" --max-seq 2048 --loads busy --bg-sessions 2 --repeats 1 \
        --warmup-runs 1 --out-dir "$OUT/nsys/run_$v" > "$OUT/nsys_$v.log" 2>&1 || log "  (failed: $v)"
    done
  else
    log "nsys not found; skipping"
  fi
fi

if have_stage perf; then
  # Host CPU sample of a short migrate run. Complements nsys (GPU) and the telemetry
  # CSVs (phase durations): flame-ish stacks show where scheduler/worker burn cycles
  # or sit in poll/IPC. Distorts latency — keep as a diagnostic stage, not the full matrix.
  if PERF_BIN="$(resolve_perf)"; then
    log "perf: using $PERF_BIN ($($PERF_BIN version 2>/dev/null | head -1))"
    mkdir -p "$OUT/perf"
    for v in "${VARIANT_LIST[@]}"; do
      IFS='|' read -r backend envs <<< "$(variant_spec "$v")"
      wm="$(worker_mode_for "$backend")"
      # CPU path only has memcpy; skip GPU-only backends instead of failing loudly.
      if [[ "$DEVICE" == "CPU" && "$backend" != "memcpy" ]]; then
        log "perf: skip $v (CPU only supports memcpy)"
        continue
      fi
      log "perf: $v (worker-mode $wm, call-graph $PERF_CALL_GRAPH)"
      data="$OUT/perf/$v.data"
      # Child tasks inherit counters by default (omit -i/--no-inherit), so process-mode
      # workers are included without --follow-forks (not available on older perf).
      # cpu-clock works without PMU access (common on VMs/WSL). Cap frequency — dwarf
      # stacks at the default ~4 kHz can write multi-GB perf.data on CPU migrate runs.
      run_with_env "$envs" "$PERF_BIN" record -e cpu-clock -F "$PERF_FREQ" \
        --call-graph "$PERF_CALL_GRAPH" -o "$data" -- \
        "$BIN_DIR/cluster_bench" --scenario migrate --device "$DEVICE" \
        --model-dirs "${MODELS%%,*}" --backends "$backend" --worker-mode "$wm" \
        --max-seq "$PERF_MAX_SEQ" --loads busy --bg-sessions "$BG_SESSIONS" \
        --migrate-at 8 --migrate-every 8 --migrations "$MIGRATIONS" \
        --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN" --repeats 1 --warmup-runs 1 \
        --out-dir "$OUT/perf/run_$v" > "$OUT/perf_${v}.log" 2>&1 || {
        log "  (failed: $v)"; continue
      }
      "$PERF_BIN" report -i "$data" --stdio --no-children --percent-limit 0.5 \
        > "$OUT/perf/$v.report.txt" 2>>"$OUT/perf_${v}.log" || log "  (report failed: $v)"
      if command -v stackcollapse-perf.pl >/dev/null && command -v flamegraph.pl >/dev/null; then
        "$PERF_BIN" script -i "$data" 2>>"$OUT/perf_${v}.log" \
          | stackcollapse-perf.pl 2>/dev/null \
          | flamegraph.pl --title "cluster_bench migrate $v ($DEVICE)" \
          > "$OUT/perf/$v.svg" 2>>"$OUT/perf_${v}.log" || log "  (flamegraph failed: $v)"
      fi
      if [[ "$PERF_STAT" == "1" ]]; then
        "$PERF_BIN" stat -e cpu-clock,task-clock,context-switches,page-faults \
          -o "$OUT/perf/$v.stat.txt" -- \
          "$BIN_DIR/cluster_bench" --scenario migrate --device "$DEVICE" \
          --model-dirs "${MODELS%%,*}" --backends "$backend" --worker-mode "$wm" \
          --max-seq "$PERF_MAX_SEQ" --loads busy --bg-sessions "$BG_SESSIONS" \
          --migrate-at 8 --migrate-every 8 --migrations "$MIGRATIONS" \
          --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN" --repeats 1 --warmup-runs 0 \
          --out-dir "$OUT/perf/stat_$v" >>"$OUT/perf_${v}.log" 2>&1 || log "  (stat failed: $v)"
      fi
    done
  else
    log "perf not found (on WSL install linux-tools and/or set PERF=/usr/lib/linux-tools/*/perf); skipping"
  fi
fi

if have_stage analyze; then
  log "analyzing"
  python3 "$ROOT/scripts/analyze.py" --root "$OUT" --out "$OUT/report.md" || log "  (analyze failed)"
fi

log "done: $OUT"
