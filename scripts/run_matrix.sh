#!/usr/bin/env bash
# Runs the full benchmark matrix on a 2-GPU VM and writes everything under
# results/<timestamp>/ (override with OUT=...). Afterwards runs analyze.py.
#
#   scripts/run_matrix.sh                       # everything
#   STAGES=transport,migrate scripts/run_matrix.sh
#   DEVICE=CPU STAGES=scale REPEATS=1 scripts/run_matrix.sh   # smoke test, no GPU
#
# Stages: env paths transport scale migrate rebalance logcheck nsys analyze
#
# Knobs (environment variables):
#   BIN_DIR   directory with transport_bench / cluster_bench   (build-cuda/benchmarks)
#   MODELS    comma separated model dirs   (models/falcon-h1-0.5b-base,models/mamba2-130m-hf)
#   DEVICE    GPU | CPU                    (GPU)
#   REPEATS   measured repeats per config  (5)
#   MAX_SEQS  slot sizes for the migrate sweep (512,1024,2048,4096)
#   VARIANTS  transport variants to run (see below)
#   OUT       output root
#
# Transport variants (name -> backend + env):
#   memcpy           MemcpyPeer
#   nccl             NCCL, default path selection
#   nccl_nop2p       NCCL with NCCL_P2P_DISABLE=1 (control: forces SHM/host path)
#   nixl_cudaipc     NIXL/UCX pinned to UCX_TLS=cuda_ipc,cuda_copy,sm (P2P path)
#   nixl_hoststaged  NIXL/UCX with UCX_TLS=cuda_copy,sm (no cuda_ipc => host staged)
# Variants whose backend is not built simply fail to load and are skipped.

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

mkdir -p "$OUT"
cd "$ROOT"

have_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# variant -> "backend|ENV=VAL ENV2=VAL2"
variant_spec() {
  case "$1" in
    memcpy)          echo "memcpy|" ;;
    nccl)            echo "nccl|" ;;
    nccl_nop2p)      echo "nccl|NCCL_P2P_DISABLE=1" ;;
    nixl_cudaipc)    echo "nixl|UCX_TLS=cuda_ipc,cuda_copy,sm" ;;
    nixl_hoststaged) echo "nixl|UCX_TLS=cuda_copy,sm" ;;
    *) echo "unknown|" ;;
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
    ucx_info -v || true
    ucx_info -d | grep -E 'Transport|Device' | head -40 || true
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
  log "cluster_bench scale"
  "$BIN_DIR/cluster_bench" --scenario scale --device "$DEVICE" --model-dirs "$MODELS" \
    --backends memcpy --workers 1,2 --sessions 1,2,4,8,16 --prompt-len "$PROMPT_LEN" \
    --gen-len "$GEN_LEN" --repeats "$REPEATS" --out-dir "$OUT/scale" \
    2>&1 | tee "$OUT/scale.log" >/dev/null || log "  (scale failed)"
fi

if have_stage migrate; then
  for v in "${VARIANT_LIST[@]}"; do
    IFS='|' read -r backend envs <<< "$(variant_spec "$v")"
    log "cluster_bench migrate: $v"
    run_with_env "$envs" "$BIN_DIR/cluster_bench" --scenario migrate --device "$DEVICE" \
      --model-dirs "$MODELS" --backends "$backend" --max-seq "$MAX_SEQS" --loads idle,busy \
      --bg-sessions 2 --migrate-at 8 --migrate-every 8 --migrations 3 \
      --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN" --repeats "$REPEATS" \
      --out-dir "$OUT/migrate/$v" 2>&1 | tee "$OUT/migrate_$v.log" >/dev/null || log "  (failed: $v)"
  done
fi

if have_stage rebalance; then
  for v in "${VARIANT_LIST[@]}"; do
    IFS='|' read -r backend envs <<< "$(variant_spec "$v")"
    log "cluster_bench rebalance: $v"
    run_with_env "$envs" "$BIN_DIR/cluster_bench" --scenario rebalance --device "$DEVICE" \
      --model-dirs "$MODELS" --backends "$backend" --max-seq 2048 \
      --sessions-rebalance 8 --configs pinned_off,pinned_on,balanced_off --thresholds 2,4 \
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
      --model-dirs "$first_model" --backends memcpy --max-seq 2048 --loads busy --bg-sessions 2 \
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
      log "nsys: $v"
      mkdir -p "$OUT/nsys"
      run_with_env "$envs" nsys profile -t cuda,nvtx,osrt -o "$OUT/nsys/$v" --force-overwrite true \
        "$BIN_DIR/cluster_bench" --scenario migrate --device "$DEVICE" --model-dirs "${MODELS%%,*}" \
        --backends "$backend" --max-seq 2048 --loads busy --bg-sessions 2 --repeats 1 \
        --warmup-runs 1 --out-dir "$OUT/nsys/run_$v" > "$OUT/nsys_$v.log" 2>&1 || log "  (failed: $v)"
    done
  else
    log "nsys not found; skipping"
  fi
fi

if have_stage analyze; then
  log "analyzing"
  python3 "$ROOT/scripts/analyze.py" --root "$OUT" --out "$OUT/report.md" || log "  (analyze failed)"
fi

log "done: $OUT"
