#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "io/config.h"
#include "model/mamba2_weights.h"
#include "runtime/cache/cache_layout.h"

// Depthwise causal conv over [T, conv_dim].
// cache is [conv_dim, kernel-1] left context (oldest .. newest), updated in place.
// weight is [conv_dim, kernel] (HF Conv1d squeeze).
Status causal_conv1d_f32(const float* x, int64_t T, int64_t conv_dim, int kernel,
                         const float* weight, const float* bias, float* cache, float* y,
                         bool apply_silu);

// Selective SSM scan over T steps. Updates state [heads, head_dim, state] in place.
// x: [T, intermediate], B/C: [T, n_groups*state], dt: [T, heads]
// y: [T, intermediate]
Status mamba2_ssm_scan_f32(const float* x, const float* B, const float* C, const float* dt,
                           const float* A_log, const float* D, const float* dt_bias, float* state,
                           float* y, int64_t T, const Mamba2Config& cfg);

// Gated RMSNorm used by Mamba2 mixer: y = rms(y * silu(gate)) * weight
Status rms_norm_gated_f32(Tensor& y, const Tensor& gate, const Tensor& weight, float eps);

// Full mixer block (pre-norm residual outside). Mutates cache views; writes mixer output into
// `out`.
Status mamba2_mixer_f32(const Tensor& normed_hidden, const Mamba2LayerWeights& w,
                        const Mamba2Config& cfg, LayerCacheView& cache, Tensor& out);
