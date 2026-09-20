#pragma once

#include <cuda_runtime.h>

// In-place RoPE on x layout [T, heads, Dh] (contiguous). Same rotate_half as CPU.
__global__ void apply_rope(float* x, int T, int heads, int Dh, int past_len, float rope_theta);

// Append K/V rows into cache layout [max_S, kv_dim] at rows [past_len, past_len+T).
__global__ void kv_append(const float* k, const float* v, float* cache_k, float* cache_v, int T,
                          int kv_dim, int past_len);

// Fused causal attention: scores + softmax + V mix.
// q: [T, Hq, Dh], cache_k/v: [max_S, Hkv*Dh], out: [T, Hq, Dh]
// One block per (t, hq). Causal mask: keys s in [0, past_len+t].
// Launch shared mem: (past_len + T) * sizeof(float) for probs.
__global__ void causal_attention_fused(const float* q, const float* cache_k, const float* cache_v,
                                       float* out, int T, int Hq, int Hkv, int Dh, int past_len,
                                       float scale);
