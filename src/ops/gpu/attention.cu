#include "ops/gpu/attention.cuh"
#include "ops/gpu/block_reductions.cuh"

__global__ void apply_rope(float* x, int T, int heads, int Dh, int past_len, float rope_theta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n_rows = T * heads;
  if (idx >= n_rows)
    return;

  int t = idx / heads;
  int h = idx % heads;
  int half = Dh / 2;
  float* row = x + (t * heads + h) * Dh;
  float pos = static_cast<float>(past_len + t);

  for (int i = 0; i < half; ++i) {
    float inv_freq = 1.f / powf(rope_theta, static_cast<float>(2 * i) / static_cast<float>(Dh));
    float freq = pos * inv_freq;
    float c = cosf(freq);
    float s = sinf(freq);
    float x1 = row[i];
    float x2 = row[half + i];
    row[i] = x1 * c - x2 * s;
    row[half + i] = x2 * c + x1 * s;
  }
}

__global__ void kv_append(const float* k, const float* v, float* cache_k, float* cache_v, int T,
                          int kv_dim, int past_len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = T * kv_dim;
  if (i >= n)
    return;

  int t = i / kv_dim;
  int d = i % kv_dim;
  int dst = (past_len + t) * kv_dim + d;
  cache_k[dst] = k[i];
  cache_v[dst] = v[i];
}

// Launch: <<<T * Hq, 256, (past_len + T) * sizeof(float)>>>
// Dynamic shared holds softmax probs for this query (length S_len <= past_len + T).
__global__ void causal_attention_fused(const float* q, const float* cache_k, const float* cache_v,
                                       float* out, int T, int Hq, int Hkv, int Dh, int past_len,
                                       float scale) {
  __shared__ float red[256];
  extern __shared__ float probs[];

  int q_id = blockIdx.x;
  if (q_id >= T * Hq)
    return;

  int t = q_id / Hq; // token
  int hq = q_id % Hq; // query head
  int n_rep = Hq / Hkv; // GQA
  int hkv = hq / n_rep; // which kv group does this Q head belongs to 
  int tid = threadIdx.x;

  int q_pos = past_len + t; 
  int S_len = q_pos + 1;
  int kv_dim = Hkv * Dh;

  const float* q_hd = q + (t * Hq + hq) * Dh;
  float* o_hd = out + (t * Hq + hq) * Dh;

  // Pass 1: max score
  float local_max = -INFINITY;
  for (int s = tid; s < S_len; s += blockDim.x) {
    const float* k_hd = cache_k + s * kv_dim + hkv * Dh;
    float dot = 0.f;
    for (int d = 0; d < Dh; ++d)
      dot += q_hd[d] * k_hd[d];
    local_max = fmaxf(local_max, dot * scale);
  }
  float m = block_reduce_max(local_max, red, tid);

  // Pass 2: sum exp + stash probs
  float local_sum = 0.f;
  for (int s = tid; s < S_len; s += blockDim.x) {
    const float* k_hd = cache_k + s * kv_dim + hkv * Dh;
    float dot = 0.f;
    for (int d = 0; d < Dh; ++d)
      dot += q_hd[d] * k_hd[d];
    float e = expf(dot * scale - m);
    probs[s] = e;
    local_sum += e;
  }
  float S = block_reduce_sum(local_sum, red, tid);
  float inv_S = 1.f / S;

  for (int s = tid; s < S_len; s += blockDim.x)
    probs[s] *= inv_S;
  __syncthreads();

  // Pass 3: out = probs @ V  (GQA: use hkv's V)
  for (int d = 0; d < Dh; ++d) {
    float local = 0.f;
    for (int s = tid; s < S_len; s += blockDim.x)
      local += probs[s] * cache_v[s * kv_dim + hkv * Dh + d];
    float total = block_reduce_sum(local, red, tid);
    if (tid == 0)
      o_hd[d] = total;
  }
}
