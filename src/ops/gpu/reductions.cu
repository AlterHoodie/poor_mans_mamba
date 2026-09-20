#include "ops/gpu/reductions.cuh"
#include "ops/gpu/block_reductions.cuh"

#include <cmath>

__global__ void argmax(const float* A, int N, int* out_idx) {
  __shared__ Pair smem[256];

  int tid = threadIdx.x;
  Pair best{-INFINITY, -1};

  for (int i = tid; i < N; i += blockDim.x)
    best = better(best, Pair{A[i], i});

  best = block_reduce_best(best, smem, tid);
  if (tid == 0)
    *out_idx = best.idx;
}

__global__ void argmax_c(const float* X, int c, int b, int* out) {
  __shared__ Pair smem[256];

  int out_id = blockIdx.x;
  int i = out_id / b;
  int j = out_id % b;
  int tid = threadIdx.x;

  Pair best{-INFINITY, -1};
  for (int k = tid; k < c; k += blockDim.x)
    best = better(best, Pair{X[offset_acb(i, k, j, c, b)], k});

  best = block_reduce_best(best, smem, tid);
  if (tid == 0)
    out[out_id] = best.idx;
}

__global__ void softmax(const float* A, float* P, int N) {
  __shared__ float smem[256];

  int tid = threadIdx.x;

  float local_max = -INFINITY;
  for (int i = tid; i < N; i += blockDim.x)
    local_max = fmaxf(local_max, A[i]);

  float m = block_reduce_max(local_max, smem, tid);

  float local_sum = 0.f;
  for (int i = tid; i < N; i += blockDim.x)
    local_sum += expf(A[i] - m);

  float S = block_reduce_sum(local_sum, smem, tid);
  float inv_S = 1.f / S;

  for (int i = tid; i < N; i += blockDim.x)
    P[i] = expf(A[i] - m) * inv_S;
}

__global__ void softmax_c(const float* X, float* P, int c, int b) {
  __shared__ float smem[256];

  int out_id = blockIdx.x;
  int i = out_id / b;
  int j = out_id % b;
  int tid = threadIdx.x;

  float local_max = -INFINITY;
  for (int k = tid; k < c; k += blockDim.x)
    local_max = fmaxf(local_max, X[offset_acb(i, k, j, c, b)]);

  float m = block_reduce_max(local_max, smem, tid);

  float local_sum = 0.f;
  for (int k = tid; k < c; k += blockDim.x)
    local_sum += expf(X[offset_acb(i, k, j, c, b)] - m);

  float S = block_reduce_sum(local_sum, smem, tid);
  float inv_S = 1.f / S;

  for (int k = tid; k < c; k += blockDim.x) {
    int idx = offset_acb(i, k, j, c, b);
    P[idx] = expf(X[idx] - m) * inv_S;
  }
}
