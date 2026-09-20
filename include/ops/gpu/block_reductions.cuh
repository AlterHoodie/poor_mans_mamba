#pragma once

#include <cuda_runtime.h>

struct Pair {
  float val;
  int idx;
};

inline __device__ float block_reduce_max(float val, float* smem, int tid) {
  smem[tid] = val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    __syncthreads();
  }
  return smem[0];
}

inline __device__ float block_reduce_sum(float val, float* smem, int tid) {
  smem[tid] = val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      smem[tid] = smem[tid] + smem[tid + stride];
    __syncthreads();
  }
  return smem[0];
}

inline __device__ Pair better(Pair a, Pair b) {
  if (a.val > b.val)
    return a;
  if (b.val > a.val)
    return b;
  return (a.idx < b.idx) ? a : b;
}

inline __device__ Pair block_reduce_best(Pair val, Pair* smem, int tid) {
  smem[tid] = val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      smem[tid] = better(smem[tid], smem[tid + stride]);
    __syncthreads();
  }
  return smem[0];
}

inline __device__ int offset_acb(int i, int k, int j, int c, int b) {
  return (i * c + k) * b + j;
}
