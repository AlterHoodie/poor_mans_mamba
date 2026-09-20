#include "ops/gpu/rms_norm.cuh"
#include "ops/gpu/block_reductions.cuh"

// One block per row. A/C layout [rows, N], weight [N].
// Launch: <<<rows, 256>>>
__global__ void rms_norm(const float* A, float* C, const float* weight, int N, float eps) {
  __shared__ float smem[256];

  int row = blockIdx.x;
  int tid = threadIdx.x;
  const float* a_row = A + static_cast<size_t>(row) * static_cast<size_t>(N);
  float* c_row = C + static_cast<size_t>(row) * static_cast<size_t>(N);

  float local_sum = 0.f;
  for (int i = tid; i < N; i += blockDim.x) {
    float x = a_row[i];
    local_sum += x * x;
  }
  local_sum = block_reduce_sum(local_sum, smem, tid);

  float ms = local_sum / static_cast<float>(N);
  float inv_rms = 1.f / sqrtf(ms + eps);

  for (int i = tid; i < N; i += blockDim.x)
    c_row[i] = a_row[i] * inv_rms * weight[i];
}

// One block per (i, j) with X layout (a, c, b). Reduce along c.
// Launch: <<<a * b, 256>>>
__global__ void rms_norm_c(const float* X, float* C, const float* weight, float eps, int c,
                           int b) {
  __shared__ float smem[256];

  int out_id = blockIdx.x;
  int i = out_id / b;
  int j = out_id % b;
  int tid = threadIdx.x;

  float local_sum = 0.f;
  for (int k = tid; k < c; k += blockDim.x) {
    float x = X[offset_acb(i, k, j, c, b)];
    local_sum += x * x;
  }

  float S = block_reduce_sum(local_sum, smem, tid);

  float ms = S / static_cast<float>(c);
  float inv_rms = 1.f / sqrtf(ms + eps);

  for (int k = tid; k < c; k += blockDim.x) {
    int idx = offset_acb(i, k, j, c, b);
    C[idx] = X[idx] * inv_rms * weight[k];
  }
}
