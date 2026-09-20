#include "ops/gpu/linear.cuh"

cudaError_t linear_f32(cublasHandle_t handle, const float* x, const float* W, float* y, int M,
                       int N, int K) {
  if (handle == nullptr || x == nullptr || W == nullptr || y == nullptr)
    return cudaErrorInvalidValue;
  if (M <= 0 || N <= 0 || K <= 0)
    return cudaErrorInvalidValue;

  const float alpha = 1.f;
  const float beta = 0.f;

  // Row-major C = A * B^T via column-major cuBLAS transpose trick:
  //   cublas sees op(W)^T-ish layout so memory of y is y_rm[M,N].
  cublasStatus_t st =
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K, x, K, &beta, y, N);

  if (st != CUBLAS_STATUS_SUCCESS)
    return cudaErrorUnknown;
  return cudaSuccess;
}
