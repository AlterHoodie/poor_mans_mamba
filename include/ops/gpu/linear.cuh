#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Row-major: y[M,N] = x[M,K] * W[N,K]^T  (same as CPU linear).
// W is stored as [N, K]. For a vector, use M = 1.
cudaError_t linear_f32(cublasHandle_t handle, const float* x, const float* W, float* y, int M,
                       int N, int K);
