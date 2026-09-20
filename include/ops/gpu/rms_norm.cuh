#pragma once

#include <cuda_runtime.h>

// <<<rows, 256>>> — one block per row of length N
__global__ void rms_norm(const float* A, float* C, const float* weight, int N, float eps);

// <<<a * b, 256>>> — reduce along c in layout (a, c, b)
__global__ void rms_norm_c(const float* X, float* C, const float* weight, float eps, int c, int b);
