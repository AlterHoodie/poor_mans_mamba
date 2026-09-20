#pragma once

#include <cuda_runtime.h>

__global__ void causal_conv1d(const float* x, int64_t T, int64_t conv_dim, int kernel,
                              const float* weight, const float* bias, float* cache, float* y,
                              bool apply_silu);