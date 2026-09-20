#pragma once

#include <cuda_runtime.h>

__global__ void mamba2_ssm_scan(const float* x, const float* B, const float* C, const float* dt,
                                const float* A_log, const float* D, const float* dt_bias,
                                float* state, float* y, int T, int H, int Dh, int N, int G);
