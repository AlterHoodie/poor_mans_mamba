#pragma once

#include <cuda_runtime.h>

__global__ void add(const float* A, const float* B, float* C, int N);

__global__ void mul(const float* A, const float* B, float* C, int N);

__global__ void sub(const float* A, const float* B, float* C, int N);

__global__ void silu(const float* A, float* C, int N);
__global__ void silu(float* A, int N);

__global__ void scale(float* A, float s, int N);

// projected [T, I + conv_dim + H] -> gate[T,I], xBC[T,conv_dim], dt[T,H]
// conv_dim = I + 2*GN
__global__ void mixer_split_proj(const float* P, float* gate, float* xBC, float* dt, int T, int I,
                                 int GN, int H, float mz, float mx, float mB, float mC, float mdt);

// xBC_out [T, conv_dim] -> x[T,I], B[T,GN], C[T,GN]
__global__ void mixer_split_xbc(const float* xBC, float* x, float* B, float* C, int T, int I,
                                int GN);
