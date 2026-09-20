#pragma once

#include <cuda_runtime.h>

__global__ void argmax(const float* A, int N, int* out_idx);
__global__ void argmax_c(const float* X, int c, int b, int* out);

// Softmax over a flat vector: P[i] = exp(A[i]-max) / sum
__global__ void softmax(const float* A, float* P, int N);
// Softmax along dim c for shape (a, c, b); P has same layout as X
__global__ void softmax_c(const float* X, float* P, int c, int b);
