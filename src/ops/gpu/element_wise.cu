#include "ops/gpu/element_wise.cuh"


__global__ void add(const float* A, const float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
        C[i] = A[i] + B[i];
}

__global__ void mul(const float* A, const float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
        C[i] = A[i] * B[i];
}

__global__ void sub(const float* A, const float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
        C[i] = A[i] - B[i];
}

__global__ void silu(const float* A, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        float x = A[i];
        C[i] = x / (1.0f + expf(-x));
    }
}

__global__ void silu(float* A, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        float x = A[i];
        A[i] = x / (1.0f + expf(-x));
    }
}

__global__ void scale(float* A, float s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        A[i] *= s;
}

__global__ void mixer_split_proj(const float* P, float* gate, float* xBC, float* dt, int T, int I,
                                 int GN, int H, float mz, float mx, float mB, float mC, float mdt) {
  const int conv_dim = I + 2 * GN;
  const int proj = I + conv_dim + H;
  const int n = T * proj;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  int t = idx / proj;
  int j = idx % proj;
  float v = P[idx];
  
  // scaling depending on which slice the j belongs to 
  if (j < I) { // if j belongs to gate
    gate[t * I + j] = v * mz;
  } else if (j < 2 * I) { // if j belongs to slice x
    xBC[t * conv_dim + (j - I)] = v * mx;
  } else if (j < 2 * I + GN) { // if j belongs to slice B 
    xBC[t * conv_dim + I + (j - 2 * I)] = v * mB;
  } else if (j < 2 * I + 2 * GN) { // if j belongs to slice C
    xBC[t * conv_dim + I + GN + (j - 2 * I - GN)] = v * mC;
  } else {
    dt[t * H + (j - I - conv_dim)] = v * mdt;
  }
}

__global__ void mixer_split_xbc(const float* xBC, float* x, float* B, float* C, int T, int I,
                                int GN) {
  const int conv_dim = I + 2 * GN;
  const int n = T * conv_dim;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  int t = idx / conv_dim;
  int j = idx % conv_dim;
  float v = xBC[idx];

  if (j < I)
    x[t * I + j] = v;
  else if (j < I + GN)
    B[t * GN + (j - I)] = v;
  else
    C[t * GN + (j - I - GN)] = v;
}