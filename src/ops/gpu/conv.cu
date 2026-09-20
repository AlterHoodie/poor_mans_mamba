#include "ops/gpu/conv.cuh"

__global__ void causal_conv1d(const float* x, int64_t T, int64_t conv_dim, int K, const float* weight, const float* bias, float* cache, float* y, bool apply_silu){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= conv_dim) return;

    float* crow = cache + c * (K - 1);
    const float* wt = weight + c * K;

    for(int t = 0; t < T; ++t){
        float xt = x[t * conv_dim + c];
        float acc = bias ? bias[c] : 0.f;

        for(int k = 0; k < K - 1; ++k) acc+=wt[k] * crow[k];
        acc+= wt[K-1] * xt;

        if(apply_silu) acc = acc / ( 1.f + expf(-acc));
        
        y[t * conv_dim + c] = acc;

        for (int k = 0; k < K - 2; ++k)
          crow[k] = crow[k + 1];
        if (K > 1)
          crow[K - 2] = xt;
    }
}