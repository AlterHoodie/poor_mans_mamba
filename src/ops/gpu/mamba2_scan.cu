#include "ops/gpu/mamba2_scan.cuh"

__device__ inline float softplus(float x) {
  if (x > 20.f)
    return x;
  if (x < -20.f)
    return expf(x);
  return log1pf(expf(x));
}

__global__ void mamba2_ssm_scan(const float* x, const float* B, const float* C, const float* dt,
                                const float* A_log, const float* D, const float* dt_bias,
                                float* state, float* y, int T, int H, int Dh, int N, int G) {
  // each thread computes at this level
  // head x head_dim
  int hd = blockIdx.x * blockDim.x + threadIdx.x;
  if (hd >= H * Dh) // H * Dh are the total number of channels assuming (H * Dh * channels)
    return;

  int h = hd / Dh; // head
  int d = hd % Dh; // head_dim
  int heads_per_group = H / G; // analogous to GQA, where BC pairs are shared among multiple state heads
  int g = h / heads_per_group; // which group g does h belong to

  const int intermediate = H * Dh;
  const int BC_width = G * N; // G * N and not H * N cause its shared

  for (int t = 0; t < T; ++t) { // for each timestep in a particular (h,d)
    const float dt_h = softplus(dt[t * H + h] + dt_bias[h]);  // input dependent change in timescale
    const float A_h = -expf(A_log[h]);
    const float dA = expf(dt_h * A_h); // by what factor should state change
    const float D_h = D[h]; // by what factor should skip input x to be added

    const float x_hd = x[t * intermediate + h * Dh + d];
    float y_hd = 0.f;

    const float* B_t = B + t * BC_width + g * N; // by what factor should input info change
    const float* C_t = C + t * BC_width + g * N; // to get back new token state from hidden state
    float* s_base = state + hd * N;

    for (int n = 0; n < N; ++n) {
      float s = s_base[n];
      s = s * dA + dt_h * B_t[n] * x_hd;
      s_base[n] = s;
      y_hd += s * C_t[n];
    }

    y[t * intermediate + h * Dh + d] = y_hd + D_h * x_hd;
  }
}
