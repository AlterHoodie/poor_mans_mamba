#include "ops/cpu/reductions.h"
#include "ops/gpu/gpu_test_utils.h"
#include "ops/gpu/reductions.cuh"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace gpu_test;

namespace {

constexpr int kBlock = 256;

std::vector<float> host_softmax(const std::vector<float>& x) {
  float m = x[0];
  for (float v : x)
    m = std::max(m, v);
  std::vector<float> p(x.size());
  float sum = 0.f;
  for (size_t i = 0; i < x.size(); ++i) {
    p[i] = std::exp(x[i] - m);
    sum += p[i];
  }
  for (float& v : p)
    v /= sum;
  return p;
}

int host_argmax(const std::vector<float>& x) {
  int best = 0;
  for (size_t i = 1; i < x.size(); ++i) {
    if (x[i] > x[static_cast<size_t>(best)])
      best = static_cast<int>(i);
  }
  return best;
}

} // namespace

TEST_F(GpuTest, ArgmaxKernelMatchesCpu) {
  const std::vector<float> logits = {0.1f, 3.0f, 2.0f, -1.0f};

  int cpu_idx = -1;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor t = make_cpu_f32({4}, logits);
    StatusOr<int32_t> idx = argmax(t);
    ASSERT_TRUE(idx.ok());
    cpu_idx = idx.value();
  }

  float* d_logits = nullptr;
  int* d_idx = nullptr;
  ASSERT_EQ(cudaMalloc(&d_logits, logits.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_idx, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  ::argmax<<<1, kBlock>>>(d_logits, static_cast<int>(logits.size()), d_idx);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int gpu_idx = -1;
  ASSERT_EQ(cudaMemcpy(&gpu_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(gpu_idx, cpu_idx);
  EXPECT_EQ(gpu_idx, 1);

  cudaFree(d_logits);
  cudaFree(d_idx);
}

TEST_F(GpuTest, ArgmaxKernelTiePicksFirst) {
  const std::vector<float> logits = {2.0f, 2.0f, 1.0f};

  float* d_logits = nullptr;
  int* d_idx = nullptr;
  ASSERT_EQ(cudaMalloc(&d_logits, logits.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_idx, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  ::argmax<<<1, kBlock>>>(d_logits, static_cast<int>(logits.size()), d_idx);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int gpu_idx = -1;
  ASSERT_EQ(cudaMemcpy(&gpu_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(gpu_idx, 0);

  cudaFree(d_logits);
  cudaFree(d_idx);
}

TEST_F(GpuTest, SoftmaxKernelMatchesHost) {
  const std::vector<float> logits = {-2.f, 0.5f, 4.f, 1.f, 3.f};
  const std::vector<float> want = host_softmax(logits);

  float* d_in = nullptr;
  float* d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_in, logits.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, logits.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_in, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);

  ::softmax<<<1, kBlock>>>(d_in, d_out, static_cast<int>(logits.size()));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> got(logits.size());
  ASSERT_EQ(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  expect_close(got, want, 1e-5f);
  EXPECT_EQ(host_argmax(got), 2);

  cudaFree(d_in);
  cudaFree(d_out);
}

TEST_F(GpuTest, ArgmaxCKernelBatched) {
  // shape (a=2, c=3, b=2), layout (i,k,j) -> ((i*c+k)*b + j)
  // row0 (i=0): over c for each j
  // values chosen so argmax along c differs per (i,j)
  const int a = 2, c = 3, b = 2;
  std::vector<float> X(static_cast<size_t>(a * c * b));
  auto at = [&](int i, int k, int j) -> float& {
    return X[static_cast<size_t>((i * c + k) * b + j)];
  };
  // i=0,j=0 -> max at k=1; i=0,j=1 -> max at k=2
  // i=1,j=0 -> max at k=0; i=1,j=1 -> max at k=1
  at(0, 0, 0) = 1.f;
  at(0, 1, 0) = 5.f;
  at(0, 2, 0) = 2.f;
  at(0, 0, 1) = 1.f;
  at(0, 1, 1) = 2.f;
  at(0, 2, 1) = 9.f;
  at(1, 0, 0) = 7.f;
  at(1, 1, 0) = 3.f;
  at(1, 2, 0) = 1.f;
  at(1, 0, 1) = 0.f;
  at(1, 1, 1) = 4.f;
  at(1, 2, 1) = 2.f;

  const int out_n = a * b;
  std::vector<int> want = {1, 2, 0, 1};

  float* d_x = nullptr;
  int* d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_x, X.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, out_n * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_x, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);

  ::argmax_c<<<out_n, kBlock>>>(d_x, c, b, d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<int> got(static_cast<size_t>(out_n));
  ASSERT_EQ(cudaMemcpy(got.data(), d_out, out_n * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int i = 0; i < out_n; ++i)
    EXPECT_EQ(got[static_cast<size_t>(i)], want[static_cast<size_t>(i)]) << "at " << i;

  cudaFree(d_x);
  cudaFree(d_out);
}

TEST_F(GpuTest, SoftmaxCKernelBatched) {
  const int a = 1, c = 3, b = 2;
  std::vector<float> X(static_cast<size_t>(a * c * b));
  auto at = [&](int i, int k, int j) -> float& {
    return X[static_cast<size_t>((i * c + k) * b + j)];
  };
  at(0, 0, 0) = 1.f;
  at(0, 1, 0) = 2.f;
  at(0, 2, 0) = 3.f;
  at(0, 0, 1) = 0.f;
  at(0, 1, 1) = 0.f;
  at(0, 2, 1) = 0.f;

  std::vector<float> want = X;
  for (int j = 0; j < b; ++j) {
    std::vector<float> col(static_cast<size_t>(c));
    for (int k = 0; k < c; ++k)
      col[static_cast<size_t>(k)] = at(0, k, j);
    auto p = host_softmax(col);
    for (int k = 0; k < c; ++k)
      want[static_cast<size_t>((0 * c + k) * b + j)] = p[static_cast<size_t>(k)];
  }

  float* d_x = nullptr;
  float* d_p = nullptr;
  ASSERT_EQ(cudaMalloc(&d_x, X.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_p, X.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_x, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);

  ::softmax_c<<<a * b, kBlock>>>(d_x, d_p, c, b);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> got(X.size());
  ASSERT_EQ(cudaMemcpy(got.data(), d_p, got.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  expect_close(got, want, 1e-5f);

  cudaFree(d_x);
  cudaFree(d_p);
}
