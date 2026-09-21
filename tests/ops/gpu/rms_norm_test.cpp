#include "ops/backend.h"
#include "ops/gpu/gpu_test_utils.h"

#include <gtest/gtest.h>

#include <vector>

using namespace gpu_test;

TEST_F(GpuTest, RmsNormRank1MatchesCpu) {
  const std::vector<float> x_h = {3.f, 4.f};
  const std::vector<float> w_h = {1.f, 1.f};
  const float eps = 0.f;

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor x = make_cpu_f32({2}, x_h);
    Tensor w = make_cpu_f32({2}, w_h);
    cpu_out = make_cpu_f32({2}, {0.f, 0.f});
    ASSERT_TRUE(cpu_ops().rms_norm(x, w, eps, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor x = make_gpu_f32({2}, x_h);
    Tensor w = make_gpu_f32({2}, w_h);
    Tensor out = make_gpu_f32({2}, {0.f, 0.f});
    ASSERT_TRUE(cuda_ops().rms_norm(x, w, eps, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-5f);
}

TEST_F(GpuTest, RmsNormBatchedMatchesCpu) {
  const std::vector<float> x_h = {3.f, 4.f, 6.f, 8.f};
  const std::vector<float> w_h = {1.f, 1.f};
  const float eps = 1e-6f;

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor x = make_cpu_f32({2, 2}, x_h);
    Tensor w = make_cpu_f32({2}, w_h);
    cpu_out = make_cpu_f32({2, 2}, {0.f, 0.f, 0.f, 0.f});
    ASSERT_TRUE(cpu_ops().rms_norm(x, w, eps, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor x = make_gpu_f32({2, 2}, x_h);
    Tensor w = make_gpu_f32({2}, w_h);
    Tensor out = make_gpu_f32({2, 2}, {0.f, 0.f, 0.f, 0.f});
    ASSERT_TRUE(cuda_ops().rms_norm(x, w, eps, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-5f);
}

TEST_F(GpuTest, RmsNormWeightScaleMatchesCpu) {
  const std::vector<float> x_h = {1.f, 0.f, 0.f};
  const std::vector<float> w_h = {2.f, 2.f, 2.f};

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor x = make_cpu_f32({3}, x_h);
    Tensor w = make_cpu_f32({3}, w_h);
    cpu_out = make_cpu_f32({3}, {0.f, 0.f, 0.f});
    ASSERT_TRUE(cpu_ops().rms_norm(x, w, 0.f, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor x = make_gpu_f32({3}, x_h);
    Tensor w = make_gpu_f32({3}, w_h);
    Tensor out = make_gpu_f32({3}, {0.f, 0.f, 0.f});
    ASSERT_TRUE(cuda_ops().rms_norm(x, w, 0.f, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-5f);
}
