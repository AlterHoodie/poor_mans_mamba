#include "ops/backend.h"
#include "ops/gpu/gpu_test_utils.h"

#include <gtest/gtest.h>

#include <vector>

using namespace gpu_test;

TEST_F(GpuTest, LinearVecMatchesCpu) {
  // W [N,K]=[2,2], x=[5,6] -> y = [17, 39]
  const std::vector<float> x_h = {5.f, 6.f};
  const std::vector<float> W_h = {1.f, 2.f, 3.f, 4.f};

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor x = make_cpu_f32({2}, x_h);
    Tensor W = make_cpu_f32({2, 2}, W_h);
    cpu_out = make_cpu_f32({2}, {0.f, 0.f});
    ASSERT_TRUE(cpu_ops().linear(x, W, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor x = make_gpu_f32({2}, x_h);
    Tensor W = make_gpu_f32({2, 2}, W_h);
    Tensor out = make_gpu_f32({2}, {0.f, 0.f});
    ASSERT_TRUE(cuda_ops().linear(x, W, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-4f);
}

TEST_F(GpuTest, LinearBatchedMatchesCpu) {
  const std::vector<float> x_h = {1.f, 2.f, 3.f, 4.f};
  const std::vector<float> W_h = {1.f, 0.f, 0.f, 1.f};

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor x = make_cpu_f32({2, 2}, x_h);
    Tensor W = make_cpu_f32({2, 2}, W_h);
    cpu_out = make_cpu_f32({2, 2}, {0.f, 0.f, 0.f, 0.f});
    ASSERT_TRUE(cpu_ops().linear(x, W, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor x = make_gpu_f32({2, 2}, x_h);
    Tensor W = make_gpu_f32({2, 2}, W_h);
    Tensor out = make_gpu_f32({2, 2}, {0.f, 0.f, 0.f, 0.f});
    ASSERT_TRUE(cuda_ops().linear(x, W, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-4f);
}

TEST_F(GpuTest, LinearInnerDimMismatchFails) {
  AllocatorScope scope(gpu_alloc_.get());
  Tensor x = make_gpu_f32({3}, {1.f, 2.f, 3.f});
  Tensor W = make_gpu_f32({2, 2}, {1.f, 2.f, 3.f, 4.f});
  Tensor out = make_gpu_f32({2}, {0.f, 0.f});
  EXPECT_FALSE(cuda_ops().linear(x, W, out).ok());
}
