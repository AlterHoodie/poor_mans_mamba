#include "ops/backend.h"
#include "ops/gpu/gpu_test_utils.h"

#include <gtest/gtest.h>

#include <vector>

using namespace gpu_test;

TEST_F(GpuTest, AddMatchesCpu) {
  const std::vector<float> a_h = {1.f, 2.f, 3.f};
  const std::vector<float> b_h = {4.f, 5.f, 6.f};

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor a = make_cpu_f32({3}, a_h);
    Tensor b = make_cpu_f32({3}, b_h);
    cpu_out = make_cpu_f32({3}, {0.f, 0.f, 0.f});
    ASSERT_TRUE(cpu_ops().add(a, b, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor a = make_gpu_f32({3}, a_h);
    Tensor b = make_gpu_f32({3}, b_h);
    Tensor out = make_gpu_f32({3}, {0.f, 0.f, 0.f});
    ASSERT_TRUE(cuda_ops().add(a, b, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-5f);
}

TEST_F(GpuTest, ScaleMatchesCpu) {
  const std::vector<float> x_h = {1.f, -2.f, 3.5f};
  const float s = 0.5f;

  Tensor cpu_x;
  {
    AllocatorScope scope(cpu_alloc_.get());
    cpu_x = make_cpu_f32({3}, x_h);
    ASSERT_TRUE(cpu_ops().scale(cpu_x, s).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor x = make_gpu_f32({3}, x_h);
    ASSERT_TRUE(cuda_ops().scale(x, s).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(x);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_x), 1e-5f);
}

TEST_F(GpuTest, AddShapeMismatchFails) {
  AllocatorScope scope(gpu_alloc_.get());
  Tensor a = make_gpu_f32({2}, {1.f, 2.f});
  Tensor b = make_gpu_f32({3}, {1.f, 2.f, 3.f});
  Tensor out = make_gpu_f32({2}, {0.f, 0.f});
  EXPECT_FALSE(cuda_ops().add(a, b, out).ok());
}
