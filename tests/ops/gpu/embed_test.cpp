#include "ops/backend.h"
#include "ops/gpu/gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace gpu_test;

TEST_F(GpuTest, EmbeddingLookupMatchesCpu) {
  // table [4, 3]
  const std::vector<float> table_h = {
      0.1f, 0.2f, 0.3f, // tok 0
      1.0f, 1.1f, 1.2f, // tok 1
      2.0f, 2.1f, 2.2f, // tok 2
      3.0f, 3.1f, 3.2f, // tok 3
  };
  const int32_t tokens[] = {1, 3, 0};
  const float scale = 2.f;

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor table = make_cpu_f32({4, 3}, table_h);
    cpu_out = make_cpu_f32({3, 3}, std::vector<float>(9, 0.f));
    ASSERT_TRUE(cpu_ops().embedding_lookup(table, tokens, 3, scale, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor table = make_gpu_f32({4, 3}, table_h);
    Tensor out = make_gpu_f32({3, 3}, std::vector<float>(9, 0.f));
    ASSERT_TRUE(cuda_ops().embedding_lookup(table, tokens, 3, scale, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-5f);
}

TEST_F(GpuTest, TakeLastRowMatchesCpu) {
  const std::vector<float> hidden_h = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}; // [3,2]

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor hidden = make_cpu_f32({3, 2}, hidden_h);
    cpu_out = make_cpu_f32({2}, {0.f, 0.f});
    ASSERT_TRUE(cpu_ops().take_last_row(hidden, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor hidden = make_gpu_f32({3, 2}, hidden_h);
    Tensor out = make_gpu_f32({2}, {0.f, 0.f});
    ASSERT_TRUE(cuda_ops().take_last_row(hidden, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-5f);
}

TEST_F(GpuTest, TakeLastRowRank1MatchesCpu) {
  const std::vector<float> hidden_h = {7.f, 8.f, 9.f};

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor hidden = make_cpu_f32({3}, hidden_h);
    cpu_out = make_cpu_f32({3}, {0.f, 0.f, 0.f});
    ASSERT_TRUE(cpu_ops().take_last_row(hidden, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor hidden = make_gpu_f32({3}, hidden_h);
    Tensor out = make_gpu_f32({3}, {0.f, 0.f, 0.f});
    ASSERT_TRUE(cuda_ops().take_last_row(hidden, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-5f);
}
