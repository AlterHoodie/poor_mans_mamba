#include "io/config.h"
#include "ops/backend.h"
#include "ops/gpu/gpu_test_utils.h"

#include <gtest/gtest.h>

#include <vector>

using namespace gpu_test;

namespace {

std::vector<float> fill_vec(size_t n, float v) { return std::vector<float>(n, v); }

} // namespace

TEST_F(GpuTest, MlpMatchesCpu) {
  const int64_t D = 4;
  const int64_t F = 8;
  const int64_t T = 2;
  MlpConfig mlp{};
  mlp.intermediate_size = static_cast<int>(F);

  ScaleConfig scales{};
  scales.mlp = {1.f, 1.f};

  const std::vector<float> hidden_h = fill_vec(static_cast<size_t>(T * D), 0.1f);
  const std::vector<float> up_h = fill_vec(static_cast<size_t>(F * D), 0.01f);
  const std::vector<float> gate_h = fill_vec(static_cast<size_t>(F * D), 0.02f);
  const std::vector<float> down_h = fill_vec(static_cast<size_t>(D * F), 0.03f);

  Tensor cpu_out;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor hidden = make_cpu_f32({T, D}, hidden_h);
    Tensor up = make_cpu_f32({F, D}, up_h);
    Tensor gate = make_cpu_f32({F, D}, gate_h);
    Tensor down = make_cpu_f32({D, F}, down_h);
    cpu_out = make_cpu_f32({T, D}, fill_vec(static_cast<size_t>(T * D), 0.f));
    MlpWeights w{.up_proj = up, .gate_proj = gate, .down_proj = down};
    ASSERT_TRUE(cpu_ops().mlp_f32(hidden, w, mlp, &scales, cpu_out).ok());
  }

  std::vector<float> gpu_h;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor hidden = make_gpu_f32({T, D}, hidden_h);
    Tensor up = make_gpu_f32({F, D}, up_h);
    Tensor gate = make_gpu_f32({F, D}, gate_h);
    Tensor down = make_gpu_f32({D, F}, down_h);
    Tensor out = make_gpu_f32({T, D}, fill_vec(static_cast<size_t>(T * D), 0.f));
    MlpWeights w{.up_proj = up, .gate_proj = gate, .down_proj = down};
    ASSERT_TRUE(cuda_ops().mlp_f32(hidden, w, mlp, &scales, out).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_h = download_f32(out);
  }

  expect_close(gpu_h, cpu_to_vec(cpu_out), 1e-3f);
}
