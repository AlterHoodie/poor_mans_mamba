#pragma once

#include "core/device.h"
#include "ops/common/tensor_checks.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace gpu_test {

inline bool cuda_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

inline Tensor make_cpu_f32(std::vector<int64_t> shape, const std::vector<float>& values) {
  StatusOr<Tensor> t_or = allocate_f32_tensor(std::move(shape));
  EXPECT_TRUE(t_or.ok()) << t_or.status().message();
  Tensor t = std::move(t_or.value());
  EXPECT_EQ(t.buffer.bytes, values.size() * sizeof(float));
  std::memcpy(t.buffer.ptr, values.data(), values.size() * sizeof(float));
  return t;
}

inline Tensor make_cpu_f32_fill(std::vector<int64_t> shape, float fill) {
  StatusOr<Tensor> t_or = allocate_f32_tensor(std::move(shape));
  EXPECT_TRUE(t_or.ok()) << t_or.status().message();
  Tensor t = std::move(t_or.value());
  auto* p = static_cast<float*>(t.buffer.ptr);
  const size_t n = t.buffer.bytes / sizeof(float);
  for (size_t i = 0; i < n; ++i)
    p[i] = fill;
  return t;
}

inline Tensor make_gpu_f32(std::vector<int64_t> shape, const std::vector<float>& values) {
  StatusOr<Tensor> t_or = allocate_f32_tensor(std::move(shape));
  EXPECT_TRUE(t_or.ok()) << t_or.status().message();
  Tensor t = std::move(t_or.value());
  EXPECT_EQ(t.buffer.device, Device::GPU);
  EXPECT_EQ(t.buffer.bytes, values.size() * sizeof(float));
  cudaError_t err =
      cudaMemcpy(t.buffer.ptr, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  return t;
}

inline Tensor upload_f32(const Tensor& cpu) {
  EXPECT_EQ(cpu.buffer.device, Device::CPU);
  StatusOr<Tensor> t_or = allocate_f32_tensor(cpu.shape);
  EXPECT_TRUE(t_or.ok()) << t_or.status().message();
  Tensor t = std::move(t_or.value());
  EXPECT_EQ(t.buffer.device, Device::GPU);
  cudaError_t err =
      cudaMemcpy(t.buffer.ptr, cpu.buffer.ptr, cpu.buffer.bytes, cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  return t;
}

inline std::vector<float> download_f32(const Tensor& gpu) {
  EXPECT_EQ(gpu.buffer.device, Device::GPU);
  std::vector<float> host(gpu.buffer.bytes / sizeof(float));
  cudaError_t err =
      cudaMemcpy(host.data(), gpu.buffer.ptr, gpu.buffer.bytes, cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  return host;
}

inline std::vector<float> cpu_to_vec(const Tensor& cpu) {
  EXPECT_EQ(cpu.buffer.device, Device::CPU);
  const size_t n = cpu.buffer.bytes / sizeof(float);
  const float* p = static_cast<const float*>(cpu.buffer.ptr);
  return std::vector<float>(p, p + n);
}

inline void expect_close(const std::vector<float>& got, const std::vector<float>& want,
                         float atol = 1e-4f) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], atol) << "at index " << i;
  }
}

class GpuTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!cuda_available()) {
      GTEST_SKIP() << "No CUDA device available";
    }
    auto gpu_or = create_device_allocator(Device::GPU, 0);
    ASSERT_TRUE(gpu_or.ok()) << gpu_or.status().message();
    gpu_alloc_ = std::move(gpu_or.value());

    auto cpu_or = create_device_allocator(Device::CPU, 0);
    ASSERT_TRUE(cpu_or.ok()) << cpu_or.status().message();
    cpu_alloc_ = std::move(cpu_or.value());
  }

  std::unique_ptr<DeviceAllocator> gpu_alloc_;
  std::unique_ptr<DeviceAllocator> cpu_alloc_;
};

} // namespace gpu_test
