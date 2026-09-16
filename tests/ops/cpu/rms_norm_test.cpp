#include "core/device.h"
#include "ops/cpu/map.h"
#include "ops/cpu/rms_norm.h"
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace {

class RmsNormTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto alloc_or = create_device_allocator(Device::CPU, 0);
    ASSERT_TRUE(alloc_or.ok());
    alloc_ = std::move(alloc_or.value());
    scope_ = std::make_unique<AllocatorScope>(alloc_.get());
  }

  std::unique_ptr<DeviceAllocator> alloc_;
  std::unique_ptr<AllocatorScope> scope_;
};

Tensor make_f32(std::vector<int64_t> shape, const std::vector<float>& values) {
  StatusOr<Tensor> t_or = allocate_f32_tensor(std::move(shape));
  EXPECT_TRUE(t_or.ok()) << t_or.status().message();
  Tensor t = std::move(t_or.value());
  std::memcpy(t.buffer.ptr, values.data(), values.size() * sizeof(float));
  return t;
}

} // namespace

TEST_F(RmsNormTest, NormalizesRank1Vector) {
  Tensor x = make_f32({2}, {3.f, 4.f});
  Tensor weight = make_f32({2}, {1.f, 1.f});

  StatusOr<Tensor> result = rms_norm(x, weight, 0.f);
  ASSERT_TRUE(result.ok());

  const float mean_sq = (9.f + 16.f) / 2.f;
  const float inv_rms = 1.f / std::sqrt(mean_sq);
  auto y = as_vec_f32(result.value());
  EXPECT_NEAR(y(0), 3.f * inv_rms, 1e-5f);
  EXPECT_NEAR(y(1), 4.f * inv_rms, 1e-5f);
}

TEST_F(RmsNormTest, AppliesWeightScale) {
  Tensor x = make_f32({3}, {1.f, 0.f, 0.f});
  Tensor weight = make_f32({3}, {2.f, 2.f, 2.f});

  StatusOr<Tensor> result = rms_norm(x, weight, 0.f);
  ASSERT_TRUE(result.ok());

  const float inv_rms = 1.f / std::sqrt(1.f / 3.f);
  auto y = as_vec_f32(result.value());
  EXPECT_NEAR(y(0), 2.f * inv_rms, 1e-5f);
  EXPECT_FLOAT_EQ(y(1), 0.f);
  EXPECT_FLOAT_EQ(y(2), 0.f);
}

TEST_F(RmsNormTest, BatchedRowsShareLastDim) {
  Tensor x = make_f32({2, 2}, {3.f, 4.f, 6.f, 8.f});
  Tensor weight = make_f32({2}, {1.f, 1.f});

  StatusOr<Tensor> result = rms_norm(x, weight, 0.f);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().shape, (std::vector<int64_t>{2, 2}));

  auto y = as_mat_f32(result.value());
  const float inv_rms_0 = 1.f / std::sqrt((9.f + 16.f) / 2.f);
  const float inv_rms_1 = 1.f / std::sqrt((36.f + 64.f) / 2.f);
  EXPECT_NEAR(y(0, 0), 3.f * inv_rms_0, 1e-5f);
  EXPECT_NEAR(y(1, 1), 8.f * inv_rms_1, 1e-5f);
}

TEST_F(RmsNormTest, InPlaceOverwritesInput) {
  Tensor x = make_f32({2}, {3.f, 4.f});
  Tensor weight = make_f32({2}, {1.f, 1.f});

  ASSERT_TRUE(rms_norm_inplace(x, weight, 0.f).ok());

  const float inv_rms = 1.f / std::sqrt(12.5f);
  auto y = as_vec_f32(x);
  EXPECT_NEAR(y(0), 3.f * inv_rms, 1e-5f);
  EXPECT_NEAR(y(1), 4.f * inv_rms, 1e-5f);
}

TEST_F(RmsNormTest, WeightDimMismatchFails) {
  Tensor x = make_f32({4}, {1.f, 2.f, 3.f, 4.f});
  Tensor weight = make_f32({2}, {1.f, 1.f});

  StatusOr<Tensor> result = rms_norm(x, weight, 1e-6f);
  EXPECT_FALSE(result.ok());
}
