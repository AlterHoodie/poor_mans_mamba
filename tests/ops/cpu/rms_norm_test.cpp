#include "ops/cpu/rms_norm.h"
#include "ops/cpu/map.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

Tensor make_f32(std::vector<int64_t> shape, const std::vector<float>& values) {
    Tensor t;
    t.shape = std::move(shape);
    t.dtype = Dtype::F32;
    t.buffer.device = Device::CPU;
    t.buffer.bytes = values.size() * sizeof(float);
    t.buffer.data = std::malloc(t.buffer.bytes);
    std::memcpy(t.buffer.data, values.data(), t.buffer.bytes);
    return t;
}

}  // namespace

TEST(RmsNorm, NormalizesRank1Vector) {
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

TEST(RmsNorm, AppliesWeightScale) {
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

TEST(RmsNorm, BatchedRowsShareLastDim) {
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

TEST(RmsNorm, InPlaceOverwritesInput) {
    Tensor x = make_f32({2}, {3.f, 4.f});
    Tensor weight = make_f32({2}, {1.f, 1.f});

    ASSERT_TRUE(rms_norm_inplace(x, weight, 0.f).ok());

    const float inv_rms = 1.f / std::sqrt(12.5f);
    auto y = as_vec_f32(x);
    EXPECT_NEAR(y(0), 3.f * inv_rms, 1e-5f);
    EXPECT_NEAR(y(1), 4.f * inv_rms, 1e-5f);
}

TEST(RmsNorm, WeightDimMismatchFails) {
    Tensor x = make_f32({4}, {1.f, 2.f, 3.f, 4.f});
    Tensor weight = make_f32({2}, {1.f, 1.f});

    StatusOr<Tensor> result = rms_norm(x, weight, 1e-6f);
    EXPECT_FALSE(result.ok());
}
