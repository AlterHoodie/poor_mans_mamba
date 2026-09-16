#include "ops/cpu/linear.h"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "core/device.h"
#include "ops/cpu/map.h"

namespace {

class LinearTest : public ::testing::Test {
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

}  // namespace

TEST_F(LinearTest, VecTimesRank2) {
    // b is [N,K] = [2,2], a is [K]=[2] with a=[5,6]
    // y = b * a -> [1*5+2*6, 3*5+4*6] = [17, 39]
    Tensor a = make_f32({2}, {5.f, 6.f});
    Tensor b = make_f32({2, 2}, {1.f, 2.f, 3.f, 4.f});

    StatusOr<Tensor> result = linear(a, b);
    ASSERT_TRUE(result.ok());
    auto y = as_vec_f32(result.value());
    ASSERT_EQ(y.size(), 2);
    EXPECT_FLOAT_EQ(y(0), 17.f);
    EXPECT_FLOAT_EQ(y(1), 39.f);
}

TEST_F(LinearTest, BatchedLeadingDimsFlatten) {
    Tensor a = make_f32({2, 2}, {1.f, 2.f, 3.f, 4.f});
    Tensor b = make_f32({2, 2}, {1.f, 0.f, 0.f, 1.f});

    StatusOr<Tensor> result = linear(a, b);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().shape, (std::vector<int64_t>{2, 2}));

    auto y = as_mat_f32(result.value());
    EXPECT_FLOAT_EQ(y(0, 0), 1.f);
    EXPECT_FLOAT_EQ(y(0, 1), 2.f);
    EXPECT_FLOAT_EQ(y(1, 0), 3.f);
    EXPECT_FLOAT_EQ(y(1, 1), 4.f);
}

TEST_F(LinearTest, Rank3PreservesLeadingShape) {
    Tensor a = make_f32({2, 1, 2}, {1.f, 2.f, 3.f, 4.f});
    Tensor b = make_f32({1, 2}, {1.f, 1.f});

    StatusOr<Tensor> result = linear(a, b);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().shape, (std::vector<int64_t>{2, 1, 1}));

    auto y = as_mat_f32(result.value());
    EXPECT_EQ(y.rows(), 2);
    EXPECT_EQ(y.cols(), 1);
    EXPECT_FLOAT_EQ(y(0, 0), 3.f);
    EXPECT_FLOAT_EQ(y(1, 0), 7.f);
}

TEST_F(LinearTest, InnerDimMismatchFails) {
    Tensor a = make_f32({3}, {1.f, 2.f, 3.f});
    Tensor b = make_f32({2, 2}, {1.f, 2.f, 3.f, 4.f});

    StatusOr<Tensor> result = linear(a, b);
    EXPECT_FALSE(result.ok());
}

TEST_F(LinearTest, OutOverloadWritesIntoProvidedBuffer) {
    Tensor a = make_f32({2}, {1.f, 2.f});
    Tensor b = make_f32({2, 2}, {1.f, 0.f, 0.f, 1.f});
    Tensor out = make_f32({2}, {0.f, 0.f});

    ASSERT_TRUE(linear(a, b, out).ok());
    auto y = as_vec_f32(out);
    EXPECT_FLOAT_EQ(y(0), 1.f);
    EXPECT_FLOAT_EQ(y(1), 2.f);
}
