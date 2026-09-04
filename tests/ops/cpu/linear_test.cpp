#include "ops/cpu/linear.h"

#include <gtest/gtest.h>

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

TEST(Linear, VecTimesRank2) {
    // a: [2], b: [2,2] with b = [[1,2],[3,4]] -> [2,2]*[2] = [10, 14]... wait
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

TEST(Linear, BatchedLeadingDimsFlatten) {
    // a: [2,2] rows [[1,2],[3,4]], b: [2,2] identity-ish [[1,0],[0,1]] -> same
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

TEST(Linear, Rank3PreservesLeadingShape) {
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

TEST(Linear, InnerDimMismatchFails) {
    Tensor a = make_f32({3}, {1.f, 2.f, 3.f});
    Tensor b = make_f32({2, 2}, {1.f, 2.f, 3.f, 4.f});

    StatusOr<Tensor> result = linear(a, b);
    EXPECT_FALSE(result.ok());
}

TEST(Linear, OutOverloadWritesIntoProvidedBuffer) {
    Tensor a = make_f32({2}, {1.f, 2.f});
    Tensor b = make_f32({2, 2}, {1.f, 0.f, 0.f, 1.f});
    Tensor out = make_f32({2}, {0.f, 0.f});

    ASSERT_TRUE(linear(a, b, out).ok());
    auto y = as_vec_f32(out);
    EXPECT_FLOAT_EQ(y(0), 1.f);
    EXPECT_FLOAT_EQ(y(1), 2.f);
}
