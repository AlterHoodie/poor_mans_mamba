#include "ops/cpu/reductions.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "ops/cpu/map.h"

namespace {

Tensor make_f32(std::vector<int64_t> shape, const std::vector<float>& values) {
    Tensor t;
    t.shape = std::move(shape);
    t.dtype = Dtype::F32;
    t.buffer.device = Device::CPU;
    t.buffer.bytes = values.size() * sizeof(float);
    t.buffer.ptr = std::malloc(t.buffer.bytes);
    std::memcpy(t.buffer.ptr, values.data(), t.buffer.bytes);
    return t;
}

}  // namespace

TEST(Reductions, ArgmaxPicksLargestIndex) {
    Tensor logits = make_f32({4}, {0.1f, 3.0f, 2.0f, -1.0f});
    StatusOr<int32_t> idx = argmax(logits);
    ASSERT_TRUE(idx.ok()) << idx.status().message();
    EXPECT_EQ(idx.value(), 1);
}

TEST(Reductions, ArgmaxPicksFirstOnTie) {
    Tensor logits = make_f32({3}, {2.0f, 2.0f, 1.0f});
    StatusOr<int32_t> idx = argmax(logits);
    ASSERT_TRUE(idx.ok()) << idx.status().message();
    EXPECT_EQ(idx.value(), 0);
}

TEST(Reductions, ArgmaxSingleElement) {
    Tensor logits = make_f32({1}, {-5.0f});
    StatusOr<int32_t> idx = argmax(logits);
    ASSERT_TRUE(idx.ok()) << idx.status().message();
    EXPECT_EQ(idx.value(), 0);
}

TEST(Reductions, SoftmaxGreedyMatchesArgmax) {
    Tensor logits = make_f32({5}, {-2.f, 0.5f, 4.f, 1.f, 3.f});
    StatusOr<int32_t> soft = softmax(logits);
    StatusOr<int32_t> hard = argmax(logits);
    ASSERT_TRUE(soft.ok()) << soft.status().message();
    ASSERT_TRUE(hard.ok()) << hard.status().message();
    EXPECT_EQ(soft.value(), hard.value());
    EXPECT_EQ(soft.value(), 2);
}
