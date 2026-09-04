#include "core/tensor.h"
#include "ops/cpu/map.h"

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

TEST(MapF32, VecFlattensAnyRank) {
    Tensor t = make_f32({2, 3}, {1, 2, 3, 4, 5, 6});
    auto v = as_vec_f32(t);
    ASSERT_EQ(v.size(), 6);
    EXPECT_FLOAT_EQ(v(0), 1.f);
    EXPECT_FLOAT_EQ(v(5), 6.f);
}

TEST(MapF32, MatIsRowMajorLeadingByLast) {
    Tensor t = make_f32({2, 3}, {1, 2, 3, 4, 5, 6});
    auto m = as_mat_f32(t);
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_FLOAT_EQ(m(0, 2), 3.f);
    EXPECT_FLOAT_EQ(m(1, 0), 4.f);
    m(0, 0) = 9.f;
    EXPECT_FLOAT_EQ(static_cast<const float*>(t.buffer.data)[0], 9.f);
}

TEST(MapF32, ConstOverloadMapsWithoutCopy) {
    Tensor t = make_f32({4}, {1, 2, 3, 4});
    const Tensor& ct = t;
    auto v = as_vec_f32(ct);
    ASSERT_EQ(v.size(), 4);
    EXPECT_FLOAT_EQ(v(2), 3.f);
}

TEST(MapF32, Rank3FlattensToRowsTimesLastDim) {
    Tensor t = make_f32({2, 3, 4}, std::vector<float>(24, 0.f));
    auto m = as_mat_f32(t);
    EXPECT_EQ(m.rows(), 6);
    EXPECT_EQ(m.cols(), 4);
}
