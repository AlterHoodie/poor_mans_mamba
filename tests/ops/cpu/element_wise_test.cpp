#include "ops/cpu/element_wise.h"

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

TEST(ElementWise, MulViaCrtp) {
    Tensor a = make_f32({4}, {1.f, 2.f, 3.f, 4.f});
    Tensor b = make_f32({4}, {2.f, 3.f, 4.f, 5.f});

    StatusOr<Tensor> result = mul(a, b);
    ASSERT_TRUE(result.ok());

    auto out = as_vec_f32(result.value());
    EXPECT_FLOAT_EQ(out(0), 2.f);
    EXPECT_FLOAT_EQ(out(3), 20.f);
}

TEST(ElementWise, AddViaCrtp) {
    Tensor a = make_f32({3}, {1.f, 2.f, 3.f});
    Tensor b = make_f32({3}, {4.f, 5.f, 6.f});
    Tensor out = make_f32({3}, {0.f, 0.f, 0.f});

    ASSERT_TRUE(add(a, b, out).ok());
    auto y = as_vec_f32(out);
    EXPECT_FLOAT_EQ(y(0), 5.f);
    EXPECT_FLOAT_EQ(y(2), 9.f);
}

TEST(ElementWise, AddInPlaceWhenOutIsA) {
    Tensor a = make_f32({2}, {1.f, 2.f});
    Tensor b = make_f32({2}, {3.f, 4.f});

    ASSERT_TRUE(add(a, b, a).ok());
    auto y = as_vec_f32(a);
    EXPECT_FLOAT_EQ(y(0), 4.f);
    EXPECT_FLOAT_EQ(y(1), 6.f);
}

TEST(ElementWise, SiluViaCrtp) {
    Tensor a = make_f32({2}, {0.f, 1.f});

    StatusOr<Tensor> result = silu(a);
    ASSERT_TRUE(result.ok());

    auto y = as_vec_f32(result.value());
    EXPECT_FLOAT_EQ(y(0), 0.f);
    EXPECT_NEAR(y(1), 1.f / (1.f + std::exp(-1.f)), 1e-6f);
}

TEST(ElementWise, MulOpObjectCallable) {
    Tensor a = make_f32({2}, {2.f, 3.f});
    Tensor b = make_f32({2}, {5.f, 7.f});
    Tensor out = make_f32({2}, {0.f, 0.f});

    ASSERT_TRUE(MulOp{}(a, b, out).ok());
    auto y = as_vec_f32(out);
    EXPECT_FLOAT_EQ(y(0), 10.f);
    EXPECT_FLOAT_EQ(y(1), 21.f);
}

TEST(ElementWise, ShapeMismatchFailsBeforeCompute) {
    Tensor a = make_f32({2}, {1.f, 2.f});
    Tensor b = make_f32({3}, {1.f, 2.f, 3.f});

    StatusOr<Tensor> result = mul(a, b);
    EXPECT_FALSE(result.ok());
}
