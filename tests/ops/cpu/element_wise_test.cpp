#include "core/device.h"
#include "ops/cpu/element_wise.h"
#include "ops/cpu/map.h"
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace {

class ElementWiseTest : public ::testing::Test {
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

TEST_F(ElementWiseTest, MulViaCrtp) {
  Tensor a = make_f32({4}, {1.f, 2.f, 3.f, 4.f});
  Tensor b = make_f32({4}, {2.f, 3.f, 4.f, 5.f});

  StatusOr<Tensor> result = mul(a, b);
  ASSERT_TRUE(result.ok());

  auto out = as_vec_f32(result.value());
  EXPECT_FLOAT_EQ(out(0), 2.f);
  EXPECT_FLOAT_EQ(out(3), 20.f);
}

TEST_F(ElementWiseTest, AddViaCrtp) {
  Tensor a = make_f32({3}, {1.f, 2.f, 3.f});
  Tensor b = make_f32({3}, {4.f, 5.f, 6.f});
  Tensor out = make_f32({3}, {0.f, 0.f, 0.f});

  ASSERT_TRUE(add(a, b, out).ok());
  auto y = as_vec_f32(out);
  EXPECT_FLOAT_EQ(y(0), 5.f);
  EXPECT_FLOAT_EQ(y(2), 9.f);
}

TEST_F(ElementWiseTest, AddInPlaceWhenOutIsA) {
  Tensor a = make_f32({2}, {1.f, 2.f});
  Tensor b = make_f32({2}, {3.f, 4.f});

  ASSERT_TRUE(add(a, b, a).ok());
  auto y = as_vec_f32(a);
  EXPECT_FLOAT_EQ(y(0), 4.f);
  EXPECT_FLOAT_EQ(y(1), 6.f);
}

TEST_F(ElementWiseTest, SiluViaCrtp) {
  Tensor a = make_f32({2}, {0.f, 1.f});

  StatusOr<Tensor> result = silu(a);
  ASSERT_TRUE(result.ok());

  auto y = as_vec_f32(result.value());
  EXPECT_FLOAT_EQ(y(0), 0.f);
  EXPECT_NEAR(y(1), 1.f / (1.f + std::exp(-1.f)), 1e-6f);
}

TEST_F(ElementWiseTest, MulOpObjectCallable) {
  Tensor a = make_f32({2}, {2.f, 3.f});
  Tensor b = make_f32({2}, {5.f, 7.f});
  Tensor out = make_f32({2}, {0.f, 0.f});

  ASSERT_TRUE(MulOp{}(a, b, out).ok());
  auto y = as_vec_f32(out);
  EXPECT_FLOAT_EQ(y(0), 10.f);
  EXPECT_FLOAT_EQ(y(1), 21.f);
}

TEST_F(ElementWiseTest, ShapeMismatchFailsBeforeCompute) {
  Tensor a = make_f32({2}, {1.f, 2.f});
  Tensor b = make_f32({3}, {1.f, 2.f, 3.f});

  StatusOr<Tensor> result = mul(a, b);
  EXPECT_FALSE(result.ok());
}
