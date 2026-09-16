#include "core/device.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>

namespace {

Mamba2Config make_valid_config() {
  Mamba2Config cfg{};
  cfg.num_hidden_layers = 2;
  cfg.hidden_size = 64;
  cfg.expand = 2;
  cfg.conv_kernel = 4;
  cfg.num_heads = 8;
  cfg.head_dim = 16;
  cfg.state_size = 16;
  cfg.n_groups = 1;
  return cfg;
}

struct PoolFixture {
  std::unique_ptr<DeviceAllocator> alloc;
  std::unique_ptr<CachePool> pool;

  static StatusOr<PoolFixture> create(const Mamba2Config& cfg, int num_slots) {
    PoolFixture f;
    ASSIGN_OR_RETURN(f.alloc, create_device_allocator(Device::CPU, 0));
    ASSIGN_OR_RETURN(f.pool,
                     create_cache_pool(cfg, f.alloc.get(), num_slots, create_mamba2_layout));
    return f;
  }
};

} // namespace

TEST(CachePool, AcquireReleaseRoundTrip) {
  const Mamba2Config cfg = make_valid_config();
  StatusOr<PoolFixture> fix_or = PoolFixture::create(cfg, /*num_slots=*/2);
  ASSERT_TRUE(fix_or.ok()) << fix_or.status().message();
  CachePool& pool = *fix_or.value().pool;

  StatusOr<CacheHandle> h0 = pool.acquire();
  StatusOr<CacheHandle> h1 = pool.acquire();
  ASSERT_TRUE(h0.ok());
  ASSERT_TRUE(h1.ok());
  EXPECT_NE(h0.value().id(), h1.value().id());

  StatusOr<CacheHandle> exhausted = pool.acquire();
  ASSERT_FALSE(exhausted.ok());
  EXPECT_EQ(exhausted.status().code(), Code::kOOM);

  CacheHandle handle0 = h0.value();
  const int released_id = handle0.id();
  ASSERT_TRUE(pool.release(handle0).ok());
  EXPECT_FALSE(handle0.valid());

  StatusOr<CacheHandle> reused = pool.acquire();
  ASSERT_TRUE(reused.ok());
  EXPECT_EQ(reused.value().id(), released_id);
}

TEST(CachePool, LayerViewsPointIntoSlotAndHaveExpectedSizes) {
  const Mamba2Config cfg = make_valid_config();
  StatusOr<PoolFixture> fix_or = PoolFixture::create(cfg, 1);
  ASSERT_TRUE(fix_or.ok()) << fix_or.status().message();
  CachePool& pool = *fix_or.value().pool;

  StatusOr<CacheHandle> handle_or = pool.acquire();
  ASSERT_TRUE(handle_or.ok());
  const CacheHandle& handle = handle_or.value();

  StatusOr<CacheLayout> layout_or = create_mamba2_layout(cfg);
  ASSERT_TRUE(layout_or.ok());
  const CacheLayout& layout = layout_or.value();

  for (int layer = 0; layer < layout.num_layers; ++layer) {
    StatusOr<LayerCacheView> view_or = pool.layer_view(handle, layer);
    ASSERT_TRUE(view_or.ok());
    const LayerCacheView& view = view_or.value();

    EXPECT_EQ(view.kind, LayerCacheKind::Mamba2);
    EXPECT_NE(view.conv.ptr, nullptr);
    EXPECT_NE(view.ssm.ptr, nullptr);
    EXPECT_EQ(view.conv.bytes, layout.layers[static_cast<size_t>(layer)].conv.bytes);
    EXPECT_EQ(view.ssm.bytes, layout.layers[static_cast<size_t>(layer)].ssm.bytes);

    if (layer > 0) {
      StatusOr<LayerCacheView> prev_or = pool.layer_view(handle, layer - 1);
      ASSERT_TRUE(prev_or.ok());
      EXPECT_NE(view.conv.ptr, prev_or.value().conv.ptr);
      EXPECT_NE(view.ssm.ptr, prev_or.value().ssm.ptr);
    }
  }
}

TEST(CachePool, ResetZerosStateButKeepsViewsUsable) {
  const Mamba2Config cfg = make_valid_config();
  StatusOr<PoolFixture> fix_or = PoolFixture::create(cfg, 1);
  ASSERT_TRUE(fix_or.ok()) << fix_or.status().message();
  CachePool& pool = *fix_or.value().pool;

  StatusOr<CacheHandle> handle_or = pool.acquire();
  ASSERT_TRUE(handle_or.ok());
  CacheHandle handle = handle_or.value();

  StatusOr<LayerCacheView> view_or = pool.layer_view(handle, 0);
  ASSERT_TRUE(view_or.ok());
  LayerCacheView view = view_or.value();

  auto* conv = static_cast<std::uint8_t*>(view.conv.ptr);
  auto* ssm = static_cast<std::uint8_t*>(view.ssm.ptr);
  std::memset(conv, 0xAB, view.conv.bytes);
  std::memset(ssm, 0xCD, view.ssm.bytes);

  ASSERT_TRUE(pool.reset(handle).ok());
  EXPECT_TRUE(handle.valid());

  StatusOr<LayerCacheView> after_or = pool.layer_view(handle, 0);
  ASSERT_TRUE(after_or.ok());
  EXPECT_EQ(after_or.value().conv.ptr, view.conv.ptr);
  EXPECT_EQ(after_or.value().ssm.ptr, view.ssm.ptr);

  for (size_t i = 0; i < view.conv.bytes; ++i) {
    EXPECT_EQ(conv[i], 0) << "conv byte " << i;
  }
  for (size_t i = 0; i < view.ssm.bytes; ++i) {
    EXPECT_EQ(ssm[i], 0) << "ssm byte " << i;
  }
}

TEST(CachePool, LayerViewRejectsBadHandleAndLayer) {
  const Mamba2Config cfg = make_valid_config();
  StatusOr<PoolFixture> fix_or = PoolFixture::create(cfg, 1);
  ASSERT_TRUE(fix_or.ok()) << fix_or.status().message();
  CachePool& pool = *fix_or.value().pool;

  CacheHandle invalid;
  StatusOr<LayerCacheView> bad_handle = pool.layer_view(invalid, 0);
  EXPECT_FALSE(bad_handle.ok());

  StatusOr<CacheHandle> handle_or = pool.acquire();
  ASSERT_TRUE(handle_or.ok());
  const CacheHandle& handle = handle_or.value();

  StatusOr<LayerCacheView> bad_layer = pool.layer_view(handle, 99);
  ASSERT_FALSE(bad_layer.ok());
  EXPECT_EQ(bad_layer.status().code(), Code::kInvalidArgument);

  StatusOr<LayerCacheView> negative_layer = pool.layer_view(handle, -1);
  ASSERT_FALSE(negative_layer.ok());
  EXPECT_EQ(negative_layer.status().code(), Code::kInvalidArgument);
}

TEST(CachePool, ReleaseThenLayerViewFails) {
  const Mamba2Config cfg = make_valid_config();
  StatusOr<PoolFixture> fix_or = PoolFixture::create(cfg, 1);
  ASSERT_TRUE(fix_or.ok()) << fix_or.status().message();
  CachePool& pool = *fix_or.value().pool;

  StatusOr<CacheHandle> handle_or = pool.acquire();
  ASSERT_TRUE(handle_or.ok());
  CacheHandle handle = handle_or.value();
  const int old_id = handle.id();

  ASSERT_TRUE(pool.release(handle).ok());
  EXPECT_FALSE(handle.valid());

  CacheHandle stale(old_id);
  StatusOr<LayerCacheView> view = pool.layer_view(stale, 0);
  ASSERT_FALSE(view.ok());
  EXPECT_EQ(view.status().code(), Code::kInvalidArgument);
}
