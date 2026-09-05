#include "runtime/cache/cache_pool.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

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

}  // namespace

TEST(CachePool, AcquireReleaseRoundTrip) {
    const Mamba2Config cfg = make_valid_config();
    CachePool pool(cfg, Device::CPU, /*device_id=*/0, /*n_slots=*/2);

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
    CachePool pool(cfg, Device::CPU, 0, 1);

    StatusOr<CacheHandle> handle_or = pool.acquire();
    ASSERT_TRUE(handle_or.ok());
    const CacheHandle& handle = handle_or.value();

    StatusOr<CacheLayout> layout_or = CacheLayout::from_config(cfg);
    ASSERT_TRUE(layout_or.ok());
    const CacheLayout& layout = layout_or.value();

    for (int layer = 0; layer < layout.num_layers; ++layer) {
        StatusOr<MambaLayerCacheView> view_or = pool.layer_view(handle, layer);
        ASSERT_TRUE(view_or.ok());
        const MambaLayerCacheView& view = view_or.value();

        EXPECT_NE(view.conv, nullptr);
        EXPECT_NE(view.ssm, nullptr);
        EXPECT_EQ(view.conv_bytes, layout.layers[static_cast<size_t>(layer)].conv_size);
        EXPECT_EQ(view.ssm_bytes, layout.layers[static_cast<size_t>(layer)].ssm_size);

        // Views for different layers should not alias.
        if (layer > 0) {
            StatusOr<MambaLayerCacheView> prev_or = pool.layer_view(handle, layer - 1);
            ASSERT_TRUE(prev_or.ok());
            EXPECT_NE(view.conv, prev_or.value().conv);
            EXPECT_NE(view.ssm, prev_or.value().ssm);
        }
    }
}

TEST(CachePool, ResetZerosStateButKeepsViewsUsable) {
    const Mamba2Config cfg = make_valid_config();
    CachePool pool(cfg, Device::CPU, 0, 1);

    StatusOr<CacheHandle> handle_or = pool.acquire();
    ASSERT_TRUE(handle_or.ok());
    CacheHandle handle = handle_or.value();

    StatusOr<MambaLayerCacheView> view_or = pool.layer_view(handle, 0);
    ASSERT_TRUE(view_or.ok());
    MambaLayerCacheView view = view_or.value();

    auto* conv = static_cast<std::uint8_t*>(view.conv);
    auto* ssm = static_cast<std::uint8_t*>(view.ssm);
    std::memset(conv, 0xAB, view.conv_bytes);
    std::memset(ssm, 0xCD, view.ssm_bytes);

    ASSERT_TRUE(pool.reset(handle).ok());
    EXPECT_TRUE(handle.valid());

    // Same views should still be valid and now zeroed.
    StatusOr<MambaLayerCacheView> after_or = pool.layer_view(handle, 0);
    ASSERT_TRUE(after_or.ok());
    EXPECT_EQ(after_or.value().conv, view.conv);
    EXPECT_EQ(after_or.value().ssm, view.ssm);

    for (size_t i = 0; i < view.conv_bytes; ++i) {
        EXPECT_EQ(conv[i], 0) << "conv byte " << i;
    }
    for (size_t i = 0; i < view.ssm_bytes; ++i) {
        EXPECT_EQ(ssm[i], 0) << "ssm byte " << i;
    }
}

TEST(CachePool, LayerViewRejectsBadHandleAndLayer) {
    const Mamba2Config cfg = make_valid_config();
    CachePool pool(cfg, Device::CPU, 0, 1);

    CacheHandle invalid;
    StatusOr<MambaLayerCacheView> bad_handle = pool.layer_view(invalid, 0);
    EXPECT_FALSE(bad_handle.ok());

    StatusOr<CacheHandle> handle_or = pool.acquire();
    ASSERT_TRUE(handle_or.ok());
    const CacheHandle& handle = handle_or.value();

    StatusOr<MambaLayerCacheView> bad_layer = pool.layer_view(handle, 99);
    ASSERT_FALSE(bad_layer.ok());
    EXPECT_EQ(bad_layer.status().code(), Code::kInvalidArgument);

    StatusOr<MambaLayerCacheView> negative_layer = pool.layer_view(handle, -1);
    ASSERT_FALSE(negative_layer.ok());
    EXPECT_EQ(negative_layer.status().code(), Code::kInvalidArgument);
}

TEST(CachePool, ReleaseThenLayerViewFails) {
    const Mamba2Config cfg = make_valid_config();
    CachePool pool(cfg, Device::CPU, 0, 1);

    StatusOr<CacheHandle> handle_or = pool.acquire();
    ASSERT_TRUE(handle_or.ok());
    CacheHandle handle = handle_or.value();
    const int old_id = handle.id();

    ASSERT_TRUE(pool.release(handle).ok());
    EXPECT_FALSE(handle.valid());

    // Using a stale copy of the old id should fail (slot not in use).
    CacheHandle stale(old_id);
    StatusOr<MambaLayerCacheView> view = pool.layer_view(stale, 0);
    ASSERT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), Code::kInvalidArgument);
}
