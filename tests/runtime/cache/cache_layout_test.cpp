#include "runtime/cache/cache_layout.h"

#include <gtest/gtest.h>

#include <cstddef>

namespace {

Mamba2Config make_valid_config() {
    Mamba2Config cfg{};
    cfg.num_hidden_layers = 2;
    cfg.hidden_size = 64;
    cfg.expand = 2;
    cfg.conv_kernel = 4;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.state_size = 16;
    return cfg;
}

}  // namespace

TEST(CacheLayout, FromConfigPacksLayersContiguously) {
    const Mamba2Config cfg = make_valid_config();
    StatusOr<CacheLayout> result = CacheLayout::from_config(cfg);
    ASSERT_TRUE(result.ok());

    const CacheLayout& layout = result.value();
    EXPECT_EQ(layout.num_layers, 2);
    ASSERT_EQ(layout.layers.size(), 2u);

    const size_t intermediate = static_cast<size_t>(cfg.hidden_size) * static_cast<size_t>(cfg.expand);
    const size_t conv_bytes = intermediate * static_cast<size_t>(cfg.conv_kernel - 1) * sizeof(float);
    const size_t ssm_bytes = static_cast<size_t>(cfg.num_heads) * static_cast<size_t>(cfg.head_dim) *
                             static_cast<size_t>(cfg.state_size) * sizeof(float);
    const size_t per_layer = conv_bytes + ssm_bytes;

    EXPECT_EQ(layout.slot_bytes, per_layer * 2);

    size_t expected_offset = 0;
    for (int layer = 0; layer < layout.num_layers; ++layer) {
        const MambaLayerLayout& L = layout.layers[static_cast<size_t>(layer)];
        EXPECT_EQ(L.layer_idx, layer);
        EXPECT_EQ(L.conv_offset, expected_offset);
        EXPECT_EQ(L.conv_size, conv_bytes);
        expected_offset += conv_bytes;
        EXPECT_EQ(L.ssm_offset, expected_offset);
        EXPECT_EQ(L.ssm_size, ssm_bytes);
        expected_offset += ssm_bytes;
    }
    EXPECT_EQ(expected_offset, layout.slot_bytes);
}

TEST(CacheLayout, RejectsNonPositiveLayerCount) {
    Mamba2Config cfg = make_valid_config();
    cfg.num_hidden_layers = 0;
    StatusOr<CacheLayout> result = CacheLayout::from_config(cfg);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), Code::kInvalidArgument);
}

TEST(CacheLayout, RejectsInvalidConvKernel) {
    Mamba2Config cfg = make_valid_config();
    cfg.conv_kernel = 1;
    StatusOr<CacheLayout> result = CacheLayout::from_config(cfg);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), Code::kInvalidArgument);
}

TEST(CacheLayout, RejectsInvalidHiddenOrExpand) {
    Mamba2Config cfg = make_valid_config();
    cfg.hidden_size = 0;
    StatusOr<CacheLayout> result = CacheLayout::from_config(cfg);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), Code::kInvalidArgument);

    cfg = make_valid_config();
    cfg.expand = -1;
    result = CacheLayout::from_config(cfg);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), Code::kInvalidArgument);
}

TEST(CacheLayout, RejectsInvalidSsmDims) {
    Mamba2Config cfg = make_valid_config();
    cfg.num_heads = 0;
    StatusOr<CacheLayout> result = CacheLayout::from_config(cfg);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), Code::kInvalidArgument);
}
