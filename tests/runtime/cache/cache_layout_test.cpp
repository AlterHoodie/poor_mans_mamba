#include "runtime/cache/cache_layout.h"
#include <gtest/gtest.h>

#include <cstddef>

namespace {

ModelConfig make_valid_config() {
  ModelConfig cfg{};
  cfg.layout = ArchLayout::MambaOnly;
  cfg.num_hidden_layers = 2;
  cfg.hidden_size = 64;
  cfg.ssm.d_inner = 128;
  cfg.ssm.d_conv = 4;
  cfg.ssm.n_heads = 8;
  cfg.ssm.d_head = 16;
  cfg.ssm.d_state = 16;
  cfg.ssm.n_groups = 1;
  return cfg;
}

} // namespace

TEST(CacheLayout, FromConfigPacksLayersContiguously) {
  const ModelConfig cfg = make_valid_config();
  StatusOr<CacheLayout> result = create_cache_layout(cfg);
  ASSERT_TRUE(result.ok());

  const CacheLayout& layout = result.value();
  EXPECT_EQ(layout.num_layers, 2);
  ASSERT_EQ(layout.layers.size(), 2u);

  const size_t intermediate = static_cast<size_t>(cfg.ssm.d_inner);
  const size_t conv_dim =
      intermediate + 2 * static_cast<size_t>(cfg.ssm.n_groups) * static_cast<size_t>(cfg.ssm.d_state);
  const size_t conv_bytes = conv_dim * static_cast<size_t>(cfg.ssm.d_conv - 1) * sizeof(float);
  const size_t ssm_bytes = static_cast<size_t>(cfg.ssm.n_heads) * static_cast<size_t>(cfg.ssm.d_head) *
                           static_cast<size_t>(cfg.ssm.d_state) * sizeof(float);
  const size_t per_layer = conv_bytes + ssm_bytes;

  EXPECT_EQ(layout.slot_bytes, per_layer * 2);

  size_t expected_offset = 0;
  for (int layer = 0; layer < layout.num_layers; ++layer) {
    const LayerEntry& L = layout.layers[static_cast<size_t>(layer)];
    EXPECT_EQ(L.layer_idx, layer);
    EXPECT_EQ(L.kind, LayerCacheKind::Mamba2);
    EXPECT_EQ(L.conv.offset, expected_offset);
    EXPECT_EQ(L.conv.bytes, conv_bytes);
    expected_offset += conv_bytes;
    EXPECT_EQ(L.ssm.offset, expected_offset);
    EXPECT_EQ(L.ssm.bytes, ssm_bytes);
    expected_offset += ssm_bytes;
  }
  EXPECT_EQ(expected_offset, layout.slot_bytes);
}

TEST(CacheLayout, RejectsNonPositiveLayerCount) {
  ModelConfig cfg = make_valid_config();
  cfg.num_hidden_layers = 0;
  StatusOr<CacheLayout> result = create_cache_layout(cfg);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), Code::kInvalidArgument);
}

TEST(CacheLayout, RejectsInvalidConvKernel) {
  ModelConfig cfg = make_valid_config();
  cfg.ssm.d_conv = 1;
  StatusOr<CacheLayout> result = create_cache_layout(cfg);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), Code::kInvalidArgument);
}

TEST(CacheLayout, RejectsInvalidHiddenOrExpand) {
  ModelConfig cfg = make_valid_config();
  cfg.hidden_size = 0;
  StatusOr<CacheLayout> result = create_cache_layout(cfg);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), Code::kInvalidArgument);

  cfg = make_valid_config();
  cfg.ssm.d_inner = 0;
  result = create_cache_layout(cfg);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), Code::kInvalidArgument);
}

TEST(CacheLayout, RejectsInvalidSsmDims) {
  ModelConfig cfg = make_valid_config();
  cfg.ssm.n_heads = 0;
  StatusOr<CacheLayout> result = create_cache_layout(cfg);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), Code::kInvalidArgument);
}
