#include "io/config_parser.h"
#include "runtime/model_registry.h"
#include <gtest/gtest.h>

TEST(FalconH1ConfigParser, ParsesHalfBBase) {
  FalconH1ConfigParser parser;
  auto parsed = parser.parse(MAMBA_TEST_FALCON_DIR, 2048);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();

  const auto* cfg = dynamic_cast<const FalconH1Config*>(parsed.value().get());
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->kind(), ModelKind::kFalconH1);
  EXPECT_EQ(cfg->model_type, "falcon_h1");
  EXPECT_EQ(cfg->hidden_size, 1024);
  EXPECT_EQ(cfg->num_hidden_layers, 36);
  EXPECT_EQ(cfg->max_seq_length, 2048);
  EXPECT_EQ(cfg->vocab_size, 32784);
  EXPECT_EQ(cfg->num_attention_heads, 8);
  EXPECT_EQ(cfg->num_key_value_heads, 2);
  EXPECT_EQ(cfg->mamba_n_heads, 24);
  EXPECT_EQ(cfg->mamba_d_ssm, 1536);
  EXPECT_EQ(cfg->mamba_d_state, 128);
  ASSERT_EQ(cfg->ssm_multipliers.size(), 5u);
  ASSERT_EQ(cfg->mlp_multipliers.size(), 2u);
  EXPECT_EQ(cfg->eos_token_id, 11);
}

TEST(ModelRegistry, LookupAndParseFalconH1) {
  auto entry_or = ModelRegistry::lookup(MAMBA_TEST_FALCON_DIR);
  ASSERT_TRUE(entry_or.ok()) << entry_or.status().message();
  EXPECT_EQ(entry_or.value()->kind, ModelKind::kFalconH1);

  auto cfg_or = entry_or.value()->parse(MAMBA_TEST_FALCON_DIR, 2048);
  ASSERT_TRUE(cfg_or.ok()) << cfg_or.status().message();
  EXPECT_EQ(cfg_or.value()->kind(), ModelKind::kFalconH1);
}
