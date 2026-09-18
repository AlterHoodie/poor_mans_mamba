#include "io/config_parser.h"
#include "runtime/model_registry.h"
#include <gtest/gtest.h>

TEST(Mamba2ConfigParser, Parses130mHf) {
  Mamba2ConfigParser parser;
  auto parsed = parser.parse(MAMBA_TEST_MODEL_DIR, 2048);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();

  const ModelConfig& cfg = *parsed.value();
  EXPECT_EQ(cfg.layout, ArchLayout::MambaOnly);
  EXPECT_EQ(cfg.model_type, "mamba2");
  EXPECT_EQ(cfg.hidden_size, 768);
  EXPECT_EQ(cfg.num_hidden_layers, 24);
  EXPECT_EQ(cfg.max_seq_length, 2048);
  EXPECT_EQ(cfg.vocab_size, 50288);
  EXPECT_TRUE(cfg.tie_word_embeddings);
  EXPECT_FALSE(cfg.attn.has_value());
  EXPECT_FALSE(cfg.mlp.has_value());
  EXPECT_FALSE(cfg.scales.has_value());

  EXPECT_EQ(cfg.ssm.n_heads, 24);
  EXPECT_EQ(cfg.ssm.d_head, 64);
  EXPECT_EQ(cfg.ssm.d_state, 128);
  EXPECT_EQ(cfg.ssm.d_conv, 4);
  EXPECT_EQ(cfg.ssm.d_inner, 1536); // expand * hidden_size
  EXPECT_EQ(cfg.ssm.n_groups, 1);
  EXPECT_EQ(cfg.ssm.chunk_size, 256);
  EXPECT_TRUE(cfg.ssm.use_conv_bias);
  EXPECT_FALSE(cfg.ssm.use_proj_bias);
  EXPECT_TRUE(cfg.ssm.gated_rms_norm);
  EXPECT_FLOAT_EQ(cfg.rms_norm_eps, 1e-5f);
  EXPECT_EQ(cfg.eos_token_id, 0);
}

TEST(ModelRegistry, LookupAndParseMamba2) {
  auto entry_or = ModelRegistry::lookup(MAMBA_TEST_MODEL_DIR);
  ASSERT_TRUE(entry_or.ok()) << entry_or.status().message();

  auto cfg_or = entry_or.value()->parse(MAMBA_TEST_MODEL_DIR, 2048);
  ASSERT_TRUE(cfg_or.ok()) << cfg_or.status().message();
  EXPECT_EQ(cfg_or.value()->layout, ArchLayout::MambaOnly);
  EXPECT_EQ(cfg_or.value()->model_type, "mamba2");
}
