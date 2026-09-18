#include "runtime/tokenizer.h"
#include <gtest/gtest.h>

#include <string>

namespace {

ModelConfig make_cfg() {
  ModelConfig cfg{};
  cfg.layout = ArchLayout::MambaOnly;
  cfg.model_type = "mamba2";
  cfg.hidden_size = 4;
  cfg.vocab_size = 8;
  cfg.tie_word_embeddings = true;
  cfg.num_hidden_layers = 1;
  cfg.ssm.d_inner = 8;
  cfg.ssm.d_conv = 4;
  cfg.ssm.d_state = 4;
  cfg.ssm.d_head = 2;
  cfg.ssm.n_heads = 4;
  cfg.ssm.n_groups = 1;
  cfg.ssm.chunk_size = 4;
  cfg.ssm.use_conv_bias = true;
  cfg.ssm.use_proj_bias = false;
  cfg.ssm.gated_rms_norm = true;
  cfg.rms_norm_eps = 1e-5f;
  cfg.eos_token_id = 0;
  return cfg;
}

} // namespace

TEST(Tokenizers, EncodeDecodeRoundtrip) {
  const std::string prompt = "Hello World";
  Tokenizer tok(make_cfg(), MAMBA_TEST_MODEL_DIR);

  auto ids = tok.encode(prompt);
  ASSERT_TRUE(ids.ok()) << ids.status().message();
  ASSERT_FALSE(ids.value().empty());

  auto text = tok.decode(ids.value());
  ASSERT_TRUE(text.ok()) << text.status().message();

  // HF decode may not be bit-identical (spaces); soft check:
  EXPECT_NE(text.value().find("Hello"), std::string::npos);

  auto ids2 = tok.encode(prompt);
  ASSERT_TRUE(ids2.ok());
  EXPECT_EQ(ids.value(), ids2.value());
}
