#include <string>

#include "runtime/tokenizer.h"

#include <gtest/gtest.h>

namespace {

Mamba2Config make_cfg() {
    Mamba2Config cfg{};
    cfg.model_type = "mamba2";
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.tie_word_embeddings = true;
    cfg.num_hidden_layers = 1;
    cfg.expand = 2;
    cfg.conv_kernel = 4;
    cfg.state_size = 4;
    cfg.head_dim = 2;
    cfg.num_heads = 4;
    cfg.n_groups = 1;
    cfg.chunk_size = 4;
    cfg.time_step_rank = 1;
    cfg.layer_norm_epsilon = 1e-5f;
    cfg.use_conv_bias = true;
    cfg.use_bias = false;
    cfg.eos_token_id = 0;
    return cfg;
}

}  // namespace

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
