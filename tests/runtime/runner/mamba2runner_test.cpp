#include "runtime/runner/mamba2runner.h"

#include "ops/cpu/map.h"
#include "ops/cpu/mamba2_mixer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

Tensor make_f32(std::vector<int64_t> shape, float fill = 0.f) {
    Tensor t;
    t.shape = std::move(shape);
    t.dtype = Dtype::F32;
    t.buffer.device = Device::CPU;
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    t.buffer.bytes = static_cast<size_t>(n) * sizeof(float);
    t.buffer.data = std::malloc(t.buffer.bytes);
    auto* p = static_cast<float*>(t.buffer.data);
    for (int64_t i = 0; i < n; ++i) p[i] = fill;
    return t;
}

Tensor make_embeddings(int vocab, int hidden) {
    Tensor t = make_f32({vocab, hidden}, 0.f);
    auto m = as_mat_f32(t);
    for (int v = 0; v < vocab; ++v) {
        for (int h = 0; h < hidden; ++h) {
            m(v, h) = static_cast<float>(v + 1) * 0.01f;
        }
    }
    return t;
}

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
    return cfg;
}

Mamba2Weights make_weights(const Mamba2Config& cfg) {
    const int64_t D = cfg.hidden_size;
    const int64_t I = D * cfg.expand;
    const int64_t G = cfg.n_groups;
    const int64_t N = cfg.state_size;
    const int64_t H = cfg.num_heads;
    const int64_t conv_dim = I + 2 * G * N;
    const int64_t proj_size = I + conv_dim + H;

    Mamba2Weights w;
    w.embeddings = make_embeddings(cfg.vocab_size, cfg.hidden_size);
    w.norm_f = make_f32({D}, 1.f);
    w.lm_head = std::nullopt;
    w.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));

    for (auto& layer : w.layers) {
        layer.layer_norm = make_f32({D}, 1.f);
        layer.mixer_norm = make_f32({I}, 1.f);
        layer.in_proj = make_f32({proj_size, D}, 0.01f);
        layer.out_proj = make_f32({D, I}, 0.01f);
        layer.conv1d = make_f32({conv_dim, cfg.conv_kernel}, 0.1f);
        // Identity-ish newest tap
        {
            auto cw = as_mat_f32(layer.conv1d);
            cw.setZero();
            for (int64_t c = 0; c < conv_dim; ++c) cw(c, cfg.conv_kernel - 1) = 1.f;
        }
        layer.conv1d_bias = make_f32({conv_dim}, 0.f);
        layer.dt_bias = make_f32({H}, 0.f);
        layer.A_log = make_f32({H}, 0.f);
        {
            auto a = as_vec_f32(layer.A_log);
            for (int h = 0; h < H; ++h) a(h) = std::log(static_cast<float>(h + 1));
        }
        layer.D = make_f32({H}, 1.f);
    }
    return w;
}

}  // namespace

TEST(Mamba2Mixer, CausalConvUpdatesCacheAndAppliesSilu) {
    const int64_t C = 2;
    const int K = 3;
    const int64_t T = 2;
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};  // t0:[1,2] t1:[3,4]
    std::vector<float> weight(static_cast<size_t>(C * K), 0.f);
    // only newest tap = 1
    weight[0 * K + (K - 1)] = 1.f;
    weight[1 * K + (K - 1)] = 1.f;
    std::vector<float> bias = {0.f, 0.f};
    std::vector<float> cache(static_cast<size_t>(C * (K - 1)), 0.f);
    std::vector<float> y(static_cast<size_t>(T * C), 0.f);

    ASSERT_TRUE(causal_conv1d_f32(x.data(), T, C, K, weight.data(), bias.data(), cache.data(),
                                  y.data(), /*silu=*/false)
                    .ok());

    EXPECT_FLOAT_EQ(y[0], 1.f);
    EXPECT_FLOAT_EQ(y[1], 2.f);
    EXPECT_FLOAT_EQ(y[2], 3.f);
    EXPECT_FLOAT_EQ(y[3], 4.f);
    // cache should hold last K-1 inputs: after t0 [0,1]/[0,2]; after t1 [1,3]/[2,4]
    EXPECT_FLOAT_EQ(cache[0 * (K - 1) + 0], 1.f);
    EXPECT_FLOAT_EQ(cache[0 * (K - 1) + 1], 3.f);
    EXPECT_FLOAT_EQ(cache[1 * (K - 1) + 0], 2.f);
    EXPECT_FLOAT_EQ(cache[1 * (K - 1) + 1], 4.f);
}

TEST(Mamba2Runner, PrefillReturnsCacheAndVocabLogits) {
    Mamba2Config cfg = make_cfg();
    Mamba2Weights weights = make_weights(cfg);
    Mamba2Runner runner(cfg, std::move(weights), Device::CPU, 0, /*n_slots=*/2);

    const int32_t tokens[] = {1, 2, 3};
    StatusOr<PrefillResult> result = runner.prefill(tokens);
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_TRUE(result.value().cache.valid());
    ASSERT_EQ(result.value().logits.shape.size(), 1u);
    EXPECT_EQ(result.value().logits.shape[0], cfg.vocab_size);

    CacheHandle handle = result.value().cache;
    ASSERT_TRUE(runner.release(handle).ok());
    EXPECT_FALSE(handle.valid());
}

TEST(Mamba2Runner, DecodeReusesHandleAndReturnsLogits) {
    Mamba2Config cfg = make_cfg();
    Mamba2Weights weights = make_weights(cfg);
    Mamba2Runner runner(cfg, std::move(weights), Device::CPU, 0, /*n_slots=*/2);

    const int32_t prompt[] = {1, 2};
    StatusOr<PrefillResult> pref = runner.prefill(prompt);
    ASSERT_TRUE(pref.ok()) << pref.status().message();

    StatusOr<DecodeResult> dec = runner.decode(pref.value().cache, /*token=*/3);
    ASSERT_TRUE(dec.ok()) << dec.status().message();
    ASSERT_EQ(dec.value().logits.shape.size(), 1u);
    EXPECT_EQ(dec.value().logits.shape[0], cfg.vocab_size);

    CacheHandle handle = pref.value().cache;
    ASSERT_TRUE(runner.release(handle).ok());
}

TEST(Mamba2Runner, PrefillEmptyTokensFails) {
    Mamba2Config cfg = make_cfg();
    Mamba2Weights weights = make_weights(cfg);
    Mamba2Runner runner(cfg, std::move(weights), Device::CPU, 0, 1);

    StatusOr<PrefillResult> result = runner.prefill({});
    EXPECT_FALSE(result.ok());
}

TEST(Mamba2Runner, DecodeInvalidHandleFails) {
    Mamba2Config cfg = make_cfg();
    Mamba2Weights weights = make_weights(cfg);
    Mamba2Runner runner(cfg, std::move(weights), Device::CPU, 0, 1);

    CacheHandle bad;
    StatusOr<DecodeResult> result = runner.decode(bad, 0);
    EXPECT_FALSE(result.ok());
}
