#include "runtime/scheduler.h"
#include "runtime/runner/mamba2runner.h"

#include "ops/cpu/map.h"
#include "ops/cpu/reductions.h"

#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace {

Tensor make_logits_peak(int vocab, int32_t peak_id, float peak = 10.f) {
    Tensor t;
    t.shape = {vocab};
    t.dtype = Dtype::F32;
    t.buffer.device = Device::CPU;
    t.buffer.bytes = static_cast<size_t>(vocab) * sizeof(float);
    t.buffer.data = std::malloc(t.buffer.bytes);
    auto* p = static_cast<float*>(t.buffer.data);
    for (int i = 0; i < vocab; ++i) p[i] = 0.f;
    p[peak_id] = peak;
    return t;
}

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

// Deterministic runner: peaks logits at successive entries of `token_seq_`.
class FakeRunner : public Runner {
   public:
    FakeRunner(int vocab, std::vector<int32_t> token_seq)
        : vocab_(vocab), token_seq_(std::move(token_seq)) {
        assert(!token_seq_.empty());
    }

    // Convenience: always emit the same token.
    FakeRunner(int vocab, int32_t next_token) : FakeRunner(vocab, std::vector<int32_t>{next_token}) {}

    StatusOr<PrefillResult> prefill(std::span<const int32_t> tokens) override {
        if (tokens.empty()) return Status::InvalidArgument("empty prompt");
        PrefillResult r;
        r.cache = CacheHandle(next_handle_++);
        live_.insert(r.cache.id());
        r.logits = make_logits_peak(vocab_, peak_token_());
        ++step_;
        return r;
    }

    StatusOr<DecodeResult> decode(const CacheHandle& cache, int32_t /*token*/) override {
        if (!cache.valid() || !live_.count(cache.id())) {
            return Status::InvalidArgument("invalid cache");
        }
        DecodeResult r;
        r.logits = make_logits_peak(vocab_, peak_token_());
        ++step_;
        return r;
    }

    Status release(CacheHandle& cache) override {
        if (!cache.valid() || !live_.count(cache.id())) {
            return Status::InvalidArgument("invalid cache");
        }
        live_.erase(cache.id());
        cache.invalidate();
        return Status::Ok();
    }

    bool has_live_caches() const { return !live_.empty(); }

   private:
    int32_t peak_token_() const {
        const size_t i = std::min(step_, token_seq_.size() - 1);
        return token_seq_[i];
    }

    int vocab_;
    std::vector<int32_t> token_seq_;
    size_t step_ = 0;
    int next_handle_ = 0;
    std::unordered_set<int> live_;
};

}  // namespace

TEST(Scheduler, RejectsEmptyPrompt) {
    FakeRunner runner(/*vocab=*/8, /*next_token=*/1);
    Scheduler sched(runner);
    GenerateParams params{.max_new_tokens = 4, .eos_id = 0};
    std::vector<int32_t> prompt;
    StatusOr<std::vector<int32_t>> out = sched.generate(prompt, params);
    EXPECT_FALSE(out.ok());
}

TEST(Scheduler, RejectsMissingEos) {
    FakeRunner runner(/*vocab=*/8, /*next_token=*/1);
    Scheduler sched(runner);
    GenerateParams params{.max_new_tokens = 4, .eos_id = -1};
    std::vector<int32_t> prompt = {1, 2};
    StatusOr<std::vector<int32_t>> out = sched.generate(prompt, params);
    EXPECT_FALSE(out.ok());
}

TEST(Scheduler, StopsAtMaxNewTokens) {
    FakeRunner runner(/*vocab=*/8, /*next_token=*/3);  // never EOS
    Scheduler sched(runner);
    GenerateParams params{.max_new_tokens = 5, .eos_id = 0};
    std::vector<int32_t> prompt = {1, 2};

    StatusOr<std::vector<int32_t>> out = sched.generate(prompt, params);
    ASSERT_TRUE(out.ok()) << out.status().message();
    ASSERT_EQ(out.value().size(), 5u);
    for (int32_t t : out.value()) EXPECT_EQ(t, 3);
    EXPECT_FALSE(runner.has_live_caches());
}

TEST(Scheduler, StopsAtEos) {
    FakeRunner runner(/*vocab=*/8, /*next_token=*/7);
    Scheduler sched(runner);
    GenerateParams params{.max_new_tokens = 16, .eos_id = 7};
    std::vector<int32_t> prompt = {1};

    StatusOr<std::vector<int32_t>> out = sched.generate(prompt, params);
    ASSERT_TRUE(out.ok()) << out.status().message();
    ASSERT_EQ(out.value().size(), 1u);
    EXPECT_EQ(out.value()[0], 7);
    EXPECT_FALSE(runner.has_live_caches());
}

TEST(Scheduler, EmitsExactTokenSequence) {
    FakeRunner runner(/*vocab=*/8, /*token_seq=*/{3, 5, 2, 0});
    Scheduler sched(runner);
    GenerateParams params{.max_new_tokens = 4, .eos_id = 0};
    std::vector<int32_t> prompt = {1};

    StatusOr<std::vector<int32_t>> out = sched.generate(prompt, params);
    ASSERT_TRUE(out.ok()) << out.status().message();
    const std::vector<int32_t> expected = {3, 5, 2, 0};
    EXPECT_EQ(out.value(), expected);
    EXPECT_FALSE(runner.has_live_caches());
}

TEST(Scheduler, GenerateMatchesManualGreedyTokens) {
    Mamba2Config cfg = make_cfg();
    Mamba2Weights weights = make_weights(cfg);
    Mamba2Runner runner(cfg, std::move(weights), Device::CPU, 0, /*n_slots=*/2);

    const std::vector<int32_t> prompt = {1, 2};
    const GenerateParams params{.max_new_tokens = 3, .eos_id = 0};

    // Independent greedy path: must match Scheduler::generate token-for-token.
    StatusOr<PrefillResult> pref = runner.prefill(prompt);
    ASSERT_TRUE(pref.ok()) << pref.status().message();

    std::vector<int32_t> expected;
    StatusOr<int32_t> token = argmax(pref.value().logits);
    ASSERT_TRUE(token.ok()) << token.status().message();
    expected.push_back(token.value());

    for (int i = 1; i < params.max_new_tokens && token.value() != params.eos_id; ++i) {
        StatusOr<DecodeResult> dec = runner.decode(pref.value().cache, token.value());
        ASSERT_TRUE(dec.ok()) << dec.status().message();
        token = argmax(dec.value().logits);
        ASSERT_TRUE(token.ok()) << token.status().message();
        expected.push_back(token.value());
    }
    ASSERT_TRUE(runner.release(pref.value().cache).ok());

    Scheduler sched(runner);
    StatusOr<std::vector<int32_t>> out = sched.generate(prompt, params);
    ASSERT_TRUE(out.ok()) << out.status().message();
    ASSERT_EQ(out.value().size(), expected.size());
    EXPECT_EQ(out.value(), expected);
    for (int32_t t : out.value()) {
        EXPECT_GE(t, 0);
        EXPECT_LT(t, cfg.vocab_size);
    }
}
