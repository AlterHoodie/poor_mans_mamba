#include "core/device.h"
#include "ops/backend.h"
#include "ops/cpu/map.h"
#include "ops/cpu/reductions.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/runner/mamba2_runner.h"
#include "runtime/runner/runner.h"
#include "runtime/worker.h"
#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace {

Tensor make_f32(std::vector<int64_t> shape, float fill = 0.f) {
  StatusOr<Tensor> t_or = allocate_f32_tensor(std::move(shape));
  EXPECT_TRUE(t_or.ok()) << t_or.status().message();
  Tensor t = std::move(t_or.value());
  auto* p = static_cast<float*>(t.buffer.ptr);
  const int64_t n = static_cast<int64_t>(t.buffer.bytes / sizeof(float));
  for (int64_t i = 0; i < n; ++i)
    p[i] = fill;
  return t;
}

Tensor make_embeddings(int vocab, int hidden) {
  Tensor t = make_f32({vocab, hidden}, 0.f);
  auto m = as_mat_f32(t);
  for (int v = 0; v < vocab; ++v) {
    for (int h = 0; h < hidden; ++h)
      m(v, h) = static_cast<float>(v + 1) * 0.01f;
  }
  return t;
}

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
  cfg.max_seq_length = 32;
  return cfg;
}

Mamba2Weights make_weights(const ModelConfig& cfg) {
  const int64_t D = cfg.hidden_size;
  const int64_t I = cfg.ssm.d_inner;
  const int64_t G = cfg.ssm.n_groups;
  const int64_t N = cfg.ssm.d_state;
  const int64_t H = cfg.ssm.n_heads;
  const int64_t K = cfg.ssm.d_conv;
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
    layer.conv1d = make_f32({conv_dim, K}, 0.1f);
    {
      auto cw = as_mat_f32(layer.conv1d);
      cw.setZero();
      for (int64_t c = 0; c < conv_dim; ++c)
        cw(c, K - 1) = 1.f;
    }
    layer.conv1d_bias = make_f32({conv_dim}, 0.f);
    layer.dt_bias = make_f32({H}, 0.f);
    layer.A_log = make_f32({H}, 0.f);
    {
      auto a = as_vec_f32(layer.A_log);
      for (int h = 0; h < H; ++h)
        a(h) = std::log(static_cast<float>(h + 1));
    }
    layer.D = make_f32({H}, 1.f);
  }
  return w;
}

// Deterministic runner: peaks logits at successive entries of `token_seq_`.
class FakeRunner : public Runner {
public:
  FakeRunner(int vocab, std::vector<int32_t> token_seq, int overflow_after_decodes = -1)
      : vocab_(vocab), token_seq_(std::move(token_seq)),
        overflow_after_decodes_(overflow_after_decodes) {
    assert(!token_seq_.empty());
    auto alloc_or = create_device_allocator(Device::CPU, 0);
    assert(alloc_or.ok());
    alloc_ = std::move(alloc_or.value());
  }

  FakeRunner(int vocab, int32_t next_token) : FakeRunner(vocab, std::vector<int32_t>{next_token}) {}

  StatusOr<Tensor> prefill(std::span<const int32_t> tokens,
                           std::span<LayerCacheView> /*layers*/) override {
    if (tokens.empty())
      return Status::InvalidArgument("empty prompt");
    Tensor logits;
    ASSIGN_OR_RETURN(logits, make_logits_peak_(vocab_, peak_token_()));
    ++step_;
    return std::move(logits);
  }

  StatusOr<Tensor> decode(int32_t /*token*/, std::span<LayerCacheView> /*layers*/,
                          int64_t /*past_len*/) override {
    if (overflow_after_decodes_ >= 0 && decode_calls_ >= overflow_after_decodes_) {
      return Status::KvCacheOverflow("KV cache overflow");
    }
    ++decode_calls_;
    Tensor logits;
    ASSIGN_OR_RETURN(logits, make_logits_peak_(vocab_, peak_token_()));
    ++step_;
    return std::move(logits);
  }

private:
  StatusOr<Tensor> make_logits_peak_(int vocab, int32_t peak_id, float peak = 10.f) {
    AllocatorScope scope(alloc_.get());
    StatusOr<Tensor> t_or = allocate_f32_tensor({vocab});
    if (!t_or.ok())
      return t_or.status();
    Tensor t = std::move(t_or.value());
    auto* p = static_cast<float*>(t.buffer.ptr);
    for (int i = 0; i < vocab; ++i)
      p[i] = 0.f;
    p[peak_id] = peak;
    return t;
  }

  int32_t peak_token_() const {
    const size_t i = std::min(step_, token_seq_.size() - 1);
    return token_seq_[i];
  }

  int vocab_;
  std::vector<int32_t> token_seq_;
  int overflow_after_decodes_ = -1;
  int decode_calls_ = 0;
  size_t step_ = 0;
  std::unique_ptr<DeviceAllocator> alloc_;
};

StatusOr<std::vector<int32_t>> greedy_generate(Runner& runner, CachePool& pool,
                                               DeviceAllocator& alloc,
                                               std::span<const int32_t> prompt,
                                               const GenerateParams& params) {
  if (prompt.empty())
    return Status::InvalidArgument("empty prompt");
  if (params.eos_id < 0)
    return Status::InvalidArgument("eos_id required");

  StatusOr<CacheHandle> handle_or = pool.acquire();
  if (!handle_or.ok())
    return handle_or.status();
  CacheHandle handle = std::move(handle_or.value());

  StatusOr<std::span<LayerCacheView>> layers = pool.layer_views(handle);
  if (!layers.ok()) {
    (void)pool.release(handle);
    return layers.status();
  }

  AllocatorScope scope(&alloc);

  std::vector<int32_t> out;
  StatusOr<Tensor> logits = runner.prefill(prompt, layers.value());
  if (!logits.ok()) {
    (void)pool.release(handle);
    return logits.status();
  }

  for (int i = 0; i < params.max_new_tokens; ++i) {
    StatusOr<int32_t> tok = argmax(logits.value());
    if (!tok.ok()) {
      (void)pool.release(handle);
      return tok.status();
    }
    out.push_back(tok.value());
    if (tok.value() == params.eos_id)
      break;
    if (i + 1 >= params.max_new_tokens)
      break;

    // out already holds the just-sampled token; KV write index is prompt + prior gens.
    const int64_t past_len = static_cast<int64_t>(prompt.size() + out.size() - 1);
    logits = runner.decode(tok.value(), layers.value(), past_len);
    if (!logits.ok()) {
      if (logits.status().code() == Code::kKvCacheOverflow) {
        (void)pool.release(handle);
        return out; // partial
      }
      (void)pool.release(handle);
      return logits.status();
    }
  }

  if (Status s = pool.release(handle); !s.ok())
    return s;
  return out;
}

StatusOr<std::unique_ptr<CachePool>> make_pool(const ModelConfig& cfg, DeviceAllocator* alloc,
                                               int n_slots) {
  return create_cache_pool(cfg, alloc, n_slots, create_cache_layout);
}

} // namespace

TEST(GreedyGenerate, RejectsEmptyPrompt) {
  FakeRunner runner(/*vocab=*/8, /*next_token=*/1);
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  ModelConfig cfg = make_cfg();
  auto pool_or = make_pool(cfg, alloc_or.value().get(), 1);
  ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();

  GenerateParams params{.max_new_tokens = 4, .eos_id = 0};
  std::vector<int32_t> prompt;
  StatusOr<std::vector<int32_t>> out =
      greedy_generate(runner, *pool_or.value(), *alloc_or.value(), prompt, params);
  EXPECT_FALSE(out.ok());
}

TEST(GreedyGenerate, RejectsMissingEos) {
  FakeRunner runner(/*vocab=*/8, /*next_token=*/1);
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  ModelConfig cfg = make_cfg();
  auto pool_or = make_pool(cfg, alloc_or.value().get(), 1);
  ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();

  GenerateParams params{.max_new_tokens = 4, .eos_id = -1};
  std::vector<int32_t> prompt = {1, 2};
  StatusOr<std::vector<int32_t>> out =
      greedy_generate(runner, *pool_or.value(), *alloc_or.value(), prompt, params);
  EXPECT_FALSE(out.ok());
}

TEST(GreedyGenerate, StopsAtMaxNewTokens) {
  FakeRunner runner(/*vocab=*/8, /*next_token=*/3);
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  ModelConfig cfg = make_cfg();
  auto pool_or = make_pool(cfg, alloc_or.value().get(), 1);
  ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();

  GenerateParams params{.max_new_tokens = 5, .eos_id = 0};
  std::vector<int32_t> prompt = {1, 2};
  StatusOr<std::vector<int32_t>> out =
      greedy_generate(runner, *pool_or.value(), *alloc_or.value(), prompt, params);
  ASSERT_TRUE(out.ok()) << out.status().message();
  ASSERT_EQ(out.value().size(), 5u);
  for (int32_t t : out.value())
    EXPECT_EQ(t, 3);
}

TEST(GreedyGenerate, StopsAtEos) {
  FakeRunner runner(/*vocab=*/8, /*next_token=*/7);
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  ModelConfig cfg = make_cfg();
  auto pool_or = make_pool(cfg, alloc_or.value().get(), 1);
  ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();

  GenerateParams params{.max_new_tokens = 16, .eos_id = 7};
  std::vector<int32_t> prompt = {1};
  StatusOr<std::vector<int32_t>> out =
      greedy_generate(runner, *pool_or.value(), *alloc_or.value(), prompt, params);
  ASSERT_TRUE(out.ok()) << out.status().message();
  ASSERT_EQ(out.value().size(), 1u);
  EXPECT_EQ(out.value()[0], 7);
}

TEST(GreedyGenerate, EmitsExactTokenSequence) {
  FakeRunner runner(/*vocab=*/8, /*token_seq=*/{3, 5, 2, 0});
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  ModelConfig cfg = make_cfg();
  auto pool_or = make_pool(cfg, alloc_or.value().get(), 1);
  ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();

  GenerateParams params{.max_new_tokens = 4, .eos_id = 0};
  std::vector<int32_t> prompt = {1};
  StatusOr<std::vector<int32_t>> out =
      greedy_generate(runner, *pool_or.value(), *alloc_or.value(), prompt, params);
  ASSERT_TRUE(out.ok()) << out.status().message();
  const std::vector<int32_t> expected = {3, 5, 2, 0};
  EXPECT_EQ(out.value(), expected);
}

TEST(GreedyGenerate, ReturnsPartialOnKvCacheOverflow) {
  FakeRunner runner(/*vocab=*/8, /*token_seq=*/{3, 5, 2, 1}, /*overflow_after_decodes=*/2);
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  ModelConfig cfg = make_cfg();
  auto pool_or = make_pool(cfg, alloc_or.value().get(), 1);
  ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();

  GenerateParams params{.max_new_tokens = 8, .eos_id = 0};
  std::vector<int32_t> prompt = {1};
  StatusOr<std::vector<int32_t>> out =
      greedy_generate(runner, *pool_or.value(), *alloc_or.value(), prompt, params);
  ASSERT_TRUE(out.ok()) << out.status().message();
  const std::vector<int32_t> expected = {3, 5, 2};
  EXPECT_EQ(out.value(), expected);
}

TEST(GreedyGenerate, RealMamba2GreedyProducesTokens) {
  ModelConfig cfg = make_cfg();
  std::unique_ptr<DeviceAllocator> alloc;
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  alloc = std::move(alloc_or.value());

  Mamba2Weights weights;
  {
    AllocatorScope scope(alloc.get());
    weights = make_weights(cfg);
  }
  auto runner = std::make_unique<Mamba2Runner>(cfg, std::move(weights), cpu_ops());
  auto pool_or = make_pool(cfg, alloc.get(), 2);
  ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();

  const std::vector<int32_t> prompt = {1, 2};
  const GenerateParams params{.max_new_tokens = 3, .eos_id = 0};
  StatusOr<std::vector<int32_t>> out =
      greedy_generate(*runner, *pool_or.value(), *alloc, prompt, params);
  ASSERT_TRUE(out.ok()) << out.status().message();
  ASSERT_EQ(out.value().size(), 3u);
  for (int32_t t : out.value()) {
    EXPECT_GE(t, 0);
    EXPECT_LT(t, cfg.vocab_size);
  }
}
