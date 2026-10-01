#include "core/device.h"
#include "ops/backend.h"
#include "ops/cpu/map.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/runner/mamba2_runner.h"
#include <gtest/gtest.h>

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
    for (int h = 0; h < hidden; ++h) {
      m(v, h) = static_cast<float>(v + 1) * 0.01f;
    }
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

StatusOr<std::unique_ptr<Mamba2Runner>> make_runner(const ModelConfig& cfg, int n_slots,
                                                     std::unique_ptr<DeviceAllocator>* out_alloc,
                                                     std::unique_ptr<CachePool>* out_pool) {
  std::unique_ptr<DeviceAllocator> alloc;
  ASSIGN_OR_RETURN(alloc, create_device_allocator(Device::CPU, 0));

  Mamba2Weights weights;
  {
    AllocatorScope scope(alloc.get());
    weights = make_weights(cfg);
  }

  std::unique_ptr<CachePool> pool;
  ASSIGN_OR_RETURN(pool, create_cache_pool(cfg, alloc.get(), n_slots, create_cache_layout));
  auto runner = std::make_unique<Mamba2Runner>(cfg, std::move(weights), cpu_ops());
  *out_alloc = std::move(alloc);
  *out_pool = std::move(pool);
  return runner;
}

} // namespace

TEST(Mamba2Runner, PrefillReturnsVocabLogits) {
  ModelConfig cfg = make_cfg();
  std::unique_ptr<DeviceAllocator> alloc;
  std::unique_ptr<CachePool> pool;
  StatusOr<std::unique_ptr<Mamba2Runner>> runner_or = make_runner(cfg, /*n_slots=*/2, &alloc, &pool);
  ASSERT_TRUE(runner_or.ok()) << runner_or.status().message();
  Mamba2Runner& runner = *runner_or.value();

  StatusOr<CacheHandle> handle_or = pool->acquire();
  ASSERT_TRUE(handle_or.ok()) << handle_or.status().message();
  CacheHandle handle = std::move(handle_or.value());
  StatusOr<std::span<LayerCacheView>> layers = pool->layer_views(handle);
  ASSERT_TRUE(layers.ok()) << layers.status().message();

  const int32_t tokens[] = {1, 2, 3};
  StatusOr<Tensor> logits = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc.get());
    return runner.prefill(tokens, layers.value());
  }();
  ASSERT_TRUE(logits.ok()) << logits.status().message();
  ASSERT_EQ(logits.value().shape.size(), 1u);
  EXPECT_EQ(logits.value().shape[0], cfg.vocab_size);

  ASSERT_TRUE(pool->release(handle).ok());
  EXPECT_FALSE(handle.valid());
}

TEST(Mamba2Runner, DecodeReusesHandleAndReturnsLogits) {
  ModelConfig cfg = make_cfg();
  std::unique_ptr<DeviceAllocator> alloc;
  std::unique_ptr<CachePool> pool;
  StatusOr<std::unique_ptr<Mamba2Runner>> runner_or = make_runner(cfg, /*n_slots=*/2, &alloc, &pool);
  ASSERT_TRUE(runner_or.ok()) << runner_or.status().message();
  Mamba2Runner& runner = *runner_or.value();

  StatusOr<CacheHandle> handle_or = pool->acquire();
  ASSERT_TRUE(handle_or.ok()) << handle_or.status().message();
  CacheHandle handle = std::move(handle_or.value());
  StatusOr<std::span<LayerCacheView>> layers = pool->layer_views(handle);
  ASSERT_TRUE(layers.ok()) << layers.status().message();

  const int32_t prompt[] = {1, 2};
  StatusOr<Tensor> dec = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc.get());
    StatusOr<Tensor> pref = runner.prefill(prompt, layers.value());
    if (!pref.ok())
      return pref.status();
    return runner.decode(/*token=*/3, layers.value(), /*past_len=*/2);
  }();
  ASSERT_TRUE(dec.ok()) << dec.status().message();
  ASSERT_EQ(dec.value().shape.size(), 1u);
  EXPECT_EQ(dec.value().shape[0], cfg.vocab_size);

  ASSERT_TRUE(pool->release(handle).ok());
}

TEST(Mamba2Runner, PrefillEmptyTokensFails) {
  ModelConfig cfg = make_cfg();
  std::unique_ptr<DeviceAllocator> alloc;
  std::unique_ptr<CachePool> pool;
  StatusOr<std::unique_ptr<Mamba2Runner>> runner_or = make_runner(cfg, 1, &alloc, &pool);
  ASSERT_TRUE(runner_or.ok()) << runner_or.status().message();

  StatusOr<CacheHandle> handle_or = pool->acquire();
  ASSERT_TRUE(handle_or.ok()) << handle_or.status().message();
  CacheHandle handle = std::move(handle_or.value());
  StatusOr<std::span<LayerCacheView>> layers = pool->layer_views(handle);
  ASSERT_TRUE(layers.ok()) << layers.status().message();

  StatusOr<Tensor> result = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc.get());
    return runner_or.value()->prefill({}, layers.value());
  }();
  EXPECT_FALSE(result.ok());
  ASSERT_TRUE(pool->release(handle).ok());
}

