#include "core/device.h"
#include "ops/backend.h"
#include "ops/cpu/map.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/model_registry.h"
#include "runtime/runner/falconh1_runner.h"
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <span>
#include <string>
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
  cfg.layout = ArchLayout::ParallelHybrid;
  cfg.model_type = "falcon_h1";
  cfg.hidden_size = 8;
  cfg.vocab_size = 8;
  cfg.tie_word_embeddings = true;
  cfg.num_hidden_layers = 1;
  cfg.max_seq_length = 32;
  cfg.rms_norm_eps = 1e-5f;

  AttnConfig attn{};
  attn.n_q_heads = 4;
  attn.n_kv_heads = 2;
  attn.head_dim = 4;
  attn.rope_theta = 10000.f;
  cfg.attn = attn;

  cfg.ssm.n_heads = 2;
  cfg.ssm.d_head = 4;
  cfg.ssm.d_inner = 8;
  cfg.ssm.d_state = 4;
  cfg.ssm.n_groups = 1;
  cfg.ssm.d_conv = 4;
  cfg.ssm.use_conv_bias = true;
  cfg.ssm.gated_rms_norm = false;

  ScaleConfig scales{};
  scales.ssm_chunk = {1.f, 1.f, 1.f, 1.f, 1.f};
  scales.ssm_in = 1.f;
  scales.ssm_out = 1.f;
  scales.attn_in = 1.f;
  scales.attn_out = 1.f;
  scales.key = 1.f;
  scales.mlp = {1.f, 1.f};
  scales.embedding = 1.f;
  scales.lm_head = 1.f;
  cfg.scales = scales;

  MlpConfig mlp{};
  mlp.intermediate_size = 16;
  cfg.mlp = mlp;
  return cfg;
}

FalconH1Weights make_weights(const ModelConfig& cfg) {
  const int64_t D = cfg.hidden_size;
  const int64_t I = cfg.ssm.d_inner;
  const int64_t G = cfg.ssm.n_groups;
  const int64_t N = cfg.ssm.d_state;
  const int64_t H = cfg.ssm.n_heads;
  const int64_t K = cfg.ssm.d_conv;
  const int64_t conv_dim = I + 2 * G * N;
  const int64_t proj_size = I + conv_dim + H;
  const int64_t q_dim = static_cast<int64_t>(cfg.attn->n_q_heads) * cfg.attn->head_dim;
  const int64_t kv_dim = static_cast<int64_t>(cfg.attn->n_kv_heads) * cfg.attn->head_dim;
  const int64_t F = cfg.mlp->intermediate_size;

  FalconH1Weights w;
  w.embeddings = make_embeddings(cfg.vocab_size, cfg.hidden_size);
  w.norm_f = make_f32({D}, 1.f);
  w.lm_head = std::nullopt;
  w.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));

  for (auto& layer : w.layers) {
    layer.input_layernorm = make_f32({D}, 1.f);
    layer.pre_ff_layer_norm = make_f32({D}, 1.f);

    layer.ff_up_proj = make_f32({F, D}, 0.01f);
    layer.ff_gate_proj = make_f32({F, D}, 0.01f);
    layer.ff_down_proj = make_f32({D, F}, 0.01f);

    layer.in_proj = make_f32({proj_size, D}, 0.01f);
    layer.out_proj = make_f32({D, I}, 0.01f);
    layer.conv1d = make_f32({conv_dim, K}, 0.f);
    {
      auto cw = as_mat_f32(layer.conv1d);
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

    layer.q_proj = make_f32({q_dim, D}, 0.01f);
    layer.k_proj = make_f32({kv_dim, D}, 0.01f);
    layer.v_proj = make_f32({kv_dim, D}, 0.01f);
    layer.o_proj = make_f32({D, q_dim}, 0.01f);
  }
  return w;
}

StatusOr<std::unique_ptr<FalconH1Runner>> make_runner(const ModelConfig& cfg, int n_slots,
                                                       std::unique_ptr<DeviceAllocator>* out_alloc,
                                                       std::unique_ptr<CachePool>* out_pool) {
  std::unique_ptr<DeviceAllocator> alloc;
  ASSIGN_OR_RETURN(alloc, create_device_allocator(Device::CPU, 0));

  FalconH1Weights weights;
  {
    AllocatorScope scope(alloc.get());
    weights = make_weights(cfg);
  }

  std::unique_ptr<CachePool> pool;
  ASSIGN_OR_RETURN(pool, create_cache_pool(cfg, alloc.get(), n_slots, create_cache_layout));
  auto runner = std::make_unique<FalconH1Runner>(cfg, std::move(weights), cpu_ops());
  *out_alloc = std::move(alloc);
  *out_pool = std::move(pool);
  return runner;
}

} // namespace

TEST(FalconH1CacheLayout, HybridPacksMambaAndKv) {
  ModelConfig cfg = make_cfg();
  StatusOr<CacheLayout> layout_or = create_cache_layout(cfg);
  ASSERT_TRUE(layout_or.ok()) << layout_or.status().message();
  const CacheLayout& layout = layout_or.value();

  EXPECT_EQ(layout.num_layers, 1);
  ASSERT_EQ(layout.layers.size(), 1u);
  EXPECT_EQ(layout.layers[0].kind, LayerCacheKind::Hybrid);
  EXPECT_GT(layout.layers[0].conv.bytes, 0u);
  EXPECT_GT(layout.layers[0].ssm.bytes, 0u);
  EXPECT_GT(layout.layers[0].k.bytes, 0u);
  EXPECT_GT(layout.layers[0].v.bytes, 0u);
  EXPECT_EQ(layout.layers[0].k.bytes, layout.layers[0].v.bytes);

  const size_t kv_dim =
      static_cast<size_t>(cfg.attn->n_kv_heads) * static_cast<size_t>(cfg.attn->head_dim);
  EXPECT_EQ(layout.layers[0].k.bytes,
            static_cast<size_t>(cfg.max_seq_length) * kv_dim * sizeof(float));
}

TEST(FalconH1CacheLayout, RejectsNonDivisibleGqa) {
  ModelConfig cfg = make_cfg();
  cfg.attn->n_q_heads = 3;
  cfg.attn->n_kv_heads = 2;
  StatusOr<CacheLayout> layout_or = create_cache_layout(cfg);
  EXPECT_FALSE(layout_or.ok());
}

TEST(FalconH1CachePool, SeqLenAdvancesAndResetClears) {
  ModelConfig cfg = make_cfg();
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  auto pool_or = create_cache_pool(cfg, alloc_or.value().get(), 1, create_cache_layout);
  ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();
  CachePool& pool = *pool_or.value();

  StatusOr<CacheHandle> h = pool.acquire();
  ASSERT_TRUE(h.ok());
  StatusOr<int64_t> len0 = pool.seq_len(h.value());
  ASSERT_TRUE(len0.ok());
  EXPECT_EQ(len0.value(), 0);

  ASSERT_TRUE(pool.set_seq_len(h.value(), 5).ok());
  EXPECT_EQ(pool.seq_len(h.value()).value(), 5);

  CacheHandle handle = h.value();
  ASSERT_TRUE(pool.reset(handle).ok());
  EXPECT_EQ(pool.seq_len(handle).value(), 0);

  ASSERT_TRUE(pool.release(handle).ok());
}

TEST(FalconH1CachePool, LayerViewsExposeKv) {
  ModelConfig cfg = make_cfg();
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  auto pool_or = create_cache_pool(cfg, alloc_or.value().get(), 1, create_cache_layout);
  ASSERT_TRUE(pool_or.ok());
  CachePool& pool = *pool_or.value();

  StatusOr<CacheHandle> h = pool.acquire();
  ASSERT_TRUE(h.ok());
  StatusOr<LayerCacheView> view = pool.layer_view(h.value(), 0);
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view.value().kind, LayerCacheKind::Hybrid);
  EXPECT_NE(view.value().k.ptr, nullptr);
  EXPECT_NE(view.value().v.ptr, nullptr);
  EXPECT_GT(view.value().k.bytes, 0u);
  EXPECT_GT(view.value().v.bytes, 0u);

  CacheHandle handle = h.value();
  ASSERT_TRUE(pool.release(handle).ok());
}

TEST(FalconH1Attn, PrefillWritesKvAndDecodeAppends) {
  ModelConfig cfg = make_cfg();
  auto alloc_or = create_device_allocator(Device::CPU, 0);
  ASSERT_TRUE(alloc_or.ok());
  AllocatorScope scope(alloc_or.value().get());

  FalconH1Weights weights = make_weights(cfg);
  const FalconH1LayerWeights& layer = weights.layers[0];
  AttnWeights aw{.q_proj = layer.q_proj,
                 .k_proj = layer.k_proj,
                 .v_proj = layer.v_proj,
                 .o_proj = layer.o_proj};

  auto pool_or = create_cache_pool(cfg, alloc_or.value().get(), 1, create_cache_layout);
  ASSERT_TRUE(pool_or.ok());
  StatusOr<CacheHandle> h = pool_or.value()->acquire();
  ASSERT_TRUE(h.ok());
  StatusOr<LayerCacheView> view = pool_or.value()->layer_view(h.value(), 0);
  ASSERT_TRUE(view.ok());

  const int64_t D = cfg.hidden_size;
  const int64_t T = 3;
  StatusOr<Tensor> hidden = allocate_f32_tensor({T, D});
  ASSERT_TRUE(hidden.ok());
  as_mat_f32(hidden.value()).setConstant(0.1f);

  StatusOr<Tensor> out = allocate_f32_tensor({T, D});
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(cpu_ops()
                  .attention_f32(hidden.value(), aw, *cfg.attn, cfg.scales ? &*cfg.scales : nullptr,
                                 cfg.max_seq_length, cfg.hidden_size, view.value(),
                                 /*past_len=*/0, out.value())
                  .ok());

  const int64_t kv_dim = static_cast<int64_t>(cfg.attn->n_kv_heads) * cfg.attn->head_dim;
  auto* k = static_cast<float*>(view.value().k.ptr);
  // First T rows should be non-zero after write; later rows still zero.
  float sum_written = 0.f;
  for (int64_t i = 0; i < T * kv_dim; ++i)
    sum_written += std::fabs(k[i]);
  EXPECT_GT(sum_written, 0.f);

  float sum_tail = 0.f;
  for (int64_t i = T * kv_dim; i < (T + 1) * kv_dim; ++i)
    sum_tail += std::fabs(k[i]);
  EXPECT_FLOAT_EQ(sum_tail, 0.f);

  StatusOr<Tensor> one = allocate_f32_tensor({D});
  ASSERT_TRUE(one.ok());
  as_vec_f32(one.value()).setConstant(0.1f);
  StatusOr<Tensor> out1 = allocate_f32_tensor({D});
  ASSERT_TRUE(out1.ok());
  ASSERT_TRUE(cpu_ops()
                  .attention_f32(one.value(), aw, *cfg.attn, cfg.scales ? &*cfg.scales : nullptr,
                                 cfg.max_seq_length, cfg.hidden_size, view.value(), /*past_len=*/T,
                                 out1.value())
                  .ok());

  float sum_new = 0.f;
  for (int64_t i = T * kv_dim; i < (T + 1) * kv_dim; ++i)
    sum_new += std::fabs(k[i]);
  EXPECT_GT(sum_new, 0.f);

  CacheHandle handle = h.value();
  ASSERT_TRUE(pool_or.value()->release(handle).ok());
}

TEST(FalconH1Runner, PrefillReturnsVocabLogits) {
  ModelConfig cfg = make_cfg();
  std::unique_ptr<DeviceAllocator> alloc;
  std::unique_ptr<CachePool> pool;
  StatusOr<std::unique_ptr<FalconH1Runner>> runner_or = make_runner(cfg, /*n_slots=*/2, &alloc, &pool);
  ASSERT_TRUE(runner_or.ok()) << runner_or.status().message();
  FalconH1Runner& runner = *runner_or.value();

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

TEST(FalconH1Runner, PrefillThenDecode) {
  ModelConfig cfg = make_cfg();
  std::unique_ptr<DeviceAllocator> alloc;
  std::unique_ptr<CachePool> pool;
  StatusOr<std::unique_ptr<FalconH1Runner>> runner_or = make_runner(cfg, /*n_slots=*/2, &alloc, &pool);
  ASSERT_TRUE(runner_or.ok()) << runner_or.status().message();
  FalconH1Runner& runner = *runner_or.value();

  StatusOr<CacheHandle> handle_or = pool->acquire();
  ASSERT_TRUE(handle_or.ok()) << handle_or.status().message();
  CacheHandle handle = std::move(handle_or.value());
  StatusOr<std::span<LayerCacheView>> layers = pool->layer_views(handle);
  ASSERT_TRUE(layers.ok()) << layers.status().message();

  const int32_t prompt[] = {1, 2};
  StatusOr<Tensor> d0 = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc.get());
    StatusOr<Tensor> pref = runner.prefill(prompt, layers.value());
    if (!pref.ok())
      return pref.status();
    if (Status s = pool->set_seq_len(handle, 2); !s.ok())
      return s;
    return runner.decode(/*token=*/3, layers.value());
  }();
  ASSERT_TRUE(d0.ok()) << d0.status().message();
  ASSERT_EQ(d0.value().shape.size(), 1u);
  EXPECT_EQ(d0.value().shape[0], cfg.vocab_size);

  StatusOr<Tensor> d1 = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc.get());
    return runner.decode(/*token=*/4, layers.value());
  }();
  ASSERT_TRUE(d1.ok()) << d1.status().message();

  ASSERT_TRUE(pool->release(handle).ok());
}

TEST(FalconH1Runner, PrefillEmptyTokensFails) {
  ModelConfig cfg = make_cfg();
  std::unique_ptr<DeviceAllocator> alloc;
  std::unique_ptr<CachePool> pool;
  StatusOr<std::unique_ptr<FalconH1Runner>> runner_or = make_runner(cfg, 1, &alloc, &pool);
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

TEST(FalconH1Runner, RegistryLoadsRealCheckpointPrefillDecode) {
  auto entry_or = ModelRegistry::open(MAMBA_TEST_FALCON_DIR, /*max_seq_length=*/64);
  ASSERT_TRUE(entry_or.ok()) << entry_or.status().message();
  ModelEntry entry = std::move(entry_or.value());
  ASSERT_NE(entry.cfg, nullptr);

  std::unique_ptr<DeviceAllocator> alloc;
  auto alloc_or = create_device_allocator(Device::CPU, /*device_id=*/0);
  ASSERT_TRUE(alloc_or.ok()) << alloc_or.status().message();
  alloc = std::move(alloc_or.value());

  auto runner_or = entry.create_runner(*alloc);
  ASSERT_TRUE(runner_or.ok()) << runner_or.status().message();

  std::unique_ptr<CachePool> pool;
  auto pool_or = create_cache_pool(*entry.cfg, alloc.get(), /*num_slots=*/1, create_cache_layout);
  ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();
  pool = std::move(pool_or.value());

  StatusOr<CacheHandle> handle_or = pool->acquire();
  ASSERT_TRUE(handle_or.ok()) << handle_or.status().message();
  CacheHandle handle = std::move(handle_or.value());
  StatusOr<std::span<LayerCacheView>> layers = pool->layer_views(handle);
  ASSERT_TRUE(layers.ok()) << layers.status().message();

  const int32_t tokens[] = {1, 2};
  StatusOr<Tensor> pref = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc.get());
    return runner_or.value()->prefill(tokens, layers.value());
  }();
  ASSERT_TRUE(pref.ok()) << pref.status().message();
  ASSERT_EQ(pref.value().shape.size(), 1u);
  EXPECT_EQ(pref.value().shape[0], entry.cfg->vocab_size);

  ASSERT_TRUE(pool->set_seq_len(handle, 2).ok());
  StatusOr<Tensor> dec = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc.get());
    return runner_or.value()->decode(/*token=*/3, layers.value());
  }();
  ASSERT_TRUE(dec.ok()) << dec.status().message();
  EXPECT_EQ(dec.value().shape[0], entry.cfg->vocab_size);

  ASSERT_TRUE(pool->release(handle).ok());
}
