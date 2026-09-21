#include "io/config.h"
#include "ops/backend.h"
#include "ops/gpu/gpu_test_utils.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"

#include <gtest/gtest.h>

#include <vector>

using namespace gpu_test;

namespace {

ModelConfig make_attn_cfg() {
  ModelConfig cfg{};
  cfg.layout = ArchLayout::ParallelHybrid;
  cfg.model_type = "falcon_h1";
  cfg.hidden_size = 8;
  cfg.vocab_size = 8;
  cfg.num_hidden_layers = 1;
  cfg.max_seq_length = 32;
  cfg.rms_norm_eps = 1e-5f;

  AttnConfig attn{};
  attn.n_q_heads = 4;
  attn.n_kv_heads = 2;
  attn.head_dim = 4;
  attn.rope_theta = 10000.f;
  cfg.attn = attn;

  // Hybrid layout also needs SSM dims for cache packing.
  cfg.ssm.n_heads = 2;
  cfg.ssm.d_head = 4;
  cfg.ssm.d_inner = 8;
  cfg.ssm.d_state = 4;
  cfg.ssm.n_groups = 1;
  cfg.ssm.d_conv = 4;
  cfg.ssm.use_conv_bias = true;
  cfg.ssm.gated_rms_norm = false;

  ScaleConfig scales{};
  scales.attn_in = 1.f;
  scales.attn_out = 1.f;
  scales.key = 1.f;
  cfg.scales = scales;

  MlpConfig mlp{};
  mlp.intermediate_size = 16;
  cfg.mlp = mlp;
  return cfg;
}

struct AttnHostWeights {
  std::vector<float> hidden;
  std::vector<float> q_proj;
  std::vector<float> k_proj;
  std::vector<float> v_proj;
  std::vector<float> o_proj;
};

AttnHostWeights make_host_weights(const ModelConfig& cfg, int64_t T) {
  const int64_t D = cfg.hidden_size;
  const int64_t q_dim = static_cast<int64_t>(cfg.attn->n_q_heads) * cfg.attn->head_dim;
  const int64_t kv_dim = static_cast<int64_t>(cfg.attn->n_kv_heads) * cfg.attn->head_dim;

  AttnHostWeights w;
  w.hidden.assign(static_cast<size_t>(T * D), 0.1f);
  w.q_proj.assign(static_cast<size_t>(q_dim * D), 0.01f);
  w.k_proj.assign(static_cast<size_t>(kv_dim * D), 0.01f);
  w.v_proj.assign(static_cast<size_t>(kv_dim * D), 0.01f);
  w.o_proj.assign(static_cast<size_t>(D * q_dim), 0.01f);
  return w;
}

} // namespace

TEST_F(GpuTest, AttentionPrefillMatchesCpu) {
  ModelConfig cfg = make_attn_cfg();
  const int64_t T = 3;
  const int64_t D = cfg.hidden_size;
  const int64_t q_dim = static_cast<int64_t>(cfg.attn->n_q_heads) * cfg.attn->head_dim;
  const int64_t kv_dim = static_cast<int64_t>(cfg.attn->n_kv_heads) * cfg.attn->head_dim;

  AttnHostWeights hw = make_host_weights(cfg, T);

  Tensor cpu_out;
  std::vector<float> cpu_k;
  std::vector<float> cpu_v;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor hidden = make_cpu_f32({T, D}, hw.hidden);
    Tensor q_proj = make_cpu_f32({q_dim, D}, hw.q_proj);
    Tensor k_proj = make_cpu_f32({kv_dim, D}, hw.k_proj);
    Tensor v_proj = make_cpu_f32({kv_dim, D}, hw.v_proj);
    Tensor o_proj = make_cpu_f32({D, q_dim}, hw.o_proj);
    cpu_out = make_cpu_f32({T, D}, std::vector<float>(static_cast<size_t>(T * D), 0.f));

    auto pool_or = create_cache_pool(cfg, cpu_alloc_.get(), 1, create_cache_layout);
    ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();
    StatusOr<CacheHandle> h = pool_or.value()->acquire();
    ASSERT_TRUE(h.ok());
    StatusOr<LayerCacheView> view = pool_or.value()->layer_view(h.value(), 0);
    ASSERT_TRUE(view.ok());

    AttnWeights aw{.q_proj = q_proj, .k_proj = k_proj, .v_proj = v_proj, .o_proj = o_proj};
    ASSERT_TRUE(cpu_ops()
                    .attention_f32(hidden, aw, *cfg.attn, cfg.scales ? &*cfg.scales : nullptr,
                                   cfg.max_seq_length, cfg.hidden_size, view.value(),
                                   /*past_len=*/0, cpu_out)
                    .ok());

    cpu_k.assign(static_cast<const float*>(view.value().k.ptr),
                 static_cast<const float*>(view.value().k.ptr) + view.value().k.bytes / sizeof(float));
    cpu_v.assign(static_cast<const float*>(view.value().v.ptr),
                 static_cast<const float*>(view.value().v.ptr) + view.value().v.bytes / sizeof(float));

    CacheHandle handle = h.value();
    ASSERT_TRUE(pool_or.value()->release(handle).ok());
  }

  std::vector<float> gpu_out_h;
  std::vector<float> gpu_k;
  std::vector<float> gpu_v;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor hidden = make_gpu_f32({T, D}, hw.hidden);
    Tensor q_proj = make_gpu_f32({q_dim, D}, hw.q_proj);
    Tensor k_proj = make_gpu_f32({kv_dim, D}, hw.k_proj);
    Tensor v_proj = make_gpu_f32({kv_dim, D}, hw.v_proj);
    Tensor o_proj = make_gpu_f32({D, q_dim}, hw.o_proj);
    Tensor out = make_gpu_f32({T, D}, std::vector<float>(static_cast<size_t>(T * D), 0.f));

    auto pool_or = create_cache_pool(cfg, gpu_alloc_.get(), 1, create_cache_layout);
    ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();
    StatusOr<CacheHandle> h = pool_or.value()->acquire();
    ASSERT_TRUE(h.ok());
    StatusOr<LayerCacheView> view = pool_or.value()->layer_view(h.value(), 0);
    ASSERT_TRUE(view.ok());

    AttnWeights aw{.q_proj = q_proj, .k_proj = k_proj, .v_proj = v_proj, .o_proj = o_proj};
    ASSERT_TRUE(cuda_ops()
                    .attention_f32(hidden, aw, *cfg.attn, cfg.scales ? &*cfg.scales : nullptr,
                                   cfg.max_seq_length, cfg.hidden_size, view.value(),
                                   /*past_len=*/0, out)
                    .ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_out_h = download_f32(out);

    gpu_k.resize(view.value().k.bytes / sizeof(float));
    ASSERT_EQ(cudaMemcpy(gpu_k.data(), view.value().k.ptr, view.value().k.bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    gpu_v.resize(view.value().v.bytes / sizeof(float));
    ASSERT_EQ(cudaMemcpy(gpu_v.data(), view.value().v.ptr, view.value().v.bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    CacheHandle handle = h.value();
    ASSERT_TRUE(pool_or.value()->release(handle).ok());
  }

  expect_close(gpu_out_h, cpu_to_vec(cpu_out), 1e-3f);
  expect_close(gpu_k, cpu_k, 1e-3f);
  expect_close(gpu_v, cpu_v, 1e-3f);
}
