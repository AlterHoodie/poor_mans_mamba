#include "io/config.h"
#include "ops/backend.h"
#include "ops/gpu/gpu_test_utils.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace gpu_test;

namespace {

ModelConfig make_mixer_cfg() {
  ModelConfig cfg{};
  cfg.layout = ArchLayout::MambaOnly;
  cfg.model_type = "mamba2";
  cfg.hidden_size = 8;
  cfg.vocab_size = 8;
  cfg.num_hidden_layers = 1;
  cfg.max_seq_length = 32;
  cfg.rms_norm_eps = 1e-5f;

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
  cfg.scales = scales;
  return cfg;
}

struct MixerHostWeights {
  std::vector<float> hidden;
  std::vector<float> in_proj;
  std::vector<float> out_proj;
  std::vector<float> conv1d;
  std::vector<float> conv1d_bias;
  std::vector<float> dt_bias;
  std::vector<float> A_log;
  std::vector<float> D;
};

MixerHostWeights make_host_weights(const ModelConfig& cfg, int64_t T) {
  const int64_t D = cfg.hidden_size;
  const int64_t I = cfg.ssm.d_inner;
  const int64_t G = cfg.ssm.n_groups;
  const int64_t N = cfg.ssm.d_state;
  const int64_t H = cfg.ssm.n_heads;
  const int64_t K = cfg.ssm.d_conv;
  const int64_t conv_dim = I + 2 * G * N;
  const int64_t proj_size = I + conv_dim + H;

  MixerHostWeights w;
  w.hidden.assign(static_cast<size_t>(T * D), 0.1f);
  w.in_proj.assign(static_cast<size_t>(proj_size * D), 0.01f);
  w.out_proj.assign(static_cast<size_t>(D * I), 0.01f);
  w.conv1d.assign(static_cast<size_t>(conv_dim * K), 0.f);
  for (int64_t c = 0; c < conv_dim; ++c)
    w.conv1d[static_cast<size_t>(c * K + (K - 1))] = 1.f;
  w.conv1d_bias.assign(static_cast<size_t>(conv_dim), 0.f);
  w.dt_bias.assign(static_cast<size_t>(H), 0.f);
  w.A_log.resize(static_cast<size_t>(H));
  for (int64_t h = 0; h < H; ++h)
    w.A_log[static_cast<size_t>(h)] = std::log(static_cast<float>(h + 1));
  w.D.assign(static_cast<size_t>(H), 1.f);
  return w;
}

} // namespace

TEST_F(GpuTest, Mamba2MixerMatchesCpu) {
  ModelConfig cfg = make_mixer_cfg();
  const int64_t T = 3;
  const int64_t D = cfg.hidden_size;
  const int64_t I = cfg.ssm.d_inner;
  const int64_t G = cfg.ssm.n_groups;
  const int64_t N = cfg.ssm.d_state;
  const int64_t H = cfg.ssm.n_heads;
  const int64_t K = cfg.ssm.d_conv;
  const int64_t conv_dim = I + 2 * G * N;
  const int64_t proj_size = I + conv_dim + H;

  MixerHostWeights hw = make_host_weights(cfg, T);

  Tensor cpu_out;
  std::vector<float> cpu_conv;
  std::vector<float> cpu_ssm;
  {
    AllocatorScope scope(cpu_alloc_.get());
    Tensor hidden = make_cpu_f32({T, D}, hw.hidden);
    Tensor in_proj = make_cpu_f32({proj_size, D}, hw.in_proj);
    Tensor out_proj = make_cpu_f32({D, I}, hw.out_proj);
    Tensor conv1d = make_cpu_f32({conv_dim, K}, hw.conv1d);
    Tensor conv1d_bias = make_cpu_f32({conv_dim}, hw.conv1d_bias);
    Tensor dt_bias = make_cpu_f32({H}, hw.dt_bias);
    Tensor A_log = make_cpu_f32({H}, hw.A_log);
    Tensor D_t = make_cpu_f32({H}, hw.D);
    cpu_out = make_cpu_f32({T, D}, std::vector<float>(static_cast<size_t>(T * D), 0.f));

    auto pool_or = create_cache_pool(cfg, cpu_alloc_.get(), 1, create_cache_layout);
    ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();
    StatusOr<CacheHandle> h = pool_or.value()->acquire();
    ASSERT_TRUE(h.ok());
    StatusOr<LayerCacheView> view = pool_or.value()->layer_view(h.value(), 0);
    ASSERT_TRUE(view.ok());

    Mamba2MixerWeights mw{.in_proj = in_proj,
                          .conv1d = conv1d,
                          .conv1d_bias = conv1d_bias,
                          .dt_bias = dt_bias,
                          .out_proj = out_proj,
                          .A_log = A_log,
                          .D = D_t,
                          .mixer_norm = nullptr};
    ASSERT_TRUE(cpu_ops()
                    .mamba2_mixer_f32(hidden, mw, cfg.ssm, cfg.scales ? &*cfg.scales : nullptr,
                                      cfg.rms_norm_eps, view.value(), cpu_out)
                    .ok());

    cpu_conv.assign(static_cast<const float*>(view.value().conv.ptr),
                    static_cast<const float*>(view.value().conv.ptr) +
                        view.value().conv.bytes / sizeof(float));
    cpu_ssm.assign(static_cast<const float*>(view.value().ssm.ptr),
                   static_cast<const float*>(view.value().ssm.ptr) +
                       view.value().ssm.bytes / sizeof(float));

    CacheHandle handle = h.value();
    ASSERT_TRUE(pool_or.value()->release(handle).ok());
  }

  std::vector<float> gpu_out_h;
  std::vector<float> gpu_conv;
  std::vector<float> gpu_ssm;
  {
    AllocatorScope scope(gpu_alloc_.get());
    Tensor hidden = make_gpu_f32({T, D}, hw.hidden);
    Tensor in_proj = make_gpu_f32({proj_size, D}, hw.in_proj);
    Tensor out_proj = make_gpu_f32({D, I}, hw.out_proj);
    Tensor conv1d = make_gpu_f32({conv_dim, K}, hw.conv1d);
    Tensor conv1d_bias = make_gpu_f32({conv_dim}, hw.conv1d_bias);
    Tensor dt_bias = make_gpu_f32({H}, hw.dt_bias);
    Tensor A_log = make_gpu_f32({H}, hw.A_log);
    Tensor D_t = make_gpu_f32({H}, hw.D);
    Tensor out = make_gpu_f32({T, D}, std::vector<float>(static_cast<size_t>(T * D), 0.f));

    auto pool_or = create_cache_pool(cfg, gpu_alloc_.get(), 1, create_cache_layout);
    ASSERT_TRUE(pool_or.ok()) << pool_or.status().message();
    StatusOr<CacheHandle> h = pool_or.value()->acquire();
    ASSERT_TRUE(h.ok());
    StatusOr<LayerCacheView> view = pool_or.value()->layer_view(h.value(), 0);
    ASSERT_TRUE(view.ok());

    Mamba2MixerWeights mw{.in_proj = in_proj,
                          .conv1d = conv1d,
                          .conv1d_bias = conv1d_bias,
                          .dt_bias = dt_bias,
                          .out_proj = out_proj,
                          .A_log = A_log,
                          .D = D_t,
                          .mixer_norm = nullptr};
    ASSERT_TRUE(cuda_ops()
                    .mamba2_mixer_f32(hidden, mw, cfg.ssm, cfg.scales ? &*cfg.scales : nullptr,
                                      cfg.rms_norm_eps, view.value(), out)
                    .ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gpu_out_h = download_f32(out);

    gpu_conv.resize(view.value().conv.bytes / sizeof(float));
    ASSERT_EQ(cudaMemcpy(gpu_conv.data(), view.value().conv.ptr, view.value().conv.bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    gpu_ssm.resize(view.value().ssm.bytes / sizeof(float));
    ASSERT_EQ(cudaMemcpy(gpu_ssm.data(), view.value().ssm.ptr, view.value().ssm.bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    CacheHandle handle = h.value();
    ASSERT_TRUE(pool_or.value()->release(handle).ok());
  }

  expect_close(gpu_out_h, cpu_to_vec(cpu_out), 1e-3f);
  expect_close(gpu_conv, cpu_conv, 1e-3f);
  expect_close(gpu_ssm, cpu_ssm, 1e-3f);
}
