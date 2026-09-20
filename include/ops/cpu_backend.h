#pragma once

#include "ops/backend.h"

class CPUBackend final : public OpsBackend {
private:
  inline float* kv_row_(float* base, int64_t row, int64_t kv_dim) const {
    return base + static_cast<size_t>(row) * static_cast<size_t>(kv_dim);
  }

  inline const float* kv_row_(const float* base, int64_t row, int64_t kv_dim) const {
    return base + static_cast<size_t>(row) * static_cast<size_t>(kv_dim);
  }

  void apply_rope_inplace_(float* q, float* k, int64_t T, int Hq, int Hkv, int Dh, int64_t past_len,
                           float rope_theta) const;

  Status mamba2_ssm_scan_f32_(const float* x, const float* B, const float* C, const float* dt,
                              const float* A_log, const float* D, const float* dt_bias,
                              float* state, float* y, int64_t T, const SsmConfig& ssm) const;

  // Gated RMSNorm used by Mamba2 mixer: y = rms(y * silu(gate)) * weight
  Status rms_norm_gated_f32_(Tensor& y, const Tensor& gate, const Tensor& weight, float eps) const;

  Status causal_conv1d_f32_(const float* x, int64_t T, int64_t conv_dim, int kernel,
                            const float* weight, const float* bias, float* cache, float* y,
                            bool apply_silu) const;

public:
  CPUBackend() : OpsBackend(Device::CPU) {}

  Status attention_f32(const Tensor& normed_hidden, const AttnWeights& w, const AttnConfig& attn,
                       const ScaleConfig* scales, int max_seq_length, int hidden_size,
                       LayerCacheView& cache, int64_t past_len, Tensor& out) const override;

  Status mamba2_mixer_f32(const Tensor& normed_hidden, const Mamba2MixerWeights& w,
                          const SsmConfig& ssm, const ScaleConfig* scales, float rms_norm_eps,
                          LayerCacheView& cache, Tensor& out) const override;

  Status mlp_f32(const Tensor& normed_hidden, const MlpWeights& w, const MlpConfig& mlp,
                 const ScaleConfig* scales, Tensor& out) const override;

  Status rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) const override;

  Status add(const Tensor& a, const Tensor& b, Tensor& out) const override;

  Status scale(Tensor& x, float s) const override;

  Status linear(const Tensor& x, const Tensor& W, Tensor& out) const override;

  Status embedding_lookup(const Tensor& table, const int32_t* tokens, int64_t n_tokens, float scale,
                          Tensor& out) const override;

  Status take_last_row(const Tensor& hidden, Tensor& out) const override;
};
