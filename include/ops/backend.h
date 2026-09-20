#pragma once

#include "core/device.h"
#include "core/status.h"
#include "core/tensor.h"
#include "io/config.h"
#include "runtime/cache/cache_layout.h"

#include <cstdint>

struct AttnWeights {
  const Tensor& q_proj;
  const Tensor& k_proj;
  const Tensor& v_proj;
  const Tensor& o_proj;
};

// Weight view for the shared Mamba2 mixer (refs into model-specific layer bags).
struct Mamba2MixerWeights {
  const Tensor& in_proj;
  const Tensor& conv1d;
  const Tensor& conv1d_bias;
  const Tensor& dt_bias;
  const Tensor& out_proj;
  const Tensor& A_log;
  const Tensor& D;
  const Tensor* mixer_norm = nullptr; // required when gated_rms_norm
};

// Weight view for SwiGLU MLP (refs into model-specific layer bags).
struct MlpWeights {
  const Tensor& up_proj;
  const Tensor& gate_proj;
  const Tensor& down_proj;
};

class OpsBackend {
protected:
  Device device_ = Device::NA;

public:
  explicit OpsBackend(Device device) : device_(device) {}
  virtual ~OpsBackend() = default;

  Device device() const { return device_; }

  // blocks
  virtual Status attention_f32(const Tensor& normed_hidden, const AttnWeights& w,
                               const AttnConfig& attn, const ScaleConfig* scales,
                               int max_seq_length, int hidden_size, LayerCacheView& cache,
                               int64_t past_len, Tensor& out) const = 0;

  virtual Status mamba2_mixer_f32(const Tensor& normed_hidden, const Mamba2MixerWeights& w,
                                  const SsmConfig& ssm, const ScaleConfig* scales,
                                  float rms_norm_eps, LayerCacheView& cache, Tensor& out) const = 0;

  virtual Status mlp_f32(const Tensor& normed_hidden, const MlpWeights& w, const MlpConfig& mlp,
                         const ScaleConfig* scales, Tensor& out) const = 0;

  // glue
  virtual Status rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) const = 0;
  virtual Status add(const Tensor& a, const Tensor& b, Tensor& out) const = 0;
  virtual Status scale(Tensor& x, float s) const = 0;

  // y = x @ W^T  with W stored as [N, K]
  virtual Status linear(const Tensor& x, const Tensor& W, Tensor& out) const = 0;

  // out [T, H] = scale * table[tokens[t]]
  virtual Status embedding_lookup(const Tensor& table, const int32_t* tokens, int64_t n_tokens,
                                  float scale, Tensor& out) const = 0;

  // rank-1: copy hidden into out; rank-2 [T,H]: copy last row into out {H}
  virtual Status take_last_row(const Tensor& hidden, Tensor& out) const = 0;
};

const OpsBackend& cpu_ops();
#ifdef MAMBASERVE_WITH_CUDA
const OpsBackend& cuda_ops();
#endif
const OpsBackend& ops_for(Device d);
