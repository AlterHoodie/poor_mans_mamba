#include "runtime/runner/falconh1_runner.h"

#include "ops/common/shapes.h"
#include "ops/common/tensor_checks.h"
#include "runtime/cache/cache_layout.h"

#include <cstring>
#include <utility>
#include <vector>

FalconH1Runner::~FalconH1Runner() = default;

FalconH1Runner::FalconH1Runner(ModelConfig cfg, FalconH1Weights weights, const OpsBackend& ops)
    : ops_(&ops), cfg_(std::move(cfg)), weights_(std::move(weights)) {}

Status FalconH1Runner::block_forward_(int layer_idx, LayerCacheView& cache, Tensor& hidden,
                                      int64_t past_len) {
  if (cache.kind != LayerCacheKind::Hybrid)
    return Status::InvalidArgument("cache kind must be hybrid");
  if (layer_idx < 0 || layer_idx >= static_cast<int>(weights_.layers.size()))
    return Status::InvalidArgument("layer_idx out of range");
  if (!cfg_.attn.has_value() || !cfg_.mlp.has_value())
    return Status::InvalidArgument("ParallelHybrid requires attn and mlp config");

  const FalconH1LayerWeights& layer = weights_.layers[static_cast<size_t>(layer_idx)];
  const ScaleConfig* scales = cfg_.scales ? &*cfg_.scales : nullptr;
  const float ssm_out = scales ? scales->ssm_out : 1.f;
  const float attn_in = scales ? scales->attn_in : 1.f;
  const float attn_out_scale = scales ? scales->attn_out : 1.f;

  StatusOr<Tensor> normed = allocate_f32_tensor(hidden.shape);
  if (!normed.ok())
    return Status(normed.status());
  if (Status s = ops_->rms_norm(hidden, layer.input_layernorm, cfg_.rms_norm_eps, normed.value());
      !s.ok())
    return s;

  StatusOr<Tensor> mamba_out = allocate_f32_tensor(hidden.shape);
  StatusOr<Tensor> attn_out = allocate_f32_tensor(hidden.shape);
  if (!mamba_out.ok())
    return Status(mamba_out.status());
  if (!attn_out.ok())
    return Status(attn_out.status());

  Mamba2MixerWeights mw{.in_proj = layer.in_proj,
                        .conv1d = layer.conv1d,
                        .conv1d_bias = layer.conv1d_bias,
                        .dt_bias = layer.dt_bias,
                        .out_proj = layer.out_proj,
                        .A_log = layer.A_log,
                        .D = layer.D,
                        .mixer_norm = nullptr};
  if (Status s = ops_->mamba2_mixer_f32(normed.value(), mw, cfg_.ssm, scales, cfg_.rms_norm_eps,
                                        cache, mamba_out.value());
      !s.ok()) {
    return s;
  }
  if (Status s = ops_->scale(mamba_out.value(), ssm_out); !s.ok())
    return s;

  StatusOr<Tensor> attn_in_t = allocate_f32_tensor(normed.value().shape);
  if (!attn_in_t.ok())
    return Status(attn_in_t.status());
  {
    // attn_in_t = normed * attn_in
    std::memcpy(attn_in_t.value().buffer.ptr, normed.value().buffer.ptr,
                normed.value().buffer.bytes);
    if (Status s = ops_->scale(attn_in_t.value(), attn_in); !s.ok())
      return s;
  }
  AttnWeights aw{.q_proj = layer.q_proj,
                 .k_proj = layer.k_proj,
                 .v_proj = layer.v_proj,
                 .o_proj = layer.o_proj};
  if (Status s = ops_->attention_f32(attn_in_t.value(), aw, *cfg_.attn, scales, cfg_.max_seq_length,
                                     cfg_.hidden_size, cache, past_len, attn_out.value());
      !s.ok()) {
    return s;
  }
  if (Status s = ops_->scale(attn_out.value(), attn_out_scale); !s.ok())
    return s;

  if (Status s = ops_->add(hidden, mamba_out.value(), hidden); !s.ok())
    return s;
  if (Status s = ops_->add(hidden, attn_out.value(), hidden); !s.ok())
    return s;

  StatusOr<Tensor> ff_normed = allocate_f32_tensor(hidden.shape);
  if (!ff_normed.ok())
    return Status(ff_normed.status());
  if (Status s =
          ops_->rms_norm(hidden, layer.pre_ff_layer_norm, cfg_.rms_norm_eps, ff_normed.value());
      !s.ok())
    return s;

  StatusOr<Tensor> ff_out = allocate_f32_tensor(hidden.shape);
  if (!ff_out.ok())
    return Status(ff_out.status());
  MlpWeights mw_mlp{.up_proj = layer.ff_up_proj,
                    .gate_proj = layer.ff_gate_proj,
                    .down_proj = layer.ff_down_proj};
  if (Status s = ops_->mlp_f32(ff_normed.value(), mw_mlp, *cfg_.mlp, scales, ff_out.value());
      !s.ok())
    return s;

  return ops_->add(hidden, ff_out.value(), hidden);
}

StatusOr<Tensor> FalconH1Runner::embed_(std::span<const int32_t> tokens) {
  if (tokens.empty())
    return Status::InvalidArgument("embed requires at least one token");
  if (weights_.embeddings.size() != 2)
    return Status::InvalidArgument("embeddings must be rank 2 [vocab, hidden]");

  const int64_t hidden = weights_.embeddings.shape[1];
  if (hidden != cfg_.hidden_size)
    return Status::InvalidArgument("embeddings hidden dim mismatch with config");

  const float emb_scale = cfg_.scales ? cfg_.scales->embedding : 1.f;

  StatusOr<Tensor> out = allocate_f32_tensor({static_cast<int64_t>(tokens.size()), hidden});
  if (!out.ok())
    return Status(out.status());

  if (Status s = ops_->embedding_lookup(weights_.embeddings, tokens.data(),
                                        static_cast<int64_t>(tokens.size()), emb_scale, out.value());
      !s.ok()) {
    return s;
  }
  return std::move(out.value());
}

Status FalconH1Runner::norm_f_(Tensor& hidden) {
  return ops_->rms_norm(hidden, weights_.norm_f, cfg_.rms_norm_eps, hidden);
}

StatusOr<Tensor> FalconH1Runner::last_token_hidden_(const Tensor& hidden) const {
  if (hidden.empty())
    return Status::InvalidArgument("hidden is empty");

  std::vector<int64_t> out_shape;
  if (hidden.size() == 1) {
    out_shape = hidden.shape;
  } else {
    int64_t cols = 0;
    ASSIGN_OR_RETURN(cols, hidden.cols());
    out_shape = {cols};
  }

  StatusOr<Tensor> last = allocate_f32_tensor(std::move(out_shape));
  if (!last.ok())
    return Status(last.status());
  if (Status s = ops_->take_last_row(hidden, last.value()); !s.ok())
    return s;
  return std::move(last.value());
}

StatusOr<Tensor> FalconH1Runner::lm_head_(const Tensor& hidden) {
  StatusOr<Tensor> last = last_token_hidden_(hidden);
  if (!last.ok())
    return Status(last.status());

  const Tensor& weight = weights_.lm_head.has_value() ? *weights_.lm_head : weights_.embeddings;
  StatusOr<std::vector<int64_t>> out_shape = linear_output_shape(last.value(), weight);
  if (!out_shape.ok())
    return Status(out_shape.status());

  StatusOr<Tensor> logits = allocate_f32_tensor(std::move(out_shape.value()));
  if (!logits.ok())
    return Status(logits.status());
  if (Status s = ops_->linear(last.value(), weight, logits.value()); !s.ok())
    return s;

  const float lm_scale = cfg_.scales ? cfg_.scales->lm_head : 1.f;
  if (Status s = ops_->scale(logits.value(), lm_scale); !s.ok())
    return s;
  return std::move(logits.value());
}

StatusOr<Tensor> FalconH1Runner::forward_hidden_(std::span<LayerCacheView> layers, Tensor& hidden,
                                                 int64_t past_len) {

  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    if (Status s = block_forward_(i, layers[i], hidden, past_len); !s.ok())
      return s;
  }

  if (Status s = norm_f_(hidden); !s.ok())
    return s;
  return std::move(hidden);
}

StatusOr<Tensor> FalconH1Runner::prefill(std::span<const int32_t> tokens, std::span<LayerCacheView> layers) {

  Tensor hidden;
  ASSIGN_OR_RETURN(hidden, embed_(tokens));

  Tensor normalized;
  ASSIGN_OR_RETURN(normalized, forward_hidden_(layers, hidden, 0));

  Tensor logits;
  ASSIGN_OR_RETURN(logits, lm_head_(normalized));
  return std::move(logits);
}

StatusOr<Tensor> FalconH1Runner::decode(const int32_t token, std::span<LayerCacheView> layers) {

  Tensor hidden;
  ASSIGN_OR_RETURN(hidden, embed_(std::span<const int32_t>(&token,1)));

  // convert shape from [1,H] to [H]
  if (hidden.size() == 2 && hidden.shape[0] == 1)
    hidden.shape = {hidden.shape[1]};

  Tensor normalized;
  ASSIGN_OR_RETURN(normalized, forward_hidden_(layers, hidden, /*past_len=*/0));

  Tensor logits;
  ASSIGN_OR_RETURN(logits, lm_head_(normalized));
  return std::move(logits);
}
