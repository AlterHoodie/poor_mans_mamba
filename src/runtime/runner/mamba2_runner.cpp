#include "runtime/runner/mamba2_runner.h"

#include "ops/common/shapes.h"
#include "ops/common/tensor_checks.h"

#include <utility>
#include <vector>

Mamba2Runner::~Mamba2Runner() = default;

Mamba2Runner::Mamba2Runner(ModelConfig cfg, Mamba2Weights weights, std::unique_ptr<CachePool> pool,
                           std::unique_ptr<DeviceAllocator> alloc, const OpsBackend& ops)
    : alloc_(std::move(alloc)), ops_(&ops), cfg_(std::move(cfg)), weights_(std::move(weights)),
      pool_(std::move(pool)) {}

Status Mamba2Runner::block_forward_(int layer_idx, LayerCacheView& cache, Tensor& hidden,
                                    bool is_prefill) {
  (void)is_prefill;
  if (cache.kind != LayerCacheKind::Mamba2)
    return Status::InvalidArgument("cache kind must be mamba2");
  if (layer_idx < 0 || layer_idx >= static_cast<int>(weights_.layers.size())) {
    return Status::InvalidArgument("layer_idx out of range");
  }
  const Mamba2LayerWeights& layer = weights_.layers[static_cast<size_t>(layer_idx)];

  StatusOr<Tensor> normed = allocate_f32_tensor(hidden.shape);
  if (!normed.ok())
    return Status(normed.status());
  if (Status s = ops_->rms_norm(hidden, layer.layer_norm, cfg_.rms_norm_eps, normed.value());
      !s.ok())
    return s;

  StatusOr<Tensor> mixer_out = allocate_f32_tensor(hidden.shape);
  if (!mixer_out.ok())
    return Status(mixer_out.status());

  Mamba2MixerWeights mw{.in_proj = layer.in_proj,
                        .conv1d = layer.conv1d,
                        .conv1d_bias = layer.conv1d_bias,
                        .dt_bias = layer.dt_bias,
                        .out_proj = layer.out_proj,
                        .A_log = layer.A_log,
                        .D = layer.D,
                        .mixer_norm = &layer.mixer_norm};
  if (Status s = ops_->mamba2_mixer_f32(normed.value(), mw, cfg_.ssm, /*scales=*/nullptr,
                                        cfg_.rms_norm_eps, cache, mixer_out.value());
      !s.ok()) {
    return s;
  }

  return ops_->add(hidden, mixer_out.value(), hidden);
}

StatusOr<Tensor> Mamba2Runner::embed_(std::span<const int32_t> tokens) {
  if (tokens.empty()) {
    return Status::InvalidArgument("embed requires at least one token");
  }
  if (weights_.embeddings.size() != 2) {
    return Status::InvalidArgument("embeddings must be rank 2 [vocab, hidden]");
  }

  const int64_t hidden = weights_.embeddings.shape[1];
  if (hidden != cfg_.hidden_size) {
    return Status::InvalidArgument("embeddings hidden dim mismatch with config");
  }

  StatusOr<Tensor> out = allocate_f32_tensor({static_cast<int64_t>(tokens.size()), hidden});
  if (!out.ok())
    return Status(out.status());

  if (Status s = ops_->embedding_lookup(weights_.embeddings, tokens.data(),
                                        static_cast<int64_t>(tokens.size()), /*scale=*/1.f,
                                        out.value());
      !s.ok()) {
    return s;
  }
  return std::move(out.value());
}

Status Mamba2Runner::norm_f_(Tensor& hidden) {
  return ops_->rms_norm(hidden, weights_.norm_f, cfg_.rms_norm_eps, hidden);
}

StatusOr<Tensor> Mamba2Runner::last_token_hidden_(const Tensor& hidden) const {
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

StatusOr<Tensor> Mamba2Runner::lm_head_(const Tensor& hidden) {
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
  return std::move(logits.value());
}

StatusOr<Tensor> Mamba2Runner::forward_hidden_(const CacheHandle& cache, Tensor hidden,
                                               bool is_prefill) {
  if (!cache.valid())
    return Status::InvalidArgument("invalid cache handle");
  if (!pool_)
    return Status::InvalidArgument("cache pool is not initialized");

  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    StatusOr<LayerCacheView> view = pool_->layer_view(cache, i);
    if (!view.ok())
      return Status(view.status());
    if (Status s = block_forward_(i, view.value(), hidden, is_prefill); !s.ok()) {
      return s;
    }
  }

  if (Status s = norm_f_(hidden); !s.ok())
    return s;
  return std::move(hidden);
}

StatusOr<PrefillResult> Mamba2Runner::prefill(std::span<const int32_t> tokens) {
  if (!pool_)
    return Status::InvalidArgument("cache pool is not initialized");
  if (!alloc_)
    return Status::InvalidArgument("allocator is not initialized");

  AllocatorScope scope(alloc_.get());

  PrefillResult result;
  ASSIGN_OR_RETURN(result.cache, pool_->acquire());

  StatusOr<Tensor> hidden = embed_(tokens);
  if (!hidden.ok()) {
    (void)pool_->release(result.cache);
    return Status(hidden.status());
  }

  StatusOr<Tensor> normalized = forward_hidden_(result.cache, std::move(hidden.value()), true);
  if (!normalized.ok()) {
    (void)pool_->release(result.cache);
    return Status(normalized.status());
  }

  StatusOr<Tensor> logits = lm_head_(normalized.value());
  if (!logits.ok()) {
    (void)pool_->release(result.cache);
    return Status(logits.status());
  }

  result.logits = std::move(logits.value());
  return result;
}

StatusOr<DecodeResult> Mamba2Runner::decode(const CacheHandle& cache, int32_t token) {
  if (!pool_)
    return Status::InvalidArgument("cache pool is not initialized");
  if (!alloc_)
    return Status::InvalidArgument("allocator is not initialized");
  if (!cache.valid())
    return Status::InvalidArgument("invalid cache handle");

  AllocatorScope scope(alloc_.get());

  const int32_t tok = token;
  StatusOr<Tensor> hidden = embed_(std::span<const int32_t>(&tok, 1));
  if (!hidden.ok())
    return Status(hidden.status());

  if (hidden.value().size() == 2 && hidden.value().shape[0] == 1) {
    hidden.value().shape = {hidden.value().shape[1]};
  }

  StatusOr<Tensor> normalized = forward_hidden_(cache, std::move(hidden.value()), false);
  if (!normalized.ok())
    return Status(normalized.status());

  StatusOr<Tensor> logits = lm_head_(normalized.value());
  if (!logits.ok())
    return Status(logits.status());

  DecodeResult result;
  result.logits = std::move(logits.value());
  return result;
}

Status Mamba2Runner::release(CacheHandle& cache) {
  if (!pool_)
    return Status::InvalidArgument("cache pool is not initialized");
  return pool_->release(cache);
}
