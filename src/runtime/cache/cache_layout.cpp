#include "runtime/cache/cache_layout.h"

#include <cstdint>

namespace {

constexpr size_t kF32Bytes = sizeof(float);

StatusOr<CacheLayout> create_mamba_only_layout(const ModelConfig& cfg) {
  if (cfg.num_hidden_layers <= 0) {
    return Status::InvalidArgument("num_hidden_layers must be positive");
  }
  if (cfg.hidden_size <= 0) {
    return Status::InvalidArgument("hidden_size must be positive");
  }
  const SsmConfig& ssm = cfg.ssm;
  if (ssm.d_conv <= 1) {
    return Status::InvalidArgument("conv_kernel must be > 1 for conv state");
  }
  if (ssm.n_heads <= 0 || ssm.d_head <= 0 || ssm.d_state <= 0 || ssm.d_inner <= 0) {
    return Status::InvalidArgument("invalid ssm dimensions");
  }
  if (ssm.n_groups <= 0) {
    return Status::InvalidArgument("n_groups must be positive");
  }
  if (ssm.n_heads % ssm.n_groups != 0) {
    return Status::InvalidArgument("n_heads must be divisible by n_groups");
  }
  if (ssm.d_inner != ssm.n_heads * ssm.d_head) {
    return Status::InvalidArgument("d_inner must equal n_heads * d_head");
  }

  const int64_t conv_dim = ssm.d_inner + 2 * static_cast<int64_t>(ssm.n_groups) * ssm.d_state;
  const size_t conv_elems = static_cast<size_t>(conv_dim) * static_cast<size_t>(ssm.d_conv - 1);
  const size_t ssm_elems = static_cast<size_t>(ssm.n_heads) * static_cast<size_t>(ssm.d_head) *
                           static_cast<size_t>(ssm.d_state);

  const size_t conv_bytes = conv_elems * kF32Bytes;
  const size_t ssm_bytes = ssm_elems * kF32Bytes;

  CacheLayout layout;
  layout.num_layers = cfg.num_hidden_layers;
  layout.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));

  size_t offset = 0;
  for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
    LayerEntry& L = layout.layers[static_cast<size_t>(layer)];
    L.layer_idx = layer;
    L.kind = LayerCacheKind::Mamba2;
    L.conv.offset = offset;
    L.conv.bytes = conv_bytes;
    offset += conv_bytes;
    L.ssm.offset = offset;
    L.ssm.bytes = ssm_bytes;
    offset += ssm_bytes;
  }

  layout.slot_bytes = offset;
  return layout;
}

StatusOr<CacheLayout> create_parallel_hybrid_layout(const ModelConfig& cfg) {
  if (cfg.num_hidden_layers <= 0) {
    return Status::InvalidArgument("num_hidden_layers must be positive");
  }
  if (cfg.hidden_size <= 0) {
    return Status::InvalidArgument("hidden_size must be positive");
  }
  if (!cfg.attn.has_value()) {
    return Status::InvalidArgument("ParallelHybrid requires attn config");
  }

  const SsmConfig& ssm = cfg.ssm;
  const AttnConfig& attn = *cfg.attn;

  if (ssm.d_conv <= 1) {
    return Status::InvalidArgument("conv_kernel must be > 1 for conv state");
  }
  if (ssm.n_heads <= 0 || ssm.d_head <= 0 || ssm.d_state <= 0 || ssm.d_inner <= 0) {
    return Status::InvalidArgument("invalid ssm dimensions");
  }
  if (ssm.n_groups <= 0) {
    return Status::InvalidArgument("n_groups must be positive");
  }
  if (ssm.n_heads % ssm.n_groups != 0) {
    return Status::InvalidArgument("n_heads must be divisible by n_groups");
  }
  if (ssm.d_inner != ssm.n_heads * ssm.d_head) {
    return Status::InvalidArgument("d_inner must equal n_heads * d_head");
  }

  if (attn.n_q_heads < 1)
    return Status::InvalidArgument("num_attention_heads should be greater than 0");
  if (attn.n_kv_heads < 1)
    return Status::InvalidArgument("num_key_value_heads should be greater than 0");
  if (attn.n_q_heads % attn.n_kv_heads != 0)
    return Status::InvalidArgument("num_attention_heads must be divisible by num_key_value_heads");
  if (attn.head_dim < 1)
    return Status::InvalidArgument("head_dim should be greater than 0");
  if (cfg.max_seq_length < 1)
    return Status::InvalidArgument("max_seq_length should be greater than 0");

  const int64_t conv_dim = ssm.d_inner + 2 * static_cast<int64_t>(ssm.n_groups) * ssm.d_state;
  const size_t conv_elems = static_cast<size_t>(conv_dim) * static_cast<size_t>(ssm.d_conv - 1);
  const size_t ssm_elems = static_cast<size_t>(ssm.n_heads) * static_cast<size_t>(ssm.d_head) *
                           static_cast<size_t>(ssm.d_state);

  const size_t conv_bytes = conv_elems * kF32Bytes;
  const size_t ssm_bytes = ssm_elems * kF32Bytes;

  const size_t kv_dim = static_cast<size_t>(attn.n_kv_heads) * static_cast<size_t>(attn.head_dim);
  const size_t kv_bytes = static_cast<size_t>(cfg.max_seq_length) * kv_dim * kF32Bytes;

  CacheLayout layout;
  layout.num_layers = cfg.num_hidden_layers;
  layout.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));

  size_t offset = 0;
  for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
    LayerEntry& L = layout.layers[static_cast<size_t>(layer)];
    L.layer_idx = layer;
    L.kind = LayerCacheKind::Hybrid;

    L.conv.offset = offset;
    L.conv.bytes = conv_bytes;
    offset += conv_bytes;

    L.ssm.offset = offset;
    L.ssm.bytes = ssm_bytes;
    offset += ssm_bytes;

    L.k.offset = offset;
    L.k.bytes = kv_bytes;
    offset += kv_bytes;

    L.v.offset = offset;
    L.v.bytes = kv_bytes;
    offset += kv_bytes;
  }

  layout.slot_bytes = offset;
  return layout;
}

} // namespace

StatusOr<CacheLayout> create_cache_layout(const ModelConfig& cfg) {
  switch (cfg.layout) {
  case ArchLayout::MambaOnly:
    return create_mamba_only_layout(cfg);
  case ArchLayout::ParallelHybrid:
    return create_parallel_hybrid_layout(cfg);
  }
  return Status::InvalidArgument("unknown ArchLayout");
}
