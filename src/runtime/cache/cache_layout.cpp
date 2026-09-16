#include "runtime/cache/cache_layout.h"

#include <cstdint>

namespace {

constexpr size_t kF32Bytes = sizeof(float);

} // namespace

StatusOr<CacheLayout> create_mamba2_layout(const Mamba2Config& cfg) {
  if (cfg.num_hidden_layers <= 0) {
    return Status::InvalidArgument("num_hidden_layers must be positive");
  }
  if (cfg.hidden_size <= 0 || cfg.expand <= 0) {
    return Status::InvalidArgument("hidden_size and expand must be positive");
  }
  if (cfg.conv_kernel <= 1) {
    return Status::InvalidArgument("conv_kernel must be > 1 for conv state");
  }
  if (cfg.num_heads <= 0 || cfg.head_dim <= 0 || cfg.state_size <= 0) {
    return Status::InvalidArgument("num_heads, head_dim, and state_size must be positive");
  }
  if (cfg.n_groups <= 0) {
    return Status::InvalidArgument("n_groups must be positive");
  }
  if (cfg.num_heads % cfg.n_groups != 0) {
    return Status::InvalidArgument("num_heads must be divisible by n_groups");
  }

  const int64_t intermediate = static_cast<int64_t>(cfg.hidden_size) * cfg.expand;
  if (intermediate != static_cast<int64_t>(cfg.num_heads) * cfg.head_dim) {
    return Status::InvalidArgument("expand*hidden_size must equal num_heads*head_dim");
  }

  // HF: conv_dim = intermediate + 2 * n_groups * state_size
  const int64_t conv_dim = intermediate + 2 * static_cast<int64_t>(cfg.n_groups) * cfg.state_size;
  const size_t conv_elems =
      static_cast<size_t>(conv_dim) * static_cast<size_t>(cfg.conv_kernel - 1);
  const size_t ssm_elems = static_cast<size_t>(cfg.num_heads) * static_cast<size_t>(cfg.head_dim) *
                           static_cast<size_t>(cfg.state_size);

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

StatusOr<CacheLayout> create_falconh1_layout(const FalconH1Config& cfg) {
  if (cfg.num_hidden_layers <= 0) {
    return Status::InvalidArgument("num_hidden_layers must be positive");
  }

  // mamba checks
  if (cfg.hidden_size <= 0 || cfg.mamba_expand <= 0) {
    return Status::InvalidArgument("hidden_size and expand must be positive");
  }
  if (cfg.mamba_d_conv <= 1) {
    return Status::InvalidArgument("conv_kernel must be > 1 for conv state");
  }
  if (cfg.mamba_n_heads <= 0 || cfg.mamba_d_head <= 0 || cfg.mamba_d_state <= 0) {
    return Status::InvalidArgument("num_heads, head_dim, and state_size must be positive");
  }
  if (cfg.mamba_n_groups <= 0) {
    return Status::InvalidArgument("n_groups must be positive");
  }
  if (cfg.mamba_n_heads % cfg.mamba_n_groups != 0) {
    return Status::InvalidArgument("num_heads must be divisible by n_groups");
  }
  if (cfg.mamba_d_ssm != cfg.mamba_n_heads * cfg.mamba_d_head) {
    return Status::InvalidArgument("mamba intermediate state must be equal to n_heads  * d_heads");
  }

  // attention checks
  if (cfg.num_attention_heads < 1)
    return Status::InvalidArgument("num_attention_heads should be greater than 0");
  if (cfg.num_attention_heads < 1)
    return Status::InvalidArgument("num_key_value_heads should be greater than 0");
  if (cfg.num_attention_heads != cfg.num_key_value_heads)
    return Status::InvalidArgument("num_attention_heads should be equal to num_key_value_heads");
  if (cfg.head_dim < 1)
    return Status::InvalidArgument("head_dim should be greater than 0");
  if (cfg.max_seq_length < 1)
    return Status::InvalidArgument("max_seq_length should be greater than 0");

  // Mamba Cache State
  // HF: conv_dim = intermediate + 2 * n_groups * state_size
  const int64_t conv_dim =
      cfg.mamba_d_ssm + 2 * static_cast<int64_t>(cfg.mamba_n_groups) * cfg.mamba_d_state;
  const size_t conv_elems =
      static_cast<size_t>(conv_dim) * static_cast<size_t>(cfg.mamba_d_conv - 1);
  const size_t ssm_elems = cfg.mamba_n_heads * cfg.mamba_d_head * cfg.mamba_d_state;

  const size_t conv_bytes = conv_elems * kF32Bytes;
  const size_t ssm_bytes = ssm_elems * kF32Bytes;

  // KV Cache State
  const size_t kv_dim =
      static_cast<size_t>(cfg.num_key_value_heads) * static_cast<size_t>(cfg.head_dim);
  const size_t kv_bytes = static_cast<size_t>(cfg.max_seq_length) * kv_dim * kF32Bytes;

  CacheLayout layout;
  layout.num_layers = cfg.num_hidden_layers;
  layout.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));

  size_t offset = 0;
  for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
    LayerEntry& L = layout.layers[static_cast<size_t>(layer)];
    L.layer_idx = layer;
    L.kind = LayerCacheKind::Hybrid;

    // mamba stuff
    L.conv.offset = offset;
    L.conv.bytes = conv_bytes;
    offset += conv_bytes;

    L.ssm.offset = offset;
    L.ssm.bytes = ssm_bytes;
    offset += ssm_bytes;

    // attn stuff
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