#include "runtime/cache/cache_layout.h"

#include <cstdint>

namespace {

constexpr size_t kF32Bytes = sizeof(float);

}  // namespace

StatusOr<CacheLayout> CacheLayout::from_config(const Mamba2Config& cfg) {
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
    const size_t ssm_elems = static_cast<size_t>(cfg.num_heads) *
                             static_cast<size_t>(cfg.head_dim) *
                             static_cast<size_t>(cfg.state_size);

    const size_t conv_bytes = conv_elems * kF32Bytes;
    const size_t ssm_bytes = ssm_elems * kF32Bytes;

    CacheLayout layout;
    layout.num_layers = cfg.num_hidden_layers;
    layout.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));

    size_t offset = 0;
    for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
        MambaLayerLayout& L = layout.layers[static_cast<size_t>(layer)];
        L.layer_idx = layer;
        L.conv_offset = offset;
        L.conv_size = conv_bytes;
        offset += conv_bytes;
        L.ssm_offset = offset;
        L.ssm_size = ssm_bytes;
        offset += ssm_bytes;
    }

    layout.slot_bytes = offset;
    return layout;
}
