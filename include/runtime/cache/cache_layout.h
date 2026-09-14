#pragma once

#include <cstddef>
#include <vector>

#include "core/status.h"
#include "io/config.h"

struct MambaLayerLayout {
    int layer_idx = 0;
    size_t conv_offset = 0;
    size_t conv_size = 0;
    size_t ssm_offset = 0;
    size_t ssm_size = 0;
};

struct CacheLayout {
    // one particular slot of mamba state ideally for one request
    size_t slot_bytes = 0;
    int num_layers = 0;
    std::vector<MambaLayerLayout> layers;

    // Build CacheLayout
    static StatusOr<CacheLayout> from_config(const Mamba2Config& cfg);
};

// Non-owning view into a slot's DeviceBuffer. Valid for the slot lifetime.
struct MambaLayerCacheView {
    void* conv = nullptr;
    size_t conv_bytes = 0;
    void* ssm = nullptr;
    size_t ssm_bytes = 0;
};
