#pragma once

#include "core/status.h"
#include "io/config.h"

#include <cstddef>
#include <vector>

enum class LayerCacheKind { Mamba2, AttnKV, Hybrid };

struct Region {
  size_t offset = 0;
  size_t bytes = 0;
};

struct View {
  void* ptr = nullptr;
  size_t bytes = 0;
};

struct LayerEntry {
  int layer_idx = 0;
  LayerCacheKind kind = LayerCacheKind::Mamba2;
  Region conv;
  Region ssm;
  Region k;
  Region v;
};

struct CacheLayout {
  size_t slot_bytes = 0;
  int num_layers = 0;
  std::vector<LayerEntry> layers;
};

StatusOr<CacheLayout> create_cache_layout(const ModelConfig& cfg);

// Non-owning view into a slot's DeviceBuffer. Valid for the slot lifetime.
struct LayerCacheView {
  LayerCacheKind kind;
  View conv, ssm, k, v;
};
