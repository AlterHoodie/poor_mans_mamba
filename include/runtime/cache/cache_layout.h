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

// Generic CacheLayout
struct CacheLayout {
  // one particular slot of mamba state ideally for one request
  size_t slot_bytes = 0;
  int num_layers = 0;
  std::vector<LayerEntry> layers;
};

// ideally should be factory classes but dont want to add too much complexity, so will be simple
// functions that reads a particular config and spits out a usable CacheLayout
StatusOr<CacheLayout> create_mamba2_layout(const Mamba2Config&);

StatusOr<CacheLayout> create_falconh1_layout(const FalconH1Config&);

// Non-owning view into a slot's DeviceBuffer. Valid for the slot lifetime.
struct LayerCacheView {
  LayerCacheKind kind;
  View conv, ssm, k, v;
};