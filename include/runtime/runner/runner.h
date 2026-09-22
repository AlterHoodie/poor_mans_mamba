#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"

#include <cstdint>
#include <cstdlib>
#include <span>

class Runner {
public:
  virtual ~Runner() = default;

  virtual StatusOr<Tensor> prefill(std::span<const int32_t> tokens, std::span<LayerCacheView> layers) = 0;
  virtual StatusOr<Tensor> decode(const int32_t token, std::span<LayerCacheView> layers) = 0;
};
