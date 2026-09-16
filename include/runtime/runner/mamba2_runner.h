#pragma once

#include "io/config.h"
#include "model/mamba2_weights.h"
#include "runner.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"

#include <memory>
#include <span>

class Mamba2Runner : public Runner {
private:
  std::unique_ptr<DeviceAllocator> alloc_;
  Mamba2Config cfg_;
  Mamba2Weights weights_;
  std::unique_ptr<CachePool> pool_;

  Status block_forward_(int layer_idx, LayerCacheView& cache, Tensor& hidden, bool is_prefill);

  StatusOr<Tensor> embed_(std::span<const int32_t> tokens);
  Status norm_f_(Tensor& hidden);
  StatusOr<Tensor> lm_head_(const Tensor& hidden);
  StatusOr<Tensor> last_token_hidden_(const Tensor& hidden) const;

  StatusOr<Tensor> forward_hidden_(const CacheHandle& cache, Tensor hidden, bool is_prefill);

public:
  Mamba2Runner(Mamba2Config cfg, Mamba2Weights weights, std::unique_ptr<CachePool> pool,
               std::unique_ptr<DeviceAllocator> alloc);

  ~Mamba2Runner() override;

  StatusOr<PrefillResult> prefill(std::span<const int32_t> tokens) override;
  StatusOr<DecodeResult> decode(const CacheHandle& cache, int32_t token) override;
  Status release(CacheHandle& cache) override;
};
