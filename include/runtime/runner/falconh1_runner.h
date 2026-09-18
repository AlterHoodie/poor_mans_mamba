#pragma once

#include "io/config.h"
#include "model/falconh1_weights.h"
#include "ops/backend.h"
#include "runner.h"
#include "runtime/cache/cache_pool.h"

#include <memory>
#include <span>

class FalconH1Runner : public Runner {
private:
  std::unique_ptr<DeviceAllocator> alloc_;
  const OpsBackend* ops_ = nullptr;
  ModelConfig cfg_;
  FalconH1Weights weights_;
  std::unique_ptr<CachePool> pool_;

  Status block_forward_(int layer_idx, LayerCacheView& cache, Tensor& hidden, int64_t past_len);

  StatusOr<Tensor> embed_(std::span<const int32_t> tokens);
  Status norm_f_(Tensor& hidden);
  StatusOr<Tensor> lm_head_(const Tensor& hidden);
  StatusOr<Tensor> last_token_hidden_(const Tensor& hidden) const;

  StatusOr<Tensor> forward_hidden_(const CacheHandle& cache, Tensor hidden, int64_t past_len);

public:
  FalconH1Runner(ModelConfig cfg, FalconH1Weights weights, std::unique_ptr<CachePool> pool,
                 std::unique_ptr<DeviceAllocator> alloc, const OpsBackend& ops);

  ~FalconH1Runner() override;

  StatusOr<PrefillResult> prefill(std::span<const int32_t> tokens) override;
  StatusOr<DecodeResult> decode(const CacheHandle& cache, int32_t token) override;
  Status release(CacheHandle& cache) override;
};
