#pragma once

#include "io/config_parser.h"
#include "model/mamba2_weights.h"
#include "runner.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"

class Mamba2Runner : public Runner {
   private:
    Mamba2Config cfg_;
    Mamba2Weights weights_;
    CachePool* pool_ = nullptr;

    Status block_forward_(MambaLayerCacheView& cache, Tensor& hidden, bool is_prefill);

    StatusOr<Tensor> embed_(const std::span<int32_t> tokens);
    Status norm_f_(Tensor& hidden);  // inplace
    StatusOr<Tensor> lm_head_(const Tensor& hidden);

   public:
    Mamba2Runner(Mamba2Config& cfg, Mamba2Weights& weights)
        : cfg_(cfg), weights_(std::move(weights)) {}

    ~Mamba2Runner();

    Status prefill(Sequence& seq, std::span<const int32_t> tokens) override;

    Status decode(Sequence& seq, const int32_t token) override;
};