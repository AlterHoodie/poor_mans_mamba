#pragma once

#include <cstdint>
#include <span>

#include "core/status.h"
#include "core/tensor.h"
#include "runtime/cache/cache_pool.h"

// Runner ↔ scheduler step outputs. Scheduler owns seq_id + token history.
struct PrefillResult {
    CacheHandle cache;
    Tensor logits;  // shape {vocab_size}, last prompt position
};

struct DecodeResult {
    Tensor logits;  // shape {vocab_size}
};

class Runner {
   public:
    virtual ~Runner() = default;

    virtual StatusOr<PrefillResult> prefill(std::span<const int32_t> tokens) = 0;
    virtual StatusOr<DecodeResult> decode(const CacheHandle& cache, int32_t token) = 0;
    virtual Status release(CacheHandle& cache) = 0;
};
