#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/status.h"
#include "runtime/runner/runner.h"

struct GenerateParams {
    int max_new_tokens = 32;
    int eos_id = -1;  // required for generate; <0 rejected
};

class Scheduler {
   private:
    Runner& runner_;

    StatusOr<int32_t> sample_(Tensor& logits);

   public:
    explicit Scheduler(Runner& runner) : runner_(runner) {}

    // Sync greedy generate: prefill → decode until EOS or max_new_tokens → release.
    // Returns only newly generated token ids (not the prompt).
    StatusOr<std::vector<int32_t>> generate(std::span<const int32_t> tokens,
                                            const GenerateParams& params);
};
