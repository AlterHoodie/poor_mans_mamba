#pragma once

#include "core/status.h"
#include "runtime/runner/runner.h"

#include <cstdint>
#include <span>
#include <vector>

struct GenerateParams {
  int max_new_tokens = 32;
  int eos_id = -1; // required for generate; <0 rejected
};

class Scheduler {
private:
  Runner& runner_;

  StatusOr<int32_t> sample_(Tensor& logits);

public:
  explicit Scheduler(Runner& runner) : runner_(runner) {}

  // Sync greedy generate: prefill → decode until EOS, max_new_tokens, or KV
  // cache overflow → release. Returns only newly generated token ids (not the
  // prompt). Overflow returns tokens produced so far instead of failing.
  StatusOr<std::vector<int32_t>> generate(std::span<const int32_t> tokens,
                                          const GenerateParams& params);
};
