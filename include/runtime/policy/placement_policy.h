#pragma once

#include "core/status.h"
#include "runtime/worker.h"

#include <span>

class PlacementPolicy {
public:
  virtual ~PlacementPolicy() = default;
  virtual StatusOr<size_t> place(std::span<const int32_t> tokens, const GenerateParams& params,
                                 std::span<const WorkerStat> views) = 0;
};

class SimplePlacementPolicy : public PlacementPolicy {
public:
  StatusOr<size_t> place(std::span<const int32_t> tokens, const GenerateParams& params,
                         std::span<const WorkerStat> views) override;
};
