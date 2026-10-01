#pragma once

#include "runtime/session.h"
#include "runtime/worker.h"

#include <cstdint>
#include <optional>
#include <span>

struct RebalanceDecision {
  uint64_t victim_req_id = 0;
  size_t dst_idx = 0;
};

class RebalancePolicy {
public:
  virtual ~RebalancePolicy() = default;
  // Returns nullopt if no rebalance is warranted right now.
  virtual std::optional<RebalanceDecision> check(std::span<const WorkerStat> views,
                                                 std::span<const SessionView> sessions) = 0;
};

class SimpleRebalancePolicy : public RebalancePolicy {
public:
  explicit SimpleRebalancePolicy(int imbalance_threshold = 2)
      : imbalance_threshold_(imbalance_threshold) {}

  std::optional<RebalanceDecision> check(std::span<const WorkerStat> views,
                                         std::span<const SessionView> sessions) override;

private:
  int imbalance_threshold_;
};
