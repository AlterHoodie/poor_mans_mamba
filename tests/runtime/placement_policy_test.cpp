#include "core/status.h"
#include "runtime/policy/placement_policy.h"
#include "runtime/worker.h"
#include <gtest/gtest.h>

#include <vector>

TEST(SimplePlacementPolicy, RejectsEmptyViews) {
  SimplePlacementPolicy policy;
  StatusOr<size_t> idx = policy.place({}, GenerateParams{}, {});
  ASSERT_FALSE(idx.ok());
}

TEST(SimplePlacementPolicy, PrefersMoreFreeSlots) {
  SimplePlacementPolicy policy;
  const std::vector<int32_t> tokens = {1};
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 3},
      {.index = 1, .capacity = 4, .inflight = 1},
  };
  StatusOr<size_t> idx = policy.place(tokens, GenerateParams{}, views);
  ASSERT_TRUE(idx.ok()) << idx.status().message();
  EXPECT_EQ(idx.value(), 1u);
}

TEST(SimplePlacementPolicy, RejectsWhenAllFull) {
  SimplePlacementPolicy policy;
  const std::vector<int32_t> tokens = {1};
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 2, .inflight = 2},
      {.index = 1, .capacity = 2, .inflight = 2},
  };
  StatusOr<size_t> idx = policy.place(tokens, GenerateParams{}, views);
  ASSERT_FALSE(idx.ok());
  EXPECT_EQ(idx.status().code(), Code::kOOM);
}
