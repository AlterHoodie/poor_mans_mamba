#include "runtime/policy/rebalance_policy.h"
#include "runtime/session.h"
#include "runtime/worker.h"
#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace {

SessionView decoding(uint64_t req_id, size_t worker_idx, size_t gen_len) {
  return SessionView{.req_id = req_id,
                     .worker_idx = worker_idx,
                     .phase = SessionPhase::Decoding,
                     .migrate_pending = false,
                     .gen_len = gen_len};
}

} // namespace

TEST(SimpleRebalancePolicy, RejectsEmptyViews) {
  SimpleRebalancePolicy policy;
  std::vector<SessionView> sessions = {decoding(/*req_id=*/1, /*worker_idx=*/0, /*gen_len=*/3)};
  EXPECT_FALSE(policy.check({}, sessions).has_value());
}

TEST(SimpleRebalancePolicy, RejectsEmptySessions) {
  SimpleRebalancePolicy policy;
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 3},
      {.index = 1, .capacity = 4, .inflight = 0},
  };
  EXPECT_FALSE(policy.check(views, {}).has_value());
}

TEST(SimpleRebalancePolicy, RejectsWhenGapBelowThreshold) {
  SimpleRebalancePolicy policy(/*imbalance_threshold=*/2);
  // free: w0=2, w1=3 → gap 1 < 2
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 2},
      {.index = 1, .capacity = 4, .inflight = 1},
  };
  std::vector<SessionView> sessions = {decoding(1, 0, 5), decoding(2, 1, 1)};
  EXPECT_FALSE(policy.check(views, sessions).has_value());
}

TEST(SimpleRebalancePolicy, MovesVictimFromHotToCold) {
  SimpleRebalancePolicy policy(/*imbalance_threshold=*/2);
  // free: w0=0, w1=3 → gap 3 >= 2
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 4},
      {.index = 1, .capacity = 4, .inflight = 1},
  };
  std::vector<SessionView> sessions = {
      decoding(/*req_id=*/10, /*worker_idx=*/0, /*gen_len=*/2),
      decoding(/*req_id=*/11, /*worker_idx=*/1, /*gen_len=*/1),
  };

  std::optional<RebalanceDecision> d = policy.check(views, sessions);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->victim_req_id, 10u);
  EXPECT_EQ(d->dst_idx, 1u);
}

TEST(SimpleRebalancePolicy, PrefersLongestGenLenOnHotWorker) {
  SimpleRebalancePolicy policy(/*imbalance_threshold=*/2);
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 4},
      {.index = 1, .capacity = 4, .inflight = 0},
  };
  std::vector<SessionView> sessions = {
      decoding(/*req_id=*/1, /*worker_idx=*/0, /*gen_len=*/2),
      decoding(/*req_id=*/2, /*worker_idx=*/0, /*gen_len=*/9),
      decoding(/*req_id=*/3, /*worker_idx=*/0, /*gen_len=*/4),
  };

  std::optional<RebalanceDecision> d = policy.check(views, sessions);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->victim_req_id, 2u);
  EXPECT_EQ(d->dst_idx, 1u);
}

TEST(SimpleRebalancePolicy, TieBreaksEqualGenLenBySmallerReqId) {
  SimpleRebalancePolicy policy(/*imbalance_threshold=*/2);
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 3},
      {.index = 1, .capacity = 4, .inflight = 0},
  };
  std::vector<SessionView> sessions = {
      decoding(/*req_id=*/20, /*worker_idx=*/0, /*gen_len=*/5),
      decoding(/*req_id=*/7, /*worker_idx=*/0, /*gen_len=*/5),
  };

  std::optional<RebalanceDecision> d = policy.check(views, sessions);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->victim_req_id, 7u);
}

TEST(SimpleRebalancePolicy, SkipsWhenAnyMigratePending) {
  SimpleRebalancePolicy policy;
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 4},
      {.index = 1, .capacity = 4, .inflight = 0},
  };
  std::vector<SessionView> sessions = {
      decoding(1, 0, 3),
      SessionView{.req_id = 2,
                  .worker_idx = 1,
                  .phase = SessionPhase::Decoding,
                  .migrate_pending = true,
                  .gen_len = 1},
  };
  EXPECT_FALSE(policy.check(views, sessions).has_value());
}

TEST(SimpleRebalancePolicy, SkipsWhenAnySessionMigrating) {
  SimpleRebalancePolicy policy;
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 4},
      {.index = 1, .capacity = 4, .inflight = 0},
  };
  std::vector<SessionView> sessions = {
      decoding(1, 0, 3),
      SessionView{.req_id = 2,
                  .worker_idx = 0,
                  .phase = SessionPhase::Migrating,
                  .migrate_pending = false,
                  .gen_len = 8},
  };
  EXPECT_FALSE(policy.check(views, sessions).has_value());
}

TEST(SimpleRebalancePolicy, IgnoresNonDecodingVictimsOnHotWorker) {
  SimpleRebalancePolicy policy;
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 4},
      {.index = 1, .capacity = 4, .inflight = 0},
  };
  std::vector<SessionView> sessions = {
      SessionView{.req_id = 1,
                  .worker_idx = 0,
                  .phase = SessionPhase::Prefilling,
                  .migrate_pending = false,
                  .gen_len = 0},
      decoding(/*req_id=*/2, /*worker_idx=*/1, /*gen_len=*/4),
  };
  EXPECT_FALSE(policy.check(views, sessions).has_value());
}

TEST(SimpleRebalancePolicy, RejectsSingleWorker) {
  SimpleRebalancePolicy policy;
  std::vector<WorkerStat> views = {{.index = 0, .capacity = 4, .inflight = 3}};
  std::vector<SessionView> sessions = {decoding(1, 0, 2)};
  EXPECT_FALSE(policy.check(views, sessions).has_value());
}

TEST(SimpleRebalancePolicy, RejectsWhenColdHasNoFreeSlots) {
  SimpleRebalancePolicy policy;
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 2, .inflight = 2},
      {.index = 1, .capacity = 2, .inflight = 2},
  };
  std::vector<SessionView> sessions = {decoding(1, 0, 2), decoding(2, 1, 2)};
  EXPECT_FALSE(policy.check(views, sessions).has_value());
}

TEST(SimpleRebalancePolicy, HonorsCustomThreshold) {
  SimpleRebalancePolicy strict(/*imbalance_threshold=*/4);
  // free: w0=1, w1=3 → gap 2 < 4
  std::vector<WorkerStat> views = {
      {.index = 0, .capacity = 4, .inflight = 3},
      {.index = 1, .capacity = 4, .inflight = 1},
  };
  std::vector<SessionView> sessions = {decoding(1, 0, 5)};

  EXPECT_FALSE(strict.check(views, sessions).has_value());

  SimpleRebalancePolicy loose(/*imbalance_threshold=*/2);
  std::optional<RebalanceDecision> d = loose.check(views, sessions);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->victim_req_id, 1u);
  EXPECT_EQ(d->dst_idx, 1u);
}
