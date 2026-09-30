#include "core/device.h"
#include "core/status.h"
#include "runtime/cluster_scheduler.h"
#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

// Polls until every req_id is done or the timeout elapses. Returns false on
// timeout (test should fail loudly rather than hang on a stuck loop).
bool wait_all_done(ClusterScheduler& sched, const std::vector<uint64_t>& req_ids,
                   std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    bool all_done = true;
    for (uint64_t id : req_ids) {
      if (!sched.poll(id).done) {
        all_done = false;
        break;
      }
    }
    if (all_done)
      return true;
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

} // namespace

TEST(ClusterScheduler, RunsConcurrentSessionsOnOneWorker) {
  ClusterScheduler sched;
  ASSERT_TRUE(sched.load_model(MAMBA_TEST_MODEL_DIR, Device::CPU, /*n_workers=*/1).ok());

  // eos_id is set to an id that greedy decoding on this checkpoint is very
  // unlikely to emit, so every session ends via max_new_tokens.
  GenerateParams params{.max_new_tokens = 3, .eos_id = 50287};

  std::vector<uint64_t> req_ids;
  for (int i = 0; i < 3; ++i) {
    std::vector<int32_t> prompt = {1, 2, 3};
    StatusOr<uint64_t> id_or = sched.submit(prompt, params);
    ASSERT_TRUE(id_or.ok()) << id_or.status().message();
    req_ids.push_back(id_or.value());
  }

  ASSERT_TRUE(wait_all_done(sched, req_ids, std::chrono::seconds(30)))
      << "sessions did not finish before timeout";

  for (uint64_t id : req_ids) {
    Response r = sched.poll(id);
    EXPECT_TRUE(r.s.ok()) << r.s.message();
    EXPECT_EQ(r.gen_seq_len, params.max_new_tokens);
    EXPECT_EQ(r.tokens.size(), static_cast<size_t>(params.max_new_tokens));
    EXPECT_EQ(r.worker_idx, 0u);
  }
}

TEST(ClusterScheduler, PlacesAcrossTwoWorkers) {
  ClusterScheduler sched;
  ASSERT_TRUE(sched.load_model(MAMBA_TEST_MODEL_DIR, Device::CPU, 2).ok());

  GenerateParams params{.max_new_tokens = 3, .eos_id = 50287};

  std::vector<uint64_t> req_ids;
  for (int i = 0; i < 2; ++i) {
    StatusOr<uint64_t> id_or = sched.submit({1, 2, 3}, params);
    ASSERT_TRUE(id_or.ok()) << id_or.status().message();
    req_ids.push_back(id_or.value());
  }

  // Sticky homes are assigned at submit time and should already differ.
  std::unordered_set<size_t> homes;
  for (uint64_t id : req_ids)
    homes.insert(sched.poll(id).worker_idx);
  EXPECT_EQ(homes.size(), 2u);

  ASSERT_TRUE(wait_all_done(sched, req_ids, std::chrono::seconds(30)))
      << "sessions did not finish before timeout";

  for (uint64_t id : req_ids) {
    Response r = sched.poll(id);
    EXPECT_TRUE(r.s.ok()) << r.s.message();
    EXPECT_EQ(r.gen_seq_len, params.max_new_tokens);
    // Sticky: worker_idx unchanged after completion.
    EXPECT_TRUE(homes.count(r.worker_idx) == 1);
  }
}

TEST(ClusterScheduler, RejectsWhenAllWorkersFull) {
  ClusterScheduler sched;
  ASSERT_TRUE(
      sched.load_model(MAMBA_TEST_MODEL_DIR, Device::CPU, /*n_workers=*/1, /*num_slots=*/1).ok());

  GenerateParams params{.max_new_tokens = 16, .eos_id = 50287};

  StatusOr<uint64_t> first = sched.submit({1, 2, 3}, params);
  ASSERT_TRUE(first.ok()) << first.status().message();

  StatusOr<uint64_t> second = sched.submit({1, 2, 3}, params);
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.status().code(), Code::kOOM);

  ASSERT_TRUE(wait_all_done(sched, {first.value()}, std::chrono::seconds(30)));
}

TEST(ClusterScheduler, RejectsEmptyTokens) {
  ClusterScheduler sched;
  ASSERT_TRUE(sched.load_model(MAMBA_TEST_MODEL_DIR, Device::CPU, /*n_workers=*/1).ok());

  GenerateParams params{.max_new_tokens = 3, .eos_id = 0};
  StatusOr<uint64_t> id_or = sched.submit({}, params);
  EXPECT_FALSE(id_or.ok());
}

TEST(ClusterScheduler, RejectsMissingEos) {
  ClusterScheduler sched;
  ASSERT_TRUE(sched.load_model(MAMBA_TEST_MODEL_DIR, Device::CPU, /*n_workers=*/1).ok());

  GenerateParams params{.max_new_tokens = 3, .eos_id = -1};
  StatusOr<uint64_t> id_or = sched.submit({1, 2}, params);
  EXPECT_FALSE(id_or.ok());
}

TEST(ClusterScheduler, PollUnknownReqIdIsNotFound) {
  ClusterScheduler sched;
  ASSERT_TRUE(sched.load_model(MAMBA_TEST_MODEL_DIR, Device::CPU, /*n_workers=*/1).ok());

  Response r = sched.poll(/*req_id=*/12345);
  EXPECT_FALSE(r.s.ok());
}

TEST(ClusterScheduler, MigratesSessionToOtherWorker) {
  ClusterScheduler sched;
  ASSERT_TRUE(sched.load_model(MAMBA_TEST_MODEL_DIR, Device::CPU, /*n_workers=*/2).ok());

  // Long enough that we can migrate mid-flight before max_new_tokens.
  GenerateParams params{.max_new_tokens = 8, .eos_id = 50287};

  StatusOr<uint64_t> id_or = sched.submit({1, 2, 3}, params);
  ASSERT_TRUE(id_or.ok()) << id_or.status().message();
  const uint64_t req_id = id_or.value();

  // Wait until sticky decode has produced at least one token (phase Decoding).
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  size_t src = 0;
  for (;;) {
    Response r = sched.poll(req_id);
    ASSERT_TRUE(r.s.ok()) << r.s.message();
    if (r.gen_seq_len >= 1 && !r.done) {
      src = r.worker_idx;
      break;
    }
    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "timed out waiting for decode";
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  const size_t dst = 1 - src;
  Status m = sched.migrate(req_id, dst);
  ASSERT_TRUE(m.ok()) << m.message();

  ASSERT_TRUE(wait_all_done(sched, {req_id}, std::chrono::seconds(30)))
      << "migrated session did not finish before timeout";

  Response r = sched.poll(req_id);
  EXPECT_TRUE(r.s.ok()) << r.s.message();
  EXPECT_TRUE(r.done);
  EXPECT_EQ(r.worker_idx, dst);
  EXPECT_EQ(r.gen_seq_len, params.max_new_tokens);
}

TEST(ClusterScheduler, GoldenContinuationAfterMigrate) {
  GenerateParams params{.max_new_tokens = 6, .eos_id = 50287};
  const std::vector<int32_t> prompt = {1, 2, 3, 4};

  std::vector<int32_t> baseline;
  {
    ClusterScheduler sched;
    ASSERT_TRUE(sched.load_model(MAMBA_TEST_MODEL_DIR, Device::CPU, /*n_workers=*/1).ok());
    StatusOr<uint64_t> id_or = sched.submit(prompt, params);
    ASSERT_TRUE(id_or.ok()) << id_or.status().message();
    ASSERT_TRUE(wait_all_done(sched, {id_or.value()}, std::chrono::seconds(60)));
    Response r = sched.poll(id_or.value());
    ASSERT_TRUE(r.s.ok()) << r.s.message();
    ASSERT_EQ(r.gen_seq_len, params.max_new_tokens);
    baseline = r.tokens;
  }

  ClusterScheduler sched;
  ASSERT_TRUE(sched.load_model(MAMBA_TEST_MODEL_DIR, Device::CPU, /*n_workers=*/2).ok());
  StatusOr<uint64_t> id_or = sched.submit(prompt, params);
  ASSERT_TRUE(id_or.ok()) << id_or.status().message();
  const uint64_t req_id = id_or.value();

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  size_t src = 0;
  for (;;) {
    Response r = sched.poll(req_id);
    ASSERT_TRUE(r.s.ok()) << r.s.message();
    if (r.gen_seq_len >= 1 && !r.done) {
      src = r.worker_idx;
      break;
    }
    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "timed out waiting for decode";
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  const size_t dst = 1 - src;
  ASSERT_TRUE(sched.migrate(req_id, dst).ok());
  ASSERT_TRUE(wait_all_done(sched, {req_id}, std::chrono::seconds(60)));

  Response r = sched.poll(req_id);
  ASSERT_TRUE(r.s.ok()) << r.s.message();
  EXPECT_EQ(r.worker_idx, dst);
  EXPECT_EQ(r.tokens, baseline);
}
