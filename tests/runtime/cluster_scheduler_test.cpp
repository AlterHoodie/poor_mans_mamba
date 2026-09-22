#include "core/device.h"
#include "runtime/cluster_scheduler.h"
#include <gtest/gtest.h>

#include <chrono>
#include <thread>
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
  }
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
