#include "core/device.h"
#include "core/status.h"
#include "runtime/cluster_config.h"
#include "runtime/cluster_scheduler.h"
#include "runtime/cluster_test_util.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#if !defined(MAMBASERVE_WITH_CUDA)
#define MAMBASERVE_WITH_CUDA 0
#endif
#if !defined(MAMBASERVE_WITH_NIXL)
#define MAMBASERVE_WITH_NIXL 0
#endif

#if MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

#if MAMBASERVE_WITH_NIXL && MAMBASERVE_WITH_CUDA

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

int cuda_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess)
    return 0;
  return n;
}

#endif

} // namespace

TEST(NixlMigrate, MigratesSessionToOtherWorker) {
#if !(MAMBASERVE_WITH_NIXL && MAMBASERVE_WITH_CUDA)
  GTEST_SKIP() << "requires MAMBASERVE_WITH_NIXL=ON and MAMBASERVE_WITH_CUDA=ON";
#else
  if (cuda_device_count() < 2)
    GTEST_SKIP() << "requires at least 2 CUDA devices";

  ClusterScheduler sched;
  ClusterConfig cfg{
      .model_dir = MAMBA_TEST_MODEL_DIR,
      .device = Device::GPU,
      .n_workers = 2,
      .num_slots = 8,
      .transport = TransportBackend::Nixl,
  };
  Status loaded = start_cluster(sched, cfg);
  ASSERT_TRUE(loaded.ok()) << loaded.message();

  GenerateParams params{.max_new_tokens = 8, .eos_id = 50287};
  StatusOr<uint64_t> id_or = sched.submit({1, 2, 3}, params);
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
  Status m = sched.migrate(req_id, dst);
  ASSERT_TRUE(m.ok()) << m.message();

  ASSERT_TRUE(wait_all_done(sched, {req_id}, std::chrono::seconds(60)))
      << "NIXL migrated session did not finish before timeout";

  Response r = sched.poll(req_id);
  EXPECT_TRUE(r.s.ok()) << r.s.message();
  EXPECT_TRUE(r.done);
  EXPECT_EQ(r.worker_idx, dst);
  EXPECT_EQ(r.gen_seq_len, params.max_new_tokens);
#endif
}
