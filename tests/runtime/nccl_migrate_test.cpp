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
#if !defined(MAMBASERVE_WITH_NCCL)
#define MAMBASERVE_WITH_NCCL 0
#endif

#if MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

#if MAMBASERVE_WITH_NCCL && MAMBASERVE_WITH_CUDA

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

ClusterConfig gpu_nccl_cfg(int n_workers) {
  return ClusterConfig{
      .model_dir = MAMBA_TEST_MODEL_DIR,
      .device = Device::GPU,
      .n_workers = n_workers,
      .num_slots = 8,
      .transport = TransportBackend::Nccl,
  };
}

bool wait_mid_decode(ClusterScheduler& sched, uint64_t req_id, size_t* src,
                     std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    Response r = sched.poll(req_id);
    if (!r.s.ok())
      return false;
    if (r.gen_seq_len >= 1 && !r.done) {
      *src = r.worker_idx;
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

#endif

} // namespace

TEST(NcclMigrate, MigratesSessionToOtherWorker) {
#if !(MAMBASERVE_WITH_NCCL && MAMBASERVE_WITH_CUDA)
  GTEST_SKIP() << "requires MAMBASERVE_WITH_NCCL=ON and MAMBASERVE_WITH_CUDA=ON";
#else
  if (cuda_device_count() < 2)
    GTEST_SKIP() << "requires at least 2 CUDA devices";

  ClusterScheduler sched;
  Status loaded = start_cluster(sched, gpu_nccl_cfg(2));
  ASSERT_TRUE(loaded.ok()) << loaded.message();

  GenerateParams params{.max_new_tokens = 8, .eos_id = 50287};
  StatusOr<uint64_t> id_or = sched.submit({1, 2, 3}, params);
  ASSERT_TRUE(id_or.ok()) << id_or.status().message();
  const uint64_t req_id = id_or.value();

  size_t src = 0;
  ASSERT_TRUE(wait_mid_decode(sched, req_id, &src, std::chrono::seconds(60)))
      << "timed out waiting for decode";

  const size_t dst = 1 - src;
  Status m = sched.migrate(req_id, dst);
  ASSERT_TRUE(m.ok()) << m.message();

  ASSERT_TRUE(wait_all_done(sched, {req_id}, std::chrono::seconds(60)))
      << "NCCL migrated session did not finish before timeout";

  Response r = sched.poll(req_id);
  EXPECT_TRUE(r.s.ok()) << r.s.message();
  EXPECT_TRUE(r.done);
  EXPECT_EQ(r.worker_idx, dst);
  EXPECT_EQ(r.gen_seq_len, params.max_new_tokens);
#endif
}

TEST(NcclMigrate, GoldenContinuationAfterMigrate) {
#if !(MAMBASERVE_WITH_NCCL && MAMBASERVE_WITH_CUDA)
  GTEST_SKIP() << "requires MAMBASERVE_WITH_NCCL=ON and MAMBASERVE_WITH_CUDA=ON";
#else
  if (cuda_device_count() < 2)
    GTEST_SKIP() << "requires at least 2 CUDA devices";

  GenerateParams params{.max_new_tokens = 6, .eos_id = 50287};
  const std::vector<int32_t> prompt = {1, 2, 3, 4};

  std::vector<int32_t> baseline;
  {
    ClusterScheduler sched;
    ASSERT_TRUE(start_cluster(sched, gpu_nccl_cfg(1)).ok());
    StatusOr<uint64_t> id_or = sched.submit(prompt, params);
    ASSERT_TRUE(id_or.ok()) << id_or.status().message();
    ASSERT_TRUE(wait_all_done(sched, {id_or.value()}, std::chrono::seconds(60)));
    Response r = sched.poll(id_or.value());
    ASSERT_TRUE(r.s.ok()) << r.s.message();
    ASSERT_EQ(r.gen_seq_len, params.max_new_tokens);
    baseline = r.tokens;
  }

  ClusterScheduler sched;
  ASSERT_TRUE(start_cluster(sched, gpu_nccl_cfg(2)).ok());
  StatusOr<uint64_t> id_or = sched.submit(prompt, params);
  ASSERT_TRUE(id_or.ok()) << id_or.status().message();
  const uint64_t req_id = id_or.value();

  size_t src = 0;
  ASSERT_TRUE(wait_mid_decode(sched, req_id, &src, std::chrono::seconds(60)))
      << "timed out waiting for decode";

  const size_t dst = 1 - src;
  ASSERT_TRUE(sched.migrate(req_id, dst).ok());
  ASSERT_TRUE(wait_all_done(sched, {req_id}, std::chrono::seconds(60)));

  Response r = sched.poll(req_id);
  ASSERT_TRUE(r.s.ok()) << r.s.message();
  EXPECT_EQ(r.worker_idx, dst);
  EXPECT_EQ(r.tokens, baseline);
#endif
}
