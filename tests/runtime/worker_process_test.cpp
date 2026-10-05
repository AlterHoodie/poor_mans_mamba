#include "core/device.h"
#include "runtime/cluster_config.h"
#include "runtime/ipc/process_comm.h"
#include "worker_process.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <unistd.h>

// These tests spawn real worker processes (this test binary re-exec'd with
// --mambaserve-worker, see tests/test_main.cpp). They use the spawn helper directly,
// so they exercise the process boundary on a CPU machine even though a full
// process-mode cluster needs GPU + NCCL/NIXL.

namespace {

ClusterConfig cpu_cfg(const char* model_dir) {
  return ClusterConfig{
      .model_dir = model_dir,
      .device = Device::CPU,
      .n_workers = 1,
      .num_slots = 4,
      .transport = TransportBackend::MemcpyPeer,
      .max_seq_length = 256,
  };
}

struct Spawned {
  std::unique_ptr<WorkerProcess> process;
  std::unique_ptr<ProcessChannel> chan;
};

Spawned spawn(const ClusterConfig& cfg) {
  auto fds_or = make_process_socketpair();
  EXPECT_TRUE(fds_or.ok());
  const ProcessChannelFds fds = fds_or.value();
  auto proc_or = spawn_worker_process(cfg, /*index=*/0, /*device_id=*/0, fds.child_fd);
  ::close(fds.child_fd);
  EXPECT_TRUE(proc_or.ok()) << (proc_or.ok() ? "" : proc_or.status().message());
  Spawned s;
  s.chan = std::make_unique<ProcessChannel>(fds.parent_fd);
  if (proc_or.ok())
    s.process = std::move(proc_or.value());
  return s;
}

mambaserve::Envelope cmd_env(mambaserve::Command cmd) {
  mambaserve::Envelope env;
  *env.mutable_cmd() = std::move(cmd);
  return env;
}

} // namespace

TEST(WorkerProcess, ServesPrefillDecodeReleaseThenExitsOnShutdown) {
  Spawned w = spawn(cpu_cfg(MAMBA_TEST_MODEL_DIR));
  ASSERT_NE(w.process, nullptr);

  auto ready = w.chan->recv();
  ASSERT_TRUE(ready.ok()) << ready.status().message();
  ASSERT_EQ(ready.value().body_case(), mambaserve::Envelope::kEvent);
  ASSERT_EQ(ready.value().event().body_case(), mambaserve::Event::kReady);
  EXPECT_EQ(ready.value().event().ready().status().code(), mambaserve::CODE_OK)
      << ready.value().event().ready().status().message();

  mambaserve::Command prefill;
  auto* p = prefill.mutable_prefill();
  p->set_req_id(1);
  p->set_max_new_tokens(2);
  p->set_eos_id(50287);
  for (int t : {1, 2, 3})
    p->add_tokens(t);
  ASSERT_TRUE(w.chan->send(cmd_env(prefill)).ok());

  auto pref = w.chan->recv();
  ASSERT_TRUE(pref.ok()) << pref.status().message();
  ASSERT_EQ(pref.value().event().body_case(), mambaserve::Event::kPrefill);
  EXPECT_EQ(pref.value().event().prefill().status().code(), mambaserve::CODE_OK);
  const int32_t first = pref.value().event().prefill().token();
  EXPECT_GE(first, 0);

  mambaserve::Command decode;
  decode.mutable_decode()->set_req_id(1);
  decode.mutable_decode()->set_token(first);
  ASSERT_TRUE(w.chan->send(cmd_env(decode)).ok());
  auto dec = w.chan->recv();
  ASSERT_TRUE(dec.ok());
  ASSERT_EQ(dec.value().event().body_case(), mambaserve::Event::kDecode);
  EXPECT_EQ(dec.value().event().decode().status().code(), mambaserve::CODE_OK);

  mambaserve::Command release;
  release.mutable_release()->set_req_id(1);
  ASSERT_TRUE(w.chan->send(cmd_env(release)).ok());
  auto rel = w.chan->recv();
  ASSERT_TRUE(rel.ok());
  ASSERT_EQ(rel.value().event().body_case(), mambaserve::Event::kRelease);

  mambaserve::Command shutdown;
  shutdown.mutable_shutdown();
  ASSERT_TRUE(w.chan->send(cmd_env(shutdown)).ok());
  w.chan->close();
  EXPECT_TRUE(w.process->wait_for_exit(std::chrono::seconds(10)));
  EXPECT_EQ(w.process->exit_code(), 0);
}

TEST(WorkerProcess, ReportsInitFailureInReadyAndExitsNonZero) {
  Spawned w = spawn(cpu_cfg("/nonexistent/model/dir"));
  ASSERT_NE(w.process, nullptr);

  auto ready = w.chan->recv();
  ASSERT_TRUE(ready.ok()) << ready.status().message();
  ASSERT_EQ(ready.value().event().body_case(), mambaserve::Event::kReady);
  EXPECT_NE(ready.value().event().ready().status().code(), mambaserve::CODE_OK);

  EXPECT_TRUE(w.process->wait_for_exit(std::chrono::seconds(10)));
  EXPECT_EQ(w.process->exit_code(), 1);
}

TEST(WorkerProcess, ExitsWhenParentChannelCloses) {
  Spawned w = spawn(cpu_cfg(MAMBA_TEST_MODEL_DIR));
  ASSERT_NE(w.process, nullptr);

  auto ready = w.chan->recv();
  ASSERT_TRUE(ready.ok());
  ASSERT_EQ(ready.value().event().ready().status().code(), mambaserve::CODE_OK);

  w.chan->close(); // no ShutdownCmd: EOF alone must stop the worker
  EXPECT_TRUE(w.process->wait_for_exit(std::chrono::seconds(10)));
  EXPECT_EQ(w.process->exit_code(), 0);
}

TEST(WorkerProcess, MaybeRunIgnoresOrdinaryArguments) {
  char prog[] = "tests";
  char flag[] = "--gtest_filter=*";
  char* argv[] = {prog, flag, nullptr};
  EXPECT_FALSE(maybe_run_worker_process(2, argv).has_value());
}

TEST(WorkerProcess, MaybeRunRejectsMalformedWorkerArguments) {
  char prog[] = "tests";
  char flag[] = "--mambaserve-worker";
  char bad[] = "--index=notanumber";
  char* argv[] = {prog, flag, bad, nullptr};
  auto rc = maybe_run_worker_process(3, argv);
  ASSERT_TRUE(rc.has_value());
  EXPECT_EQ(*rc, 2);
}
