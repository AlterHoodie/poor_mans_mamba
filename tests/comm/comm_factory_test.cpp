#include "comm/comm_factory.h"
#include "comm/memcpy_peer_comm_agent.h"
#include "comm/memcpy_transport_control.h"
#include "comm/nccl_transport_control.h"
#include "comm/nixl_transport_control.h"
#include "core/device.h"
#include "core/status.h"
#include "runtime/cluster_config.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#if !defined(MAMBASERVE_WITH_CUDA)
#define MAMBASERVE_WITH_CUDA 0
#endif
#if !defined(MAMBASERVE_WITH_NCCL)
#define MAMBASERVE_WITH_NCCL 0
#endif
#if !defined(MAMBASERVE_WITH_NIXL)
#define MAMBASERVE_WITH_NIXL 0
#endif

namespace {

ClusterConfig base_cfg() {
  return ClusterConfig{
      .model_dir = "unused-for-factory",
      .device = Device::CPU,
      .n_workers = 2,
      .num_slots = 4,
      .transport = TransportBackend::MemcpyPeer,
  };
}

} // namespace

TEST(CommFactory, MemcpyPeerCreatesAgent) {
  ClusterConfig cfg = base_cfg();
  auto ctx_or = create_transport_context(cfg);
  ASSERT_TRUE(ctx_or.ok()) << ctx_or.status().message();

  auto agent_or = create_comm_agent(cfg, 0, ctx_or.value());
  ASSERT_TRUE(agent_or.ok()) << agent_or.status().message();
  ASSERT_NE(agent_or.value(), nullptr);
  EXPECT_EQ(agent_or.value()->device_id(), 0);
  EXPECT_TRUE(dynamic_cast<MemcpyPeerCommAgent*>(agent_or.value().get()) != nullptr);
}

TEST(CommFactory, CpuRejectsNccl) {
  ClusterConfig cfg = base_cfg();
  cfg.transport = TransportBackend::Nccl;

  auto ctx_or = create_transport_context(cfg);
  ASSERT_FALSE(ctx_or.ok());
  EXPECT_EQ(ctx_or.status().code(), Code::kInvalidArgument);

  auto agent_or = create_comm_agent(cfg, 0, nullptr);
  ASSERT_FALSE(agent_or.ok());
  EXPECT_EQ(agent_or.status().code(), Code::kInvalidArgument);
}

TEST(CommFactory, CpuRejectsNixl) {
  ClusterConfig cfg = base_cfg();
  cfg.transport = TransportBackend::Nixl;

  auto ctx_or = create_transport_context(cfg);
  ASSERT_FALSE(ctx_or.ok());
  EXPECT_EQ(ctx_or.status().code(), Code::kInvalidArgument);
}

TEST(CommFactory, GpuNcclWithoutFlagIsRejected) {
  ClusterConfig cfg = base_cfg();
  cfg.device = Device::GPU;
  cfg.transport = TransportBackend::Nccl;

  auto ctx_or = create_transport_context(cfg);
#if MAMBASERVE_WITH_NCCL && MAMBASERVE_WITH_CUDA
  EXPECT_TRUE(ctx_or.ok()) << ctx_or.status().message();
#else
  ASSERT_FALSE(ctx_or.ok());
  EXPECT_TRUE(ctx_or.status().code() == Code::kNotImplemented ||
              ctx_or.status().code() == Code::kInvalidArgument);
#endif
}

TEST(CommFactory, GpuNixlWithoutFlagIsRejected) {
  ClusterConfig cfg = base_cfg();
  cfg.device = Device::GPU;
  cfg.transport = TransportBackend::Nixl;

  auto ctx_or = create_transport_context(cfg);
#if MAMBASERVE_WITH_NIXL && MAMBASERVE_WITH_CUDA
  EXPECT_TRUE(ctx_or.ok()) << ctx_or.status().message();
#else
  ASSERT_FALSE(ctx_or.ok());
  EXPECT_TRUE(ctx_or.status().code() == Code::kNotImplemented ||
              ctx_or.status().code() == Code::kInvalidArgument);
#endif
}

TEST(CommFactory, TransportPlaneMatchesBackend) {
  ClusterConfig cfg = base_cfg();
  auto plane_or = create_transport_control_plane(cfg);
  ASSERT_TRUE(plane_or.ok()) << plane_or.status().message();
  EXPECT_TRUE(dynamic_cast<MemcpyTransportControl*>(plane_or.value().get()) != nullptr);
}

// Captures what a plane asks the scheduler to deliver.
struct RecordingTransportSender final : TransportSender {
  struct Sent {
    size_t worker;
    mambaserve::TransportControl msg;
  };
  std::vector<Sent> sent;
  size_t n = 2;

  void send_transport(size_t worker_idx, const mambaserve::TransportControl& msg) override {
    sent.push_back({worker_idx, msg});
  }
  void broadcast_transport(const mambaserve::TransportControl& msg) override {
    for (size_t i = 0; i < n; ++i)
      sent.push_back({i, msg});
  }
  size_t n_workers() const override { return n; }
};

TEST(MemcpyTransportControl, IsNoop) {
  MemcpyTransportControl plane;
  RecordingTransportSender sender;
  mambaserve::TransportControl empty;
  plane.on_upstream(1, empty, sender);
  EXPECT_TRUE(sender.sent.empty());
}

TEST(MemcpyPeerCommAgent, RecvCopiesFromRemotePtr) {
  MemcpyPeerCommAgent send(0, Device::CPU);
  MemcpyPeerCommAgent recv(1, Device::CPU);

  std::vector<uint8_t> src(64), dst(64, 0);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i + 1);
  constexpr uint64_t kXfer = 42;

  XferDesc sd{.local_ptr = src.data(),
              .bytes = src.size(),
              .peer_device_id = 1,
              .role = XferRole::Send,
              .xfer_id = kXfer,
              .seq_len = 5};
  XferDesc rd{.local_ptr = dst.data(),
              .remote_ptr = src.data(),
              .bytes = dst.size(),
              .peer_device_id = 0,
              .role = XferRole::Recv,
              .xfer_id = kXfer,
              .seq_len = 5};

  auto sh = send.post(sd);
  auto rh = recv.post(rd);
  ASSERT_TRUE(sh.ok());
  ASSERT_TRUE(rh.ok());
  EXPECT_EQ(sh.value().state, XferState::Done);
  EXPECT_EQ(rh.value().state, XferState::Done);
  EXPECT_EQ(dst, src);
  EXPECT_EQ(recv.xfer_seq_len(rh.value().handle), 5);
}

TEST(MemcpyPeerCommAgent, RecvRequiresRemotePtr) {
  MemcpyPeerCommAgent recv(1, Device::CPU);
  std::vector<uint8_t> buf(8, 1);
  XferDesc rd{.local_ptr = buf.data(),
              .bytes = buf.size(),
              .peer_device_id = 0,
              .role = XferRole::Recv,
              .xfer_id = 9};
  auto rh = recv.post(rd);
  EXPECT_FALSE(rh.ok());
}

TEST(NcclTransportControl, BootstrapsEveryRankAndWaitsForAcks) {
  static constexpr int kRanks = 2;
  NcclTransportControl plane(kRanks, std::string(128, 'x'));
  plane.set_ack_timeout(std::chrono::milliseconds(2000));

  // Acks the bootstrap from a "worker" the moment it is delivered.
  struct AckingTransportSender final : TransportSender {
    NcclTransportControl* plane = nullptr;
    std::vector<int> ranks;
    bool ok = true;
    size_t n_workers() const override { return kRanks; }
    void send_transport(size_t worker_idx, const mambaserve::TransportControl& msg) override {
      ASSERT_EQ(msg.body_case(), mambaserve::TransportControl::kNcclCtrl);
      const auto& boot = msg.nccl_ctrl().bootstrap();
      EXPECT_EQ(boot.nranks(), kRanks);
      EXPECT_EQ(static_cast<size_t>(boot.rank()), worker_idx);
      EXPECT_EQ(boot.unique_id().size(), 128u);
      ranks.push_back(boot.rank());

      mambaserve::TransportControl reply;
      auto* ack = reply.mutable_nccl_ctrl()->mutable_bootstrap_ack();
      ack->set_rank(boot.rank());
      ack->set_ok(ok);
      if (!ok)
        ack->set_error("boom");
      plane->on_upstream(worker_idx, reply, *this);
    }
    void broadcast_transport(const mambaserve::TransportControl&) override {}
  };

  AckingTransportSender good;
  good.plane = &plane;
  EXPECT_TRUE(plane.on_cluster_ready(good).ok());
  EXPECT_EQ(good.ranks.size(), static_cast<size_t>(kRanks));

  AckingTransportSender bad;
  bad.plane = &plane;
  bad.ok = false;
  Status s = plane.on_cluster_ready(bad);
  EXPECT_FALSE(s.ok());
}

TEST(NcclTransportControl, TimesOutWithoutAcks) {
  NcclTransportControl plane(2, std::string(128, 'x'));
  plane.set_ack_timeout(std::chrono::milliseconds(50));
  RecordingTransportSender sender; // swallows bootstraps, never acks
  EXPECT_FALSE(plane.on_cluster_ready(sender).ok());
  EXPECT_EQ(sender.sent.size(), 2u);
}

TEST(NixlTransportControl, FansOutPeerMetadataOnceEveryWorkerPublished) {
  NixlTransportControl plane(std::chrono::milliseconds(2000));
  RecordingTransportSender sender;
  sender.n = 3;

  for (int dev = 0; dev < 3; ++dev) {
    mambaserve::TransportControl pub;
    auto* md = pub.mutable_nixl_ctrl()->mutable_publish_md();
    md->set_device_id(dev);
    md->set_md("md-" + std::to_string(dev));
    plane.on_upstream(static_cast<size_t>(dev), pub, sender);
  }
  ASSERT_TRUE(sender.sent.empty()); // publishing alone sends nothing

  ASSERT_TRUE(plane.on_cluster_ready(sender).ok());

  // 3 workers x 2 peers each, then one peers_finalized per worker.
  std::vector<std::vector<std::string>> peers(3);
  std::vector<int> finalized(3, 0);
  for (const auto& s : sender.sent) {
    ASSERT_EQ(s.msg.body_case(), mambaserve::TransportControl::kNixlCtrl);
    const auto& ctrl = s.msg.nixl_ctrl();
    if (ctrl.body_case() == mambaserve::NixlControl::kPeerMd) {
      EXPECT_NE(static_cast<size_t>(ctrl.peer_md().device_id()), s.worker);
      peers[s.worker].push_back(ctrl.peer_md().md());
    } else if (ctrl.body_case() == mambaserve::NixlControl::kPeersFinalized) {
      EXPECT_TRUE(ctrl.peers_finalized());
      finalized[s.worker]++;
    }
  }
  for (size_t w = 0; w < 3; ++w) {
    EXPECT_EQ(peers[w].size(), 2u);
    EXPECT_EQ(finalized[w], 1);
  }
}

TEST(NixlTransportControl, TimesOutWhenAWorkerNeverPublishes) {
  NixlTransportControl plane(std::chrono::milliseconds(50));
  RecordingTransportSender sender;
  mambaserve::TransportControl pub;
  pub.mutable_nixl_ctrl()->mutable_publish_md()->set_device_id(0);
  pub.mutable_nixl_ctrl()->mutable_publish_md()->set_md("md-0");
  plane.on_upstream(0, pub, sender);
  EXPECT_FALSE(plane.on_cluster_ready(sender).ok());
  EXPECT_TRUE(sender.sent.empty());
}

TEST(NixlTransportControl, IgnoresNonPublishUpstream) {
  NixlTransportControl plane;
  RecordingTransportSender sender;
  mambaserve::TransportControl peer;
  peer.mutable_nixl_ctrl()->mutable_peer_md()->set_device_id(1);
  peer.mutable_nixl_ctrl()->mutable_peer_md()->set_md("x");
  plane.on_upstream(1, peer, sender);
  EXPECT_TRUE(sender.sent.empty());
}

TEST(WorkerMode, ProcessModeRejectsMemcpyPeer) {
  ClusterConfig cfg = base_cfg();
  cfg.worker_mode = WorkerMode::Process;
  Status s = validate_cluster_config(cfg);
  ASSERT_FALSE(s.ok());
  EXPECT_EQ(s.code(), Code::kInvalidArgument);
  EXPECT_NE(s.message().find("MemcpyPeer"), std::string::npos);

  cfg.device = Device::GPU; // still MemcpyPeer: thread-only regardless of device
  s = validate_cluster_config(cfg);
  ASSERT_FALSE(s.ok());
  EXPECT_NE(s.message().find("MemcpyPeer"), std::string::npos);

  EXPECT_FALSE(create_transport_context(cfg).ok());
}

TEST(WorkerMode, ProcessModeRejectsCpu) {
  ClusterConfig cfg = base_cfg();
  cfg.worker_mode = WorkerMode::Process;
  cfg.transport = TransportBackend::Nccl;
  Status s = validate_cluster_config(cfg);
  ASSERT_FALSE(s.ok());
  EXPECT_EQ(s.code(), Code::kInvalidArgument);
}

TEST(WorkerMode, ThreadModeStillAcceptsMemcpyPeer) {
  ClusterConfig cfg = base_cfg();
  ASSERT_EQ(cfg.worker_mode, WorkerMode::Thread);
  EXPECT_TRUE(validate_cluster_config(cfg).ok());
}

TEST(WorkerMode, ProcessModeAcceptsGpuNcclAndNixlWhenBuilt) {
  for (TransportBackend t : {TransportBackend::Nccl, TransportBackend::Nixl}) {
    ClusterConfig cfg = base_cfg();
    cfg.device = Device::GPU;
    cfg.transport = t;
    cfg.worker_mode = WorkerMode::Process;
    Status s = validate_cluster_config(cfg);
#if MAMBASERVE_WITH_CUDA
    const bool built = (t == TransportBackend::Nccl) ? MAMBASERVE_WITH_NCCL : MAMBASERVE_WITH_NIXL;
    EXPECT_EQ(s.ok(), built != 0) << s.message();
#else
    EXPECT_FALSE(s.ok());
#endif
  }
}

TEST(CommFactory, FinalizePeersNoOpForMemcpyPeer) {
  ClusterConfig cfg = base_cfg();
  auto ctx_or = create_transport_context(cfg);
  ASSERT_TRUE(ctx_or.ok());
  EXPECT_TRUE(finalize_transport_peers(ctx_or.value()).ok());
}
