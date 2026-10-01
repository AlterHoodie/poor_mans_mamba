#include "comm/comm_factory.h"
#include "comm/memcpy_peer_comm_agent.h"
#include "core/device.h"
#include "core/status.h"
#include "runtime/cluster_config.h"

#include <gtest/gtest.h>

#include <memory>

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

TEST(CommFactory, FinalizePeersNoOpForMemcpyPeer) {
  ClusterConfig cfg = base_cfg();
  auto ctx_or = create_transport_context(cfg);
  ASSERT_TRUE(ctx_or.ok());
  EXPECT_TRUE(finalize_transport_peers(ctx_or.value()).ok());
}
