#pragma once

#include "core/device.h"

#include <string>

enum class TransportBackend { MemcpyPeer, Nccl, Nixl };

// Deployment / topology / transport for a ClusterScheduler instance.
// Model architecture stays in ModelRegistry / ModelConfig.
struct ClusterConfig {
  std::string model_dir;
  Device device = Device::CPU;
  int n_workers = 1;
  int num_slots = 8;
  TransportBackend transport = TransportBackend::MemcpyPeer;
  // Per-slot capacity; sets cache slot bytes and therefore bytes moved per migrate.
  int max_seq_length = 2048;
};
