#pragma once

#include "core/device.h"
#include "core/status.h"
#include "io/config.h"
#include "runtime/runner/runner.h"

#include <memory>
#include <string>

// Opened model: config + dir + bound create_runner (no re-pass of model_dir).
// create_runner loads weights onto the given allocator (sets AllocatorScope).
struct ModelEntry {
  std::string model_dir;
  std::unique_ptr<ModelConfig> cfg;

  StatusOr<std::unique_ptr<Runner>> create_runner(DeviceAllocator& alloc) const;

private:
  friend class ModelRegistry;

  using CreateRunnerFn = StatusOr<std::unique_ptr<Runner>> (*)(const ModelConfig& config,
                                                              const std::string& model_dir,
                                                              DeviceAllocator& alloc);
  CreateRunnerFn create_runner_fn_ = nullptr;
};

class ModelRegistry {
public:
  // lookup model_type, parse config, return entry bound to model_dir
  static StatusOr<ModelEntry> open(const std::string& model_dir, int max_seq_length);
};
