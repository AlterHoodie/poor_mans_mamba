#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "io/config.h"
#include "runtime/runner/runner.h"

#include <memory>
#include <string>
#include <unordered_map>

// model_type → entry { parse, create_runner }
using ParseFn = StatusOr<std::unique_ptr<ModelConfig>> (*)(const std::string& model_dir,
                                                           int max_seq_length);

using CreateRunnerFn = StatusOr<std::unique_ptr<Runner>> (*)(const ModelConfig& config,
                                                             const std::string& model_dir,
                                                             Device device, int num_slots,
                                                             int device_id);

struct ModelEntry {
  ParseFn parse;
  CreateRunnerFn create_runner;
};

class ModelRegistry {
public:
  static StatusOr<const ModelEntry*> lookup(const std::string& model_dir);

  static const std::unordered_map<std::string, ModelEntry>& entries();
};
