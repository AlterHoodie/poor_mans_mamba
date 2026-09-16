#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "io/config.h"
#include "runtime/runner/runner.h"

#include <memory>
#include <string>
#include <unordered_map>

// model_type → entry { kind, parse, create_runner }
using ParseFn = StatusOr<std::unique_ptr<ModelConfig>> (*)(const std::string& model_dir,
                                                           int max_seq_length);

using CreateRunnerFn = StatusOr<std::unique_ptr<Runner>> (*)(const ModelConfig& config,
                                                             const std::string& model_dir,
                                                             Device device, int num_slots,
                                                             int device_id);

struct ModelEntry {
  ModelKind kind;
  // function pointer to the model's parser
  ParseFn parse;
  // function pointer to the model's runner
  CreateRunnerFn create_runner;
};

class ModelRegistry {
public:
  // Read config.json model_type and look up the matching entry.
  static StatusOr<const ModelEntry*> lookup(const std::string& model_dir);

  static const std::unordered_map<std::string, ModelEntry>& entries();
};
