#include "runtime/model_registry.h"

#include "core/device.h"
#include "io/config.h"
#include "io/config_parser.h"
#include "model/falconh1_loader.h"
#include "model/falconh1_weights.h"
#include "model/mamba2_loader.h"
#include "model/mamba2_weights.h"
#include "ops/backend.h"
#include "runtime/runner/falconh1_runner.h"
#include "runtime/runner/mamba2_runner.h"

#include <unordered_map>
#include <utility>

namespace {

using ParseFn = StatusOr<std::unique_ptr<ModelConfig>> (*)(const std::string& model_dir,
                                                           int max_seq_length);
using CreateRunnerFn = StatusOr<std::unique_ptr<Runner>> (*)(const ModelConfig& config,
                                                            const std::string& model_dir,
                                                            DeviceAllocator& alloc);

struct ModelTypeEntry {
  ParseFn parse;
  CreateRunnerFn create_runner;
};

StatusOr<std::unique_ptr<ModelConfig>> parse_mamba2(const std::string& model_dir,
                                                    int max_seq_length) {
  return Mamba2ConfigParser{}.parse(model_dir, max_seq_length);
}

StatusOr<std::unique_ptr<Runner>> create_mamba2_runner(const ModelConfig& config,
                                                       const std::string& model_dir,
                                                       DeviceAllocator& alloc) {
  if (config.layout != ArchLayout::MambaOnly) {
    return Status::InvalidArgument("create_mamba2_runner expected MambaOnly layout");
  }

  Mamba2Weights weights{};
  Mamba2ModelLoader loader;
  if (Status s = loader.load(config, weights, model_dir, alloc.device_id()); !s.ok()) {
    return s;
  }

  return std::unique_ptr<Runner>(
      std::make_unique<Mamba2Runner>(config, std::move(weights), ops_for(alloc.kind())));
}

StatusOr<std::unique_ptr<ModelConfig>> parse_falcon_h1(const std::string& model_dir,
                                                       int max_seq_length) {
  return FalconH1ConfigParser{}.parse(model_dir, max_seq_length);
}

StatusOr<std::unique_ptr<Runner>> create_falcon_h1_runner(const ModelConfig& config,
                                                          const std::string& model_dir,
                                                          DeviceAllocator& alloc) {
  if (config.layout != ArchLayout::ParallelHybrid) {
    return Status::InvalidArgument("create_falcon_h1_runner expected ParallelHybrid layout");
  }

  FalconH1Weights weights{};
  FalconH1ModelLoader loader;
  if (Status s = loader.load(config, weights, model_dir, alloc.device_id()); !s.ok()) {
    return s;
  }

  return std::unique_ptr<Runner>(
      std::make_unique<FalconH1Runner>(config, std::move(weights), ops_for(alloc.kind())));
}

const std::unordered_map<std::string, ModelTypeEntry>& type_entries() {
  static const std::unordered_map<std::string, ModelTypeEntry> kEntries = {
      {"mamba2", ModelTypeEntry{&parse_mamba2, &create_mamba2_runner}},
      {"falcon_h1", ModelTypeEntry{&parse_falcon_h1, &create_falcon_h1_runner}},
  };
  return kEntries;
}

} // namespace

StatusOr<std::unique_ptr<Runner>> ModelEntry::create_runner(DeviceAllocator& alloc) const {
  if (!cfg)
    return Status::RuntimeError("ModelEntry has no config");
  if (!create_runner_fn_)
    return Status::RuntimeError("ModelEntry has no create_runner");
  AllocatorScope scope(&alloc);
  return create_runner_fn_(*cfg, model_dir, alloc);
}

StatusOr<ModelEntry> ModelRegistry::open(const std::string& model_dir, int max_seq_length) {
  auto model_type = read_model_type(model_dir);
  if (!model_type.ok())
    return Status(model_type.status());

  const auto& map = type_entries();
  auto it = map.find(model_type.value());
  if (it == map.end()) {
    return Status::NotFound("unsupported model_type: " + model_type.value());
  }

  auto cfg_or = it->second.parse(model_dir, max_seq_length);
  if (!cfg_or.ok())
    return Status(cfg_or.status());

  ModelEntry entry;
  entry.model_dir = model_dir;
  entry.cfg = std::move(cfg_or.value());
  entry.create_runner_fn_ = it->second.create_runner;
  return entry;
}
