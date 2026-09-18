#include "runtime/model_registry.h"

#include "core/device.h"
#include "io/config.h"
#include "io/config_parser.h"
#include "model/falconh1_loader.h"
#include "model/falconh1_weights.h"
#include "model/mamba2_loader.h"
#include "model/mamba2_weights.h"
#include "ops/backend.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/runner/falconh1_runner.h"
#include "runtime/runner/mamba2_runner.h"

#include <utility>

namespace {

StatusOr<std::unique_ptr<ModelConfig>> parse_mamba2(const std::string& model_dir,
                                                    int max_seq_length) {
  return Mamba2ConfigParser{}.parse(model_dir, max_seq_length);
}

StatusOr<std::unique_ptr<Runner>> create_mamba2_runner(const ModelConfig& config,
                                                       const std::string& model_dir, Device device,
                                                       int num_slots, int device_id) {
  if (config.layout != ArchLayout::MambaOnly) {
    return Status::InvalidArgument("create_mamba2_runner expected MambaOnly layout");
  }

  std::unique_ptr<DeviceAllocator> alloc;
  ASSIGN_OR_RETURN(alloc, create_device_allocator(device, device_id));

  Mamba2Weights weights{};
  {
    AllocatorScope scope(alloc.get());
    Mamba2ModelLoader loader;
    if (Status s = loader.load(config, weights, model_dir, device_id); !s.ok()) {
      return s;
    }
  }

  DeviceAllocator* raw = alloc.get();
  auto pool = create_cache_pool(config, raw, num_slots, create_cache_layout);
  if (!pool.ok())
    return pool.status();

  return std::unique_ptr<Runner>(std::make_unique<Mamba2Runner>(
      config, std::move(weights), std::move(pool.value()), std::move(alloc), ops_for(device)));
}

StatusOr<std::unique_ptr<ModelConfig>> parse_falcon_h1(const std::string& model_dir,
                                                       int max_seq_length) {
  return FalconH1ConfigParser{}.parse(model_dir, max_seq_length);
}

StatusOr<std::unique_ptr<Runner>> create_falcon_h1_runner(const ModelConfig& config,
                                                          const std::string& model_dir,
                                                          Device device, int num_slots,
                                                          int device_id) {
  if (config.layout != ArchLayout::ParallelHybrid) {
    return Status::InvalidArgument("create_falcon_h1_runner expected ParallelHybrid layout");
  }

  std::unique_ptr<DeviceAllocator> alloc;
  ASSIGN_OR_RETURN(alloc, create_device_allocator(device, device_id));

  FalconH1Weights weights{};
  {
    AllocatorScope scope(alloc.get());
    FalconH1ModelLoader loader;
    if (Status s = loader.load(config, weights, model_dir, device_id); !s.ok()) {
      return s;
    }
  }

  // TODO: Dont like this method of passing raw value of an unique_ptr
  DeviceAllocator* raw = alloc.get();
  auto pool = create_cache_pool(config, raw, num_slots, create_cache_layout);
  if (!pool.ok())
    return pool.status();

  return std::unique_ptr<Runner>(std::make_unique<FalconH1Runner>(
      config, std::move(weights), std::move(pool.value()), std::move(alloc), ops_for(device)));
}

} // namespace

const std::unordered_map<std::string, ModelEntry>& ModelRegistry::entries() {
  static const std::unordered_map<std::string, ModelEntry> kEntries = {
      {"mamba2", ModelEntry{&parse_mamba2, &create_mamba2_runner}},
      {"falcon_h1", ModelEntry{&parse_falcon_h1, &create_falcon_h1_runner}},
  };
  return kEntries;
}

StatusOr<const ModelEntry*> ModelRegistry::lookup(const std::string& model_dir) {
  auto model_type = read_model_type(model_dir);
  if (!model_type.ok())
    return Status(model_type.status());

  const auto& map = entries();
  auto it = map.find(model_type.value());
  if (it == map.end()) {
    return Status::NotFound("unsupported model_type: " + model_type.value());
  }
  return &it->second;
}
