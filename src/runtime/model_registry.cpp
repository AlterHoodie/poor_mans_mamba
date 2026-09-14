#include "runtime/model_registry.h"

#include <utility>

#include "io/config.h"
#include "io/config_parser.h"
#include "model/mamba2_loader.h"
#include "model/mamba2_weights.h"
#include "runtime/runner/mamba2runner.h"

namespace {

StatusOr<std::unique_ptr<ModelConfig>> parse_mamba2(const std::string& model_dir) {
    return Mamba2ConfigParser{}.parse(model_dir);
}

StatusOr<std::unique_ptr<Runner>> create_mamba2_runner(const ModelConfig& config,
                                                       const std::string& model_dir, Device device,
                                                       int device_id) {
    const auto* cfg = dynamic_cast<const Mamba2Config*>(&config);
    if (cfg == nullptr) {
        return Status::InvalidArgument("create_mamba2_runner expected Mamba2Config");
    }

    Mamba2Weights weights{};
    Mamba2ModelLoader loader;
    if (Status s = loader.load(*cfg, weights, model_dir, device_id); !s.ok()) {
        return s;
    }

    return std::unique_ptr<Runner>(
        std::make_unique<Mamba2Runner>(*cfg, std::move(weights), device, device_id));
}

StatusOr<std::unique_ptr<ModelConfig>> parse_falcon_h1(const std::string& model_dir) {
    return FalconH1ConfigParser{}.parse(model_dir);
}

StatusOr<std::unique_ptr<Runner>> create_falcon_h1_runner(const ModelConfig&,
                                                          const std::string& /*model_dir*/,
                                                          Device /*device*/, int /*device_id*/) {
    return Status::InvalidArgument("Falcon-H1 runner not implemented yet");
    // TODO include actual runner
}

}  // namespace

const std::unordered_map<std::string, ModelEntry>& ModelRegistry::entries() {
    static const std::unordered_map<std::string, ModelEntry> kEntries = {
        {"mamba2", ModelEntry{ModelKind::kMamba2, &parse_mamba2, &create_mamba2_runner}},
        {"falcon_h1", ModelEntry{ModelKind::kFalconH1, &parse_falcon_h1, &create_falcon_h1_runner}},
    };
    return kEntries;
}

StatusOr<const ModelEntry*> ModelRegistry::lookup(const std::string& model_dir) {
    auto model_type = read_model_type(model_dir);
    if (!model_type.ok()) return Status(model_type.status());

    const auto& registry = entries();
    auto it = registry.find(model_type.value());
    if (it == registry.end()) {
        return Status::InvalidArgument("unsupported model_type: " + model_type.value());
    }
    return &it->second;
}
