#include "io/config.h"

#include "io/utils.h"

ModelKind Mamba2Config::kind() const { return ModelKind::kMamba2; }

ModelKind FalconH1Config::kind() const { return ModelKind::kFalconH1; }

StatusOr<std::string> read_model_type(const std::string& model_dir) {
    auto config_json_status = json_load(model_dir + "/config.json");
    if (!config_json_status.ok()) return Status(config_json_status.status());

    return json_at<std::string>(config_json_status.value(), "model_type");
}
