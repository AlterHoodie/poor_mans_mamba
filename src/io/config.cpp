#include "io/config.h"

#include "io/utils.h"

StatusOr<std::string> read_model_type(const std::string& model_dir) {
  auto config_json_status = json_load(model_dir + "/config.json");
  if (!config_json_status.ok())
    return Status(config_json_status.status());

  return json_at<std::string>(config_json_status.value(), "model_type");
}
