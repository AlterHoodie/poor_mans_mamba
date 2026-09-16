#include "io/utils.h"

#include <fstream>

StatusOr<json> json_load(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    return Status::NotFound("Cannot open config at path: " + path);
  }

  json j;
  try {
    file >> j;
  } catch (const json::exception& e) {
    return Status::InvalidArgument(std::string("JSON parse error: ") + e.what());
  }

  return j;
}
