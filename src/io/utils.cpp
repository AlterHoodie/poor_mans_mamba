#include "io/utils.h"

#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

namespace {

void replace_all(std::string& s, std::string_view from, std::string_view to) {
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
}

// Hugging Face configs sometimes emit Infinity / NaN (not strict JSON).
std::string sanitize_hf_json(std::string content) {
  replace_all(content, "-Infinity", "-1e308");
  replace_all(content, "Infinity", "1e308");
  replace_all(content, "NaN", "null");
  return content;
}

} // namespace

StatusOr<json> json_load(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    return Status::NotFound("Cannot open config at path: " + path);
  }

  const std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

  try {
    return json::parse(sanitize_hf_json(content));
  } catch (const json::exception& e) {
    return Status::InvalidArgument(std::string("JSON parse error: ") + e.what());
  }
}
