#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "status.h"

using json = nlohmann::json;

StatusOr<json> json_load(const std::string& path);

template <typename T>
StatusOr<T> json_at(const json& j, const std::string& key) {
    try {
        return j.at(key).get<T>();
    } catch (const json::exception& e) {
        return Status::InvalidArgument("config field '" + key + "': " + e.what());
    }
}
