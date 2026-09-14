#pragma once

#include <memory>
#include <string>

#include "core/status.h"
#include "io/config.h"

class IConfigParser {
   public:
    virtual ~IConfigParser() = default;
    virtual StatusOr<std::unique_ptr<ModelConfig>> parse(const std::string& model_dir) const = 0;
};

class Mamba2ConfigParser final : public IConfigParser {
   public:
    StatusOr<std::unique_ptr<ModelConfig>> parse(const std::string& model_dir) const override;
};

class FalconH1ConfigParser final : public IConfigParser {
   public:
    StatusOr<std::unique_ptr<ModelConfig>> parse(const std::string& model_dir) const override;
};
