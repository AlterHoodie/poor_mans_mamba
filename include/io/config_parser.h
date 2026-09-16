#pragma once

#include "core/status.h"
#include "io/config.h"

#include <memory>
#include <string>

class IConfigParser {
public:
  virtual ~IConfigParser() = default;
  virtual StatusOr<std::unique_ptr<ModelConfig>> parse(const std::string& model_dir,
                                                       int max_seq_length) const = 0;
};

class Mamba2ConfigParser final : public IConfigParser {
public:
  StatusOr<std::unique_ptr<ModelConfig>> parse(const std::string& model_dir,
                                               int max_seq_length) const override;
};

class FalconH1ConfigParser final : public IConfigParser {
public:
  StatusOr<std::unique_ptr<ModelConfig>> parse(const std::string& model_dir,
                                               int max_seq_length) const override;
};
