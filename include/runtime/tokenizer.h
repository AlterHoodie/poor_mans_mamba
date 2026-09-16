#pragma once

#include "core/status.h"
#include "io/config.h"

#include <memory>
#include <vector>

class Tokenizer {
private:
  struct Impl; // incomplete declaration
  std::unique_ptr<Impl> impl_;

public:
  Tokenizer(const ModelConfig& cfg, const std::string& model_dir);
  ~Tokenizer();

  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;

  // tokenize
  StatusOr<std::vector<int32_t>> encode(const std::string& prompt);
  // decode
  StatusOr<std::string> decode(const std::vector<int32_t>& tokens);

  StatusOr<int32_t> eos_id() const;
};