#include "runtime/tokenizer.h"

#include <tokenizers_cpp.h>

#include <fstream>
#include <sstream>

namespace {
std::string read_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    std::ostringstream ss;
    ss << ifs.rdbuf();

    return ss.str();
}
}  // namespace

struct Tokenizer::Impl {
    int32_t eos_id = -1;
    std::unique_ptr<tokenizers::Tokenizer> tok;
};

Tokenizer::Tokenizer(const ModelConfig& cfg, const std::string& model_dir)
    : impl_(std::make_unique<Impl>()) {
    auto blob = read_file(model_dir + "/tokenizer.json");

    impl_->eos_id = cfg.eos_token_id;
    impl_->tok = tokenizers::Tokenizer::FromBlobJSON(blob);
}

StatusOr<std::vector<int32_t>> Tokenizer::encode(const std::string& prompt) {
    if (!impl_ || !impl_->tok) return Status::InvalidArgument("tokenizer not initialized");
    if (impl_->eos_id < 0) return Status::InvalidArgument("eos token id not initialized");

    return impl_->tok->Encode(prompt);
}

StatusOr<std::string> Tokenizer::decode(const std::vector<int32_t>& tokens) {
    if (!impl_ || !impl_->tok) return Status::InvalidArgument("tokenizer not initialized");
    if (tokens.empty()) return Status::InvalidArgument("Empty tokens list");
    return impl_->tok->Decode(tokens);
}

StatusOr<int32_t> Tokenizer::eos_id() const {
    if (impl_->eos_id < 0) return Status::InvalidArgument("eos token id not initialized");
    return impl_->eos_id;
}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;