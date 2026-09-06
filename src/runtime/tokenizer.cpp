#include "runtime/tokenizer.h"

#include <fstream>
#include <sstream>

#include "tokenizers_cpp.h"

namespace {
std::string read_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    std::ostringstream ss;
    ss << ifs.rdbuf();

    return ss.str();
}
}  // namespace

Tokenizer::Tokenizer(const Mamba2Config& cfg, const std::string& model_dir)
    : eos_id_(cfg.eos_token_id) {
    auto blob = read_file(model_dir + "/tokenizer.json");
    tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
}

StatusOr<std::vector<int32_t>> Tokenizer::encode(const std::string& prompt) {
    if (!tok_) return Status::InvalidArgument("tokenizer not initialized");
    if (eos_id_ == -1) return Status::InvalidArgument("eos token id not initialized");

    return tok_->Encode(prompt);
}

StatusOr<std::string> Tokenizer::decode(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return Status::InvalidArgument("Empty tokens list");
    return tok_->Decode(tokens);
}

StatusOr<int32_t> Tokenizer::eos_id() const {
    if (eos_id_) return Status::InvalidArgument("eos token id not initialized");
    return eos_id_;
}