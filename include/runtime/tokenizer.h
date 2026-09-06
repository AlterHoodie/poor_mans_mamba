#include <tokenizers_cpp.h>

#include <vector>

#include "core/status.h"
#include "io/config_parser.h"

class Tokenizer {
   private:
    int32_t eos_id_ = -1;
    std::unique_ptr<tokenizers::Tokenizer> tok_;

   public:
    Tokenizer(const Mamba2Config& cfg, const std::string& model_dir);

    // tokenize
    StatusOr<std::vector<int32_t>> encode(const std::string& prompt);
    // decode
    StatusOr<std::string> decode(const std::vector<int32_t>& tokens);

    StatusOr<int32_t> eos_id() const;
};