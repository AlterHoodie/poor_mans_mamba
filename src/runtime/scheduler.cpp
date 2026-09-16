#include "runtime/scheduler.h"

#include "ops/cpu/reductions.h"

StatusOr<int32_t> Scheduler::sample_(Tensor& logits) { return argmax(logits); }

StatusOr<std::vector<int32_t>> Scheduler::generate(std::span<const int32_t> tokens,
                                                   const GenerateParams& params) {
  if (tokens.empty())
    return Status::InvalidArgument("token_ids cannot be empty");
  if (params.eos_id < 0)
    return Status::InvalidArgument("EOS token_id not set");
  if (params.max_new_tokens <= 0) {
    return Status::InvalidArgument("max_new_tokens cannot be 0 or negative");
  }

  PrefillResult prefill;
  ASSIGN_OR_RETURN(prefill, runner_.prefill(tokens));

  std::vector<int32_t> out;
  out.reserve(static_cast<size_t>(params.max_new_tokens));

  int32_t token = 0;
  ASSIGN_OR_RETURN(token, sample_(prefill.logits));
  out.push_back(token);

  for (int i = 1; i < params.max_new_tokens && token != params.eos_id; ++i) {
    DecodeResult decode;
    ASSIGN_OR_RETURN(decode, runner_.decode(prefill.cache, token));
    ASSIGN_OR_RETURN(token, sample_(decode.logits));
    out.push_back(token);
  }

  if (Status s = runner_.release(prefill.cache); !s.ok())
    return s;
  return out;
}
