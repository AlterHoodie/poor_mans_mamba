#include "runtime/scheduler.h"

#include "ops/cpu/reductions.h"

#include <string>
#include <vector>

#ifdef MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

bool is_kv_cache_overflow(const Status& s) {
  return s.code() == Code::kKvCacheOverflow;
}

StatusOr<std::vector<int32_t>> finish_generate(Runner& runner, CacheHandle& cache,
                                               std::vector<int32_t> out) {
  if (Status s = runner.release(cache); !s.ok())
    return s;
  return out;
}

StatusOr<int32_t> argmax_gpu_d2h(Tensor& logits) {
#ifdef MAMBASERVE_WITH_CUDA
  StatusOr<int64_t> n_or = logits.numel();
  if (!n_or.ok())
    return Status(n_or.status());
  const int64_t n = n_or.value();
  if (n <= 0)
    return Status::InvalidArgument("empty logits");

  std::vector<float> host(static_cast<size_t>(n));
  cudaError_t err =
      cudaMemcpy(host.data(), logits.buffer.ptr, static_cast<size_t>(n) * sizeof(float),
                 cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaMemcpy D2H failed: ") +
                                   cudaGetErrorString(err));
  }

  int32_t best = 0;
  float best_v = host[0];
  for (int64_t i = 1; i < n; ++i) {
    if (host[static_cast<size_t>(i)] > best_v) {
      best_v = host[static_cast<size_t>(i)];
      best = static_cast<int32_t>(i);
    }
  }
  return best;
#else
  (void)logits;
  return Status::InvalidArgument("GPU sampling requires CUDA build");
#endif
}

} // namespace

StatusOr<int32_t> Scheduler::sample_(Tensor& logits) {
  if (logits.buffer.device == Device::CPU)
    return argmax(logits);
  if (logits.buffer.device == Device::GPU)
    return argmax_gpu_d2h(logits);
  return Status::InvalidArgument("logits device not supported for sampling");
}

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
  {
    StatusOr<int32_t> sampled = sample_(prefill.logits);
    if (!sampled.ok()) {
      (void)runner_.release(prefill.cache);
      return Status(sampled.status());
    }
    token = std::move(sampled.value());
  }
  out.push_back(token);

  for (int i = 1; i < params.max_new_tokens && token != params.eos_id; ++i) {
    StatusOr<DecodeResult> decode = runner_.decode(prefill.cache, token);
    if (!decode.ok()) {
      if (is_kv_cache_overflow(decode.status()))
        return finish_generate(runner_, prefill.cache, std::move(out));
      (void)runner_.release(prefill.cache);
      return Status(decode.status());
    }
    StatusOr<int32_t> sampled = sample_(decode.value().logits);
    if (!sampled.ok()) {
      (void)runner_.release(prefill.cache);
      return Status(sampled.status());
    }
    token = std::move(sampled.value());
    out.push_back(token);
  }

  return finish_generate(runner_, prefill.cache, std::move(out));
}
