#include "runtime/worker.h"

#include "core/status.h"
#include "ops/cpu/reductions.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include <sys/types.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

#ifdef MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

bool is_kv_cache_overflow(const Status& s) {
  return s.code() == Code::kKvCacheOverflow;
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

template<class... Ts>
struct overloaded : Ts...{
  using Ts::operator()...;
};
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

} // namespace

StatusOr<int32_t> Worker::sample_(Tensor& logits) {
  if (logits.buffer.device == Device::CPU)
    return argmax(logits);
  if (logits.buffer.device == Device::GPU)
    return argmax_gpu_d2h(logits);
  return Status::InvalidArgument("logits device not supported for sampling");
}

PrefillEvent Worker::prefill_(uint64_t req_id, std::span<const int32_t> tokens) {
  if(!pool_)
    return {.req_id = req_id, .s = Status::RuntimeError("cache pool not yet initialized")};
  if(!alloc_)
    return {.req_id = req_id, .s = Status::RuntimeError("allocator not yet initialized")};
  if (tokens.empty())
    return {.req_id = req_id, .s = Status::InvalidArgument("token_ids cannot be empty")};

  auto it = cache_handles_.find(req_id);
  if(it == cache_handles_.end())
    return {.req_id = req_id, .s = Status::NotFound("Could not find Cache Handle for req_id" + std::to_string(req_id))};
  
  StatusOr<std::span<LayerCacheView>> views = pool_->layer_views(it->second);
  if(!views.ok())
    return {.req_id = req_id, .s = views.status()};
  
  StatusOr<Tensor> prefill = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc_.get());
    return runner_->prefill(tokens, views.value());
  }();
  if(!prefill.ok())
    return {.req_id = req_id, .s = prefill.status()};
  
  int32_t token = 0;
  {
    StatusOr<int32_t> sampled = sample_(prefill.value());
    if (!sampled.ok()) {
      return {.req_id = req_id, .s = sampled.status()};
    }
    token = std::move(sampled.value());
  }

  return {
    .req_id = req_id,
    .token = token,
    .s = Status::Ok()
  };
}


DecodeEvent Worker::decode_(uint64_t req_id, const int32_t token){
  if(!pool_)
    return {.req_id = req_id, .s = Status::RuntimeError("cache pool not yet initialized")};
  if(!alloc_)
    return {.req_id = req_id, .s = Status::RuntimeError("allocator not yet initialized")};

  auto it = cache_handles_.find(req_id);
  if(it == cache_handles_.end())
    return {.req_id = req_id, .s = Status::NotFound("Could not find Cache Handle for req_id" + std::to_string(req_id))};
  
  StatusOr<std::span<LayerCacheView>> views = pool_->layer_views(it->second);
  if(!views.ok())
    return {.req_id = req_id, .s = views.status()};

  StatusOr<Tensor> logits = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc_.get());
    return runner_->decode(token, views.value());
  }();
  if(!logits.ok())
    return {.req_id = req_id, .s = logits.status()};
  
  StatusOr<int32_t> sampled = sample_(logits.value());
  if (!sampled.ok())
    return {.req_id = req_id, .s = sampled.status()};

  return {
    .req_id = req_id,
    .token = std::move(sampled.value()),
    .s = Status::Ok()
  };
}

Status Worker::register_(uint64_t req_id){
  if(!pool_) return Status::RuntimeError("Cache Pool not yet initialized");
  if(!alloc_) return Status::RuntimeError("Allocator not yet initialized");
  if(cache_handles_.find(req_id) != cache_handles_.end()) return Status::RuntimeError("Cache Handle already registered for request_id" + std::to_string(req_id));
  
  StatusOr<CacheHandle> handle= pool_->acquire();
  if(!handle.ok()) return handle.status();
  cache_handles_[req_id] = handle.value();

  return Status::Ok();
}

Status Worker::release_(uint64_t req_id){
  if(!pool_) return Status::RuntimeError("Cache Pool not yet initialized");
  if(!alloc_) return Status::RuntimeError("Allocator not yet initialized");

  auto it = cache_handles_.find(req_id);
  if(it == cache_handles_.end()) return Status::NotFound("Could not find Cache Handle for req_id" + std::to_string(req_id));

  Status s = pool_->release(it->second);
  cache_handles_.erase(it);

  return s;
}

Worker::Worker(std::unique_ptr<Runner> runner, std::unique_ptr<DeviceAllocator> alloc,
               std::unique_ptr<CachePool> pool, std::function<void(Event)> emit)
    : alloc_(std::move(alloc)), pool_(std::move(pool)), runner_(std::move(runner)),
      emit_(std::move(emit)) {
  thread_ = std::thread(&Worker::loop_, this);
}

Worker::~Worker() {
  {
    std::lock_guard<std::mutex> lk(cmd_mu_);
    stop_ = true;
  }
  cmd_cv_.notify_all();
  if (thread_.joinable())
    thread_.join();
}

void Worker::enqueue(Command cmd){
  {
    std::lock_guard<std::mutex> lk(cmd_mu_);
    cmdq_.push_back(std::move(cmd));
  }
  cmd_cv_.notify_one();
}

void Worker::loop_(){
  for(;;){
    Command cmd;
    {
      std::unique_lock<std::mutex> lk(cmd_mu_);
      cmd_cv_.wait(lk, [&](){return !cmdq_.empty() || stop_;});
      if (cmdq_.empty() && stop_)
        return;
      cmd = std::move(cmdq_.front());
      cmdq_.pop_front();
    }

    // execute the command
    Event ev = std::visit(overloaded{
      [&](PrefillCmd& c) -> Event {
        if(auto s = register_(c.req_id); !s.ok()){
          return PrefillEvent{.req_id = c.req_id, .s = s};
        }
        return prefill_(c.req_id, c.tokens);
      },
      [&](DecodeCmd& c) -> Event {
        return decode_(c.req_id, c.token);
      },
      [&](ReleaseCmd& c) -> Event {
        return ReleaseEvent{.req_id = c.req_id, .s = release_(c.req_id)};
      },
      [&](MigrateCmd& c) -> Event {
        return MigrateEvent{.req_id = c.req_id,
                            .s = Status::NotImplemented("migrate")};
      },
    }, cmd);
    // push the result into the cluster queue
    emit_(std::move(ev));
  }
}