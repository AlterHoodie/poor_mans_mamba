#include "runtime/worker.h"

#include "comm/comm_agent.h"
#include "core/status.h"
#include "ops/cpu/reductions.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "telemetry/log.h"
#include "telemetry/nvtx.h"
#include "telemetry/recorder.h"
#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
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

std::optional<MigrateEvent> Worker::post_xfer_(const MigrateCmd& cmd, int64_t seq_len) {
  auto hit = cache_handles_.find(cmd.req_id);
  if (hit == cache_handles_.end())
    return MigrateEvent{.req_id = cmd.req_id,
                        .role = cmd.role,
                        .s = Status::NotFound("migrate missing cache handle for req_id " +
                                              std::to_string(cmd.req_id))};

  StatusOr<void*> slot_or = pool_->slot_ptr(hit->second);
  if (!slot_or.ok())
    return MigrateEvent{.req_id = cmd.req_id, .role = cmd.role, .s = slot_or.status()};

  XferDesc desc{
      .local_ptr = slot_or.value(),
      .remote_ptr = nullptr, // resolved via MemcpyPeer rendezvous
      .bytes = pool_->slot_bytes(),
      .peer_device_id = cmd.peer_device_id,
      .role = cmd.role,
      .xfer_id = cmd.req_id,
      .seq_len = seq_len,
  };

  StatusOr<XferHandle> handle_or = comm_agent_->post(desc);
  if (!handle_or.ok())
    return MigrateEvent{.req_id = cmd.req_id, .role = cmd.role, .s = handle_or.status()};

  pending_transfers_[cmd.req_id] =
      PendingXfer{.handle = handle_or.value(), .role = cmd.role};
  telemetry::trace(telemetry::TraceKind::MigrateXferPosted, cmd.req_id, static_cast<int>(index_),
                   cmd.role == XferRole::Recv ? 1 : 0, static_cast<int64_t>(desc.bytes));
  LOG_DEBUG("migrate xfer posted req_id=%llu worker=%zu role=%s peer=%d bytes=%zu",
            static_cast<unsigned long long>(cmd.req_id), index_,
            cmd.role == XferRole::Recv ? "recv" : "send", cmd.peer_device_id, desc.bytes);
  return std::nullopt;
}

std::optional<MigrateEvent> Worker::migrate_recv_(const MigrateCmd& cmd) {
  // Recv: register a fresh slot for the incoming cache, then post.
  if (Status s = register_(cmd.req_id); !s.ok())
    return MigrateEvent{.req_id = cmd.req_id, .role = cmd.role, .s = s};
  return post_xfer_(cmd, /*seq_len=*/0);
}

std::optional<MigrateEvent> Worker::migrate_send_(const MigrateCmd& cmd) {
  // Send: existing handle required; publish seq_len with the transfer.
  auto hit = cache_handles_.find(cmd.req_id);
  if (hit == cache_handles_.end())
    return MigrateEvent{
        .req_id = cmd.req_id,
        .role = cmd.role,
        .s = Status::NotFound("Send migrate missing cache handle for req_id " +
                              std::to_string(cmd.req_id))};

  StatusOr<int64_t> sl = pool_->seq_len(hit->second);
  if (!sl.ok())
    return MigrateEvent{.req_id = cmd.req_id, .role = cmd.role, .s = sl.status()};
  return post_xfer_(cmd, sl.value());
}

std::optional<MigrateEvent> Worker::migrate_(const MigrateCmd& cmd) {
  if (!pool_)
    return MigrateEvent{.req_id = cmd.req_id,
                        .role = cmd.role,
                        .s = Status::RuntimeError("cache pool not yet initialized")};
  if (!alloc_)
    return MigrateEvent{.req_id = cmd.req_id,
                        .role = cmd.role,
                        .s = Status::RuntimeError("allocator not yet initialized")};
  if (!comm_agent_)
    return MigrateEvent{.req_id = cmd.req_id,
                        .role = cmd.role,
                        .s = Status::RuntimeError("comm agent not yet initialized")};

  if (cmd.role == XferRole::Recv)
    return migrate_recv_(cmd);
  return migrate_send_(cmd);
}

Worker::Worker(size_t index, std::unique_ptr<Runner> runner, std::unique_ptr<DeviceAllocator> alloc,
               std::unique_ptr<CachePool> pool, std::unique_ptr<CommAgent> comm_agent,
               std::function<void(Event)> emit)
    : index_(index), alloc_(std::move(alloc)), pool_(std::move(pool)), runner_(std::move(runner)),
      comm_agent_(std::move(comm_agent)), emit_(std::move(emit)) {
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

void Worker::poll_transfer_states_(){
  using telemetry::TraceKind;
  auto xfer_done = [&](uint64_t req_id, XferRole role, bool ok) {
    telemetry::trace(TraceKind::MigrateXferDone, req_id, static_cast<int>(index_),
                     role == XferRole::Recv ? 1 : 0, ok ? 1 : 0);
    if (ok && role == XferRole::Send && pool_)
      telemetry::counters().bytes_migrated += pool_->slot_bytes();
    LOG_DEBUG("migrate xfer done req_id=%llu worker=%zu role=%s ok=%d",
              static_cast<unsigned long long>(req_id), index_,
              role == XferRole::Recv ? "recv" : "send", ok ? 1 : 0);
  };
  for (auto it = pending_transfers_.begin(); it != pending_transfers_.end(); ) {
    XferState state = comm_agent_->poll(it->second.handle);
    switch (state) {
      case XferState::Pending:
        ++it;
        break;
      case XferState::Done: {
        if (it->second.role == XferRole::Recv) {
          auto hit = cache_handles_.find(it->first);
          if (hit != cache_handles_.end()) {
            const int64_t sl = comm_agent_->xfer_seq_len(it->second.handle);
            if (Status s = pool_->set_seq_len(hit->second, sl); !s.ok()) {
              xfer_done(it->first, it->second.role, false);
              emit_(MigrateEvent{.req_id = it->first, .role = it->second.role, .s = s});
              it = pending_transfers_.erase(it);
              break;
            }
          }
        }
        xfer_done(it->first, it->second.role, true);
        emit_(MigrateEvent{
            .req_id = it->first, .role = it->second.role, .s = Status::Ok()});
        it = pending_transfers_.erase(it);
        break;
      }
      case XferState::Error:
        xfer_done(it->first, it->second.role, false);
        LOG_ERROR("transfer failed req_id=%llu worker=%zu",
                  static_cast<unsigned long long>(it->first), index_);
        emit_(MigrateEvent{
            .req_id = it->first,
            .role = it->second.role,
            .s = Status::RuntimeError("transfer failed for req_id: " +
                                      std::to_string(it->first))});
        it = pending_transfers_.erase(it);
        break;
    }
  }
}

void Worker::loop_(){
#ifdef MAMBASERVE_WITH_CUDA
  if (alloc_ && alloc_->kind() == Device::GPU) {
    cudaError_t err = cudaSetDevice(alloc_->device_id());
    if (err != cudaSuccess) {
      // Device bind failed; subsequent CUDA work will surface errors.
    }
  }
#endif
  for(;;){
    poll_transfer_states_();

    Command cmd;
    {
      std::unique_lock<std::mutex> lk(cmd_mu_);

      // poll only when they are active transfers
      if(!pending_transfers_.empty()){
        // wake on: new cmd, stop or timeout so we can poll again
        cmd_cv_.wait_for(lk, std::chrono::milliseconds(1), [&] {return !cmdq_.empty() || stop_;});
      } else { // else simply wait on the queue
      cmd_cv_.wait(lk, [&](){return !cmdq_.empty() || stop_;});
      }

      if (cmdq_.empty() && stop_ && pending_transfers_.empty())
        return;
      if(cmdq_.empty()) // wait_for timed out with only pending work - loop back to poll
        continue;
      cmd = std::move(cmdq_.front());
      cmdq_.pop_front();
    }

    // Migrate posts asynchronously; only Prefill/Decode/Release/fail emit here.
    // Event is not assignable (const token fields), so emplace into optional.
    std::optional<Event> ev;
    std::visit(overloaded{
                   [&](PrefillCmd& c) {
                     MS_NVTX_RANGE("prefill");
                     const int w = static_cast<int>(index_);
                     telemetry::trace(telemetry::TraceKind::PrefillStart, c.req_id, w,
                                      static_cast<int64_t>(c.tokens.size()));
                     if (auto s = register_(c.req_id); !s.ok()) {
                       if (s.code() == Code::kOOM)
                         telemetry::counters().slot_rejects++;
                       LOG_WARN("prefill register failed req_id=%llu worker=%d: %s",
                                static_cast<unsigned long long>(c.req_id), w, s.message().c_str());
                       telemetry::trace(telemetry::TraceKind::PrefillEnd, c.req_id, w, 0);
                       ev.emplace(PrefillEvent{.req_id = c.req_id, .s = s});
                       return;
                     }
                     ev.emplace(prefill_(c.req_id, c.tokens));
                     telemetry::trace(telemetry::TraceKind::PrefillEnd, c.req_id, w,
                                      std::get<PrefillEvent>(*ev).s.ok() ? 1 : 0);
                   },
                   [&](DecodeCmd& c) {
                     MS_NVTX_RANGE("decode");
                     const int w = static_cast<int>(index_);
                     telemetry::trace(telemetry::TraceKind::DecodeStart, c.req_id, w);
                     ev.emplace(decode_(c.req_id, c.token));
                     telemetry::trace(telemetry::TraceKind::DecodeEnd, c.req_id, w, 0,
                                      std::get<DecodeEvent>(*ev).s.ok() ? 1 : 0);
                   },
                   [&](ReleaseCmd& c) {
                     ev.emplace(ReleaseEvent{.req_id = c.req_id,
                                             .worker_idx = index_,
                                             .s = release_(c.req_id)});
                   },
                   [&](MigrateCmd& c) {
                     MS_NVTX_RANGE("migrate_post");
                     if (auto m = migrate_(c)) {
                       LOG_ERROR("migrate post failed req_id=%llu worker=%zu: %s",
                                 static_cast<unsigned long long>(c.req_id), index_,
                                 m->s.message().c_str());
                       ev.emplace(*m);
                     }
                   },
               },
               cmd);
    if (ev)
      emit_(std::move(*ev));
  }
}