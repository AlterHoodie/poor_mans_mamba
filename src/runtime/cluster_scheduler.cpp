#include "runtime/cluster_scheduler.h"

#include "core/device.h"
#include "core/status.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/runner/runner.h"

#include <memory>
#include <utility>

namespace {

// TODO: thread through load_model once callers need non-default values.
constexpr int kDefaultMaxSeqLength = 2048;
constexpr int kDefaultNumSlots = 8;

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

} // namespace

ClusterScheduler::~ClusterScheduler() { shutdown(); }

Status ClusterScheduler::load_model(const std::string model_dir, Device kind, int n_workers) {
#if !MAMBASERVE_WITH_CUDA
  if (kind == Device::GPU)
    return Status::InvalidArgument("Cannot create gpu workers in a non gpu host");
#endif
  if (n_workers <= 0)
    return Status::InvalidArgument("n_workers must be > 0");
  if (!workers_.empty())
    return Status::InvalidArgument("ClusterScheduler already has a loaded model");

  ModelEntry entry;
  ASSIGN_OR_RETURN(entry, ModelRegistry::open(model_dir, kDefaultMaxSeqLength));

  for (int device_id = 0; device_id < n_workers; ++device_id) {
    StatusOr<WorkerSlot> slot_or = create_worker_(entry, kind, device_id, kDefaultNumSlots);
    if (!slot_or.ok())
      return slot_or.status();
    workers_.push_back(std::move(slot_or.value()));
  }

  event_thread_ = std::thread(&ClusterScheduler::event_loop_, this);
  return Status::Ok();
}

StatusOr<WorkerSlot> ClusterScheduler::create_worker_(const ModelEntry& entry, Device kind,
                                                      int device_id, int num_slots) {
  if (device_id < 0)
    return Status::InvalidArgument("device id cannot be less than 0");
  if (!entry.cfg)
    return Status::RuntimeError("ModelEntry has no config");

  std::unique_ptr<DeviceAllocator> alloc;
  ASSIGN_OR_RETURN(alloc, create_device_allocator(kind, device_id));

  std::unique_ptr<Runner> runner;
  ASSIGN_OR_RETURN(runner, entry.create_runner(*alloc));

  std::unique_ptr<CachePool> pool;
  ASSIGN_OR_RETURN(pool, create_cache_pool(*entry.cfg, alloc.get(), num_slots, create_cache_layout));

  auto emit = [this](Event ev) {
    {
      std::lock_guard<std::mutex> lk(event_mu_);
      eventq_.push_back(std::move(ev));
    }
    event_cv_.notify_one();
  };

  WorkerSlot slot;
  slot.device_id = device_id;
  slot.worker = std::make_unique<Worker>(std::move(runner), std::move(alloc), std::move(pool),
                                         std::move(emit));
  return slot;
}

StatusOr<uint64_t> ClusterScheduler::submit(std::vector<int32_t> tokens, GenerateParams params) {
  if (tokens.empty())
    return Status::InvalidArgument("tokens cannot be empty");
  if (params.eos_id < 0)
    return Status::InvalidArgument("eos_id required");
  if (params.max_new_tokens <= 0)
    return Status::InvalidArgument("max_new_tokens must be > 0");
  if (workers_.empty())
    return Status::RuntimeError("no workers loaded; call load_model first");

  const uint64_t req_id = req_id_counter_.fetch_add(1, std::memory_order_relaxed);
  const size_t worker_idx =
      next_worker_.fetch_add(1, std::memory_order_relaxed) % workers_.size();

  {
    std::lock_guard<std::mutex> lk(session_mu_);
    Session sess;
    sess.params = params;
    sess.worker_idx = worker_idx;
    sessions_.emplace(req_id, std::move(sess));
  }

  workers_[worker_idx].worker->enqueue(PrefillCmd{req_id, std::move(tokens), params});
  return req_id;
}

Response ClusterScheduler::poll(uint64_t req_id) const {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(req_id);
  if (it == sessions_.end())
    return Response{.req_id = req_id, .s = Status::NotFound("unknown req_id")};

  const Session& sess = it->second;
  return Response{
      .req_id = req_id,
      .tokens = sess.generated_tokens,
      .gen_seq_len = static_cast<int>(sess.generated_tokens.size()),
      .done = (sess.phase == SessionPhase::Done || sess.phase == SessionPhase::Failed),
      .s = sess.s,
  };
}

void ClusterScheduler::event_loop_() {
  for (;;) {
    std::unique_lock<std::mutex> lk(event_mu_);
    event_cv_.wait(lk, [&]() { return !eventq_.empty() || stop_; });
    if (eventq_.empty() && stop_)
      return;
    // PrefillEvent/DecodeEvent hold a const token, so Event (their variant)
    // is move-constructible but not move-assignable; construct, don't assign.
    Event ev = std::move(eventq_.front());
    eventq_.pop_front();
    lk.unlock();
    handle_event_(std::move(ev));
  }
}

void ClusterScheduler::handle_event_(Event ev) {
  std::visit(overloaded{
                 [&](PrefillEvent& e) { on_prefill_event_(e); },
                 [&](DecodeEvent& e) { on_decode_event_(e); },
                 [&](ReleaseEvent&) { /* session already terminal; nothing else to do */ },
                 [&](MigrateEvent&) { /* migrate not implemented yet */ },
             },
             ev);
}

void ClusterScheduler::on_prefill_event_(const PrefillEvent& ev) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(ev.req_id);
  if (it == sessions_.end())
    return; // unknown/late event
  Session& sess = it->second;
  Worker& worker = *workers_[sess.worker_idx].worker;

  if (!ev.s.ok()) {
    sess.phase = SessionPhase::Failed;
    sess.s = ev.s;
    // Tell worker to release resources for that request
    worker.enqueue(ReleaseCmd{ev.req_id});
    return;
  }

  // Check if we have either hit max tokens or eos token
  sess.generated_tokens.push_back(ev.token);
  const bool hit_eos = ev.token == sess.params.eos_id;
  const bool hit_max = static_cast<int>(sess.generated_tokens.size()) >= sess.params.max_new_tokens;
  // if yes set phase as done, tell worker to cleanup resources
  if (hit_eos || hit_max) {
    sess.phase = SessionPhase::Done;
    worker.enqueue(ReleaseCmd{ev.req_id});
  } else { // else continue decoding
    sess.phase = SessionPhase::Decoding;
    worker.enqueue(DecodeCmd{ev.req_id, ev.token});
  }
}

void ClusterScheduler::on_decode_event_(const DecodeEvent& ev) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(ev.req_id);
  if (it == sessions_.end())
    return; // unknown/late event
  Session& sess = it->second;
  Worker& worker = *workers_[sess.worker_idx].worker;

  if (!ev.s.ok()) {
    // KV cache overflow ends the session with a partial result, not a hard failure.
    if (ev.s.code() == Code::kKvCacheOverflow) {
      sess.phase = SessionPhase::Done;
    } else {
      sess.phase = SessionPhase::Failed;
      sess.s = ev.s;
    }
    worker.enqueue(ReleaseCmd{ev.req_id});
    return;
  }

  sess.generated_tokens.push_back(ev.token);
  const bool hit_eos = ev.token == sess.params.eos_id;
  const bool hit_max = static_cast<int>(sess.generated_tokens.size()) >= sess.params.max_new_tokens;
  if (hit_eos || hit_max) {
    sess.phase = SessionPhase::Done;
    worker.enqueue(ReleaseCmd{ev.req_id});
  } else {
    worker.enqueue(DecodeCmd{ev.req_id, ev.token});
  }
}

void ClusterScheduler::shutdown() {
  {
    // stop the event loop to stop listening to extra events from workers
    std::lock_guard<std::mutex> lk(event_mu_);
    stop_ = true;
  }
  event_cv_.notify_all();
  if (event_thread_.joinable())
    event_thread_.join();
  workers_.clear(); // each Worker's dtor stops + joins its own thread
}
