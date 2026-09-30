#include "runtime/cluster_scheduler.h"

#include "comm/memcpy_peer_comm_agent.h"
#include "core/device.h"
#include "core/status.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/runner/runner.h"
#include "runtime/worker.h"

#include <memory>
#include <utility>

namespace {

constexpr int kDefaultMaxSeqLength = 2048;

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

} // namespace

ClusterScheduler::ClusterScheduler(std::unique_ptr<PlacementPolicy> policy)
    : policy_(std::move(policy)) {
  if (!policy_)
    policy_ = std::make_unique<SimplePlacementPolicy>();
}

ClusterScheduler::~ClusterScheduler() { shutdown(); }

Status ClusterScheduler::load_model(const std::string model_dir, Device kind, int n_workers,
                                    int num_slots) {
#if !MAMBASERVE_WITH_CUDA
  if (kind == Device::GPU)
    return Status::InvalidArgument("Cannot create gpu workers in a non gpu host");
#endif
  if (n_workers <= 0)
    return Status::InvalidArgument("n_workers must be > 0");
  if (num_slots <= 0)
    return Status::InvalidArgument("num_slots must be > 0");
  if (!workers_.empty())
    return Status::InvalidArgument("ClusterScheduler already has a loaded model");

  ModelEntry entry;
  ASSIGN_OR_RETURN(entry, ModelRegistry::open(model_dir, kDefaultMaxSeqLength));

  for (int device_id = 0; device_id < n_workers; ++device_id) {
    StatusOr<WorkerSlot> slot_or = create_worker_(entry, kind, device_id, num_slots);
    if (!slot_or.ok())
      return slot_or.status();
    workers_.push_back(std::move(slot_or.value()));
    worker_stats_.push_back(WorkerStat{
        .index = workers_.size() - 1,
        .capacity = num_slots,
        .inflight = 0,
    });
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

  auto agent = std::make_unique<MemcpyPeerCommAgent>(device_id, kind);

  WorkerSlot slot;
  slot.device_id = device_id;
  slot.worker = std::make_unique<Worker>(static_cast<size_t>(device_id), std::move(runner),
                                         std::move(alloc), std::move(pool), std::move(agent),
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
  size_t worker_idx = 0;

  {
    std::lock_guard<std::mutex> lk(session_mu_);
    ASSIGN_OR_RETURN(worker_idx, policy_->place(tokens, params, worker_stats_));

    worker_stats_[worker_idx].inflight++;
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
      .worker_idx = sess.worker_idx,
      .done = (sess.phase == SessionPhase::Done || sess.phase == SessionPhase::Failed),
      .s = sess.s,
  };
}

Status ClusterScheduler::migrate(uint64_t req_id, size_t dst_idx) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(req_id);
  if (it == sessions_.end())
    return Status::NotFound("unknown req_id");
  if (dst_idx >= workers_.size())
    return Status::InvalidArgument("dst_idx out of range");

  Session& sess = it->second;
  if (sess.phase != SessionPhase::Decoding && sess.phase != SessionPhase::Prefilling)
    return Status::InvalidArgument("migrate requires an active Prefilling/Decoding session");
  if (sess.phase == SessionPhase::Migrating || sess.migrate_pending)
    return Status::InvalidArgument("migrate already in progress");
  if (dst_idx == sess.worker_idx)
    return Status::InvalidArgument("migrate dst must differ from current worker");

  // check if destination worker has enough free slots
  const int free_slots = worker_stats_[dst_idx].capacity - worker_stats_[dst_idx].inflight;
  if (free_slots <= 0)
    return Status::OOM("migrate destination has no free slots");

  // Reserve dst now; begin_migrate_ runs later at the decode boundary.
  worker_stats_[dst_idx].inflight++;
  sess.migrate_pending = true;
  sess.migrate_dst_idx = dst_idx;
  return Status::Ok();
}

void ClusterScheduler::begin_migrate_(Session& sess, uint64_t req_id, size_t dst_idx) {
  const size_t src_idx = sess.worker_idx;
  sess.phase = SessionPhase::Migrating;
  sess.migrate_dst_idx = dst_idx;
  sess.migrate_acks = 0;
  // dst inflight already reserved in migrate()

  const int src_dev = workers_[src_idx].device_id;
  const int dst_dev = workers_[dst_idx].device_id;

  workers_[dst_idx].worker->enqueue(MigrateCmd{
      .req_id = req_id,
      .role = XferRole::Recv,
      .peer_device_id = src_dev,
  });
  workers_[src_idx].worker->enqueue(MigrateCmd{
      .req_id = req_id,
      .role = XferRole::Send,
      .peer_device_id = dst_dev,
  });
}

void ClusterScheduler::fail_migrate_(Session& sess, uint64_t req_id, const Status& err) {
  const size_t src_idx = sess.worker_idx;
  const size_t dst_idx = sess.migrate_dst_idx;
  sess.phase = SessionPhase::Failed;
  sess.s = err;
  workers_[src_idx].worker->enqueue(ReleaseCmd{req_id});
  workers_[dst_idx].worker->enqueue(ReleaseCmd{req_id});
}

void ClusterScheduler::cancel_pending_migrate_(Session& sess) {
  if (!sess.migrate_pending)
    return;
  worker_stats_[sess.migrate_dst_idx].inflight--;
  sess.migrate_pending = false;
}

void ClusterScheduler::continue_after_token_(Session& sess, uint64_t req_id, int32_t token) {
  Worker& worker = *workers_[sess.worker_idx].worker;
  // Check if we have either hit max tokens or eos token
  const bool hit_eos = token == sess.params.eos_id;
  const bool hit_max =
      static_cast<int>(sess.generated_tokens.size()) >= sess.params.max_new_tokens;
  // if yes set phase as done, tell worker to cleanup resources
  if (hit_eos || hit_max) {
    sess.phase = SessionPhase::Done;
    // migration pending and decode phase is done, then whats the use of migrating
    // simply send a release command
    cancel_pending_migrate_(sess);
    worker.enqueue(ReleaseCmd{req_id});
    return;
  }
  // if migration pending and decode phase is not done yet, then send migration requests to
  // both the workers using begin_migrate_
  if (sess.migrate_pending) {
    sess.migrate_pending = false;
    begin_migrate_(sess, req_id, sess.migrate_dst_idx);
    return;
  }
  // if not migration pending and decode phase is not done yet let it decode more
  sess.phase = SessionPhase::Decoding;
  worker.enqueue(DecodeCmd{req_id, token});
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
                 [&](ReleaseEvent& e) { on_release_event_(e); },
                 [&](MigrateEvent& e) { on_migrate_event_(e); },
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

  sess.generated_tokens.push_back(ev.token);
  continue_after_token_(sess, ev.req_id, ev.token);
}

void ClusterScheduler::on_decode_event_(const DecodeEvent& ev) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(ev.req_id);
  if (it == sessions_.end())
    return; // unknown/late event
  Session& sess = it->second;

  // Late decode after migrate started: drop it; migrate owns the session.
  if (sess.phase == SessionPhase::Migrating)
    return;

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
  continue_after_token_(sess, ev.req_id, ev.token);
}

void ClusterScheduler::on_migrate_event_(const MigrateEvent& ev) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(ev.req_id);
  if (it == sessions_.end())
    return;
  Session& sess = it->second;
  if (sess.phase != SessionPhase::Migrating)
    return;

  if (!ev.s.ok()) {
    fail_migrate_(sess, ev.req_id, ev.s);
    return;
  }

  sess.migrate_acks++;
  // acks done only by one side
  if (sess.migrate_acks < 2)
    return;

  // Both sides done — release src slot; commit home on ReleaseEvent.
  workers_[sess.worker_idx].worker->enqueue(ReleaseCmd{ev.req_id});
}

void ClusterScheduler::on_release_event_(const ReleaseEvent& ev) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(ev.req_id);
  if (it == sessions_.end())
    return; // unknown/late event
  Session& sess = it->second;

  if (sess.phase == SessionPhase::Migrating) {
    // Src cleanup after successful xfer transfer — commit sticky home to dst.
    if (ev.worker_idx != sess.worker_idx)
      return; // ignore unexpected (e.g. spurious) release
    worker_stats_[sess.worker_idx].inflight--;
    sess.worker_idx = sess.migrate_dst_idx;
    // dst inflight already reserved in begin_migrate_
    sess.phase = SessionPhase::Decoding;
    sess.migrate_acks = 0;
    if (sess.generated_tokens.empty()) {
      sess.phase = SessionPhase::Failed;
      sess.s = Status::RuntimeError("migrate commit with empty token history");
      workers_[sess.worker_idx].worker->enqueue(ReleaseCmd{ev.req_id});
      return;
    }
    const int32_t last = sess.generated_tokens.back();
    workers_[sess.worker_idx].worker->enqueue(DecodeCmd{ev.req_id, last});
    return;
  }

  if (sess.phase == SessionPhase::Failed) {
    // Fail path releases both src and dst; debit whichever worker reported.
    if (ev.worker_idx < worker_stats_.size())
      worker_stats_[ev.worker_idx].inflight--;
    return;
  }

  // Normal Done/Failed terminal release.
  if (ev.worker_idx < worker_stats_.size())
    worker_stats_[ev.worker_idx].inflight--;
  else
    worker_stats_[sess.worker_idx].inflight--;
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
  worker_stats_.clear();
}
