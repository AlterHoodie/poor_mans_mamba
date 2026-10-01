#include "runtime/cluster_scheduler.h"

#include "comm/comm_factory.h"
#include "core/device.h"
#include "core/status.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/policy/rebalance_policy.h"
#include "runtime/runner/runner.h"
#include "runtime/worker.h"
#include "telemetry/log.h"
#include "telemetry/recorder.h"

#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace {

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

} // namespace

ClusterScheduler::ClusterScheduler(std::unique_ptr<PlacementPolicy> policy,
                                   std::unique_ptr<RebalancePolicy> rebalance)
    : policy_(std::move(policy)), rebalance_(std::move(rebalance)) {
  if (!policy_)
    policy_ = std::make_unique<SimplePlacementPolicy>();
  if (!rebalance_)
    rebalance_ = std::make_unique<SimpleRebalancePolicy>();
}

ClusterScheduler::~ClusterScheduler() { shutdown(); }

Status ClusterScheduler::load_model(const ClusterConfig& cfg) {
  if (!workers_.empty())
    return Status::InvalidArgument("ClusterScheduler already has a loaded model");

  std::shared_ptr<TransportContext> ctx;
  ASSIGN_OR_RETURN(ctx, create_transport_context(cfg));

  ModelEntry entry;
  if (cfg.max_seq_length <= 0)
    return Status::InvalidArgument("max_seq_length must be > 0");
  ASSIGN_OR_RETURN(entry, ModelRegistry::open(cfg.model_dir, cfg.max_seq_length));

  cfg_ = cfg;
  transport_ctx_ = std::move(ctx);

  const int n_workers = cfg_.n_workers;
  std::vector<WorkerSlot> slots(static_cast<size_t>(n_workers));
  std::vector<Status> statuses(static_cast<size_t>(n_workers), Status::Ok());
  auto build = [&](int device_id) {
    StatusOr<WorkerSlot> slot_or = create_worker_(entry, device_id);
    if (!slot_or.ok())
      statuses[static_cast<size_t>(device_id)] = slot_or.status();
    else
      slots[static_cast<size_t>(device_id)] = std::move(slot_or.value());
  };

  if (cfg_.transport == TransportBackend::Nccl && n_workers > 1) {
    // ncclCommInitRank blocks until every rank has joined, so ranks must be
    // created concurrently; a sequential loop would deadlock on rank 0.
    std::vector<std::thread> builders;
    builders.reserve(static_cast<size_t>(n_workers));
    for (int device_id = 0; device_id < n_workers; ++device_id)
      builders.emplace_back(build, device_id);
    for (std::thread& t : builders)
      t.join();
  } else {
    for (int device_id = 0; device_id < n_workers; ++device_id) {
      build(device_id);
      if (!statuses[static_cast<size_t>(device_id)].ok())
        break;
    }
  }

  for (const Status& st : statuses) {
    if (!st.ok()) {
      slots.clear(); // joins any already-started worker threads
      workers_.clear();
      worker_stats_.clear();
      transport_ctx_.reset();
      return st;
    }
  }
  for (WorkerSlot& slot : slots) {
    workers_.push_back(std::move(slot));
    worker_stats_.push_back(WorkerStat{
        .index = workers_.size() - 1,
        .capacity = cfg_.num_slots,
        .inflight = 0,
    });
  }
  LOG_INFO("cluster loaded: model=%s device=%s workers=%d slots=%d transport=%d max_seq=%d",
           cfg_.model_dir.c_str(), cfg_.device == Device::GPU ? "GPU" : "CPU", n_workers,
           cfg_.num_slots, static_cast<int>(cfg_.transport), cfg_.max_seq_length);

  if (Status s = finalize_transport_peers(transport_ctx_); !s.ok()) {
    workers_.clear();
    worker_stats_.clear();
    transport_ctx_.reset();
    return s;
  }

  event_thread_ = std::thread(&ClusterScheduler::event_loop_, this);
  return Status::Ok();
}

StatusOr<WorkerSlot> ClusterScheduler::create_worker_(const ModelEntry& entry, int device_id) {
  if (device_id < 0)
    return Status::InvalidArgument("device id cannot be less than 0");
  if (!entry.cfg)
    return Status::RuntimeError("ModelEntry has no config");

  std::unique_ptr<DeviceAllocator> alloc;
  ASSIGN_OR_RETURN(alloc, create_device_allocator(cfg_.device, device_id));

  std::unique_ptr<Runner> runner;
  ASSIGN_OR_RETURN(runner, entry.create_runner(*alloc));

  std::unique_ptr<CachePool> pool;
  ASSIGN_OR_RETURN(pool,
                   create_cache_pool(*entry.cfg, alloc.get(), cfg_.num_slots, create_cache_layout));

  std::unique_ptr<CommAgent> agent;
  ASSIGN_OR_RETURN(agent, create_comm_agent(cfg_, device_id, transport_ctx_));

  // No-op for MemcpyPeer/NCCL; required for NIXL.
  if (Status s = agent->register_slab(pool->slab_ptr(), pool->slab_bytes()); !s.ok())
    return s;

  auto emit = [this](Event ev) {
    {
      std::lock_guard<std::mutex> lk(event_mu_);
      eventq_.push_back(std::move(ev));
    }
    event_cv_.notify_one();
  };

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
    StatusOr<size_t> placed = policy_->place(tokens, params, worker_stats_);
    if (!placed.ok()) {
      if (placed.status().code() == Code::kOOM)
        telemetry::counters().slot_rejects++;
      return Status(placed.status());
    }
    worker_idx = placed.value();

    worker_stats_[worker_idx].inflight++;
    Session sess;
    sess.params = params;
    sess.worker_idx = worker_idx;
    sessions_.emplace(req_id, std::move(sess));
  }

  telemetry::counters().submits++;
  telemetry::trace(telemetry::TraceKind::Submit, req_id, static_cast<int>(worker_idx),
                   static_cast<int64_t>(tokens.size()), params.max_new_tokens);
  LOG_DEBUG("submit req_id=%llu worker=%zu prompt_tokens=%zu max_new=%d",
            static_cast<unsigned long long>(req_id), worker_idx, tokens.size(),
            params.max_new_tokens);
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

Status ClusterScheduler::migrate_locked_(uint64_t req_id, size_t dst_idx) {
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
  telemetry::trace(telemetry::TraceKind::MigrateRequested, req_id,
                   static_cast<int>(sess.worker_idx), static_cast<int64_t>(dst_idx));
  LOG_DEBUG("migrate requested req_id=%llu src=%zu dst=%zu",
            static_cast<unsigned long long>(req_id), sess.worker_idx, dst_idx);
  return Status::Ok();
}

Status ClusterScheduler::migrate(uint64_t req_id, size_t dst_idx) {
  std::lock_guard<std::mutex> lk(session_mu_);
  return migrate_locked_(req_id, dst_idx);
}

void ClusterScheduler::maybe_rebalance_() {
  std::vector<SessionView> views;
  views.reserve(sessions_.size());
  for (const auto& [req_id, sess] : sessions_) {
    views.push_back(SessionView{
        .req_id = req_id,
        .worker_idx = sess.worker_idx,
        .phase = sess.phase,
        .migrate_pending = sess.migrate_pending,
        .gen_len = sess.generated_tokens.size(),
    });
  }

  telemetry::counters().rebalance_checks++;
  std::optional<RebalanceDecision> decision = rebalance_->check(worker_stats_, views);
  if (!decision.has_value())
    return;

  telemetry::counters().rebalance_decisions++;
  // Control-path logging; debug level only so timed runs stay quiet.
  const auto vit = sessions_.find(decision->victim_req_id);
  const size_t src = (vit != sessions_.end()) ? vit->second.worker_idx : 0;
  const size_t gen_len = (vit != sessions_.end()) ? vit->second.generated_tokens.size() : 0;
  telemetry::trace(telemetry::TraceKind::RebalanceDecision, decision->victim_req_id,
                   static_cast<int>(src), static_cast<int64_t>(decision->dst_idx),
                   static_cast<int64_t>(gen_len));
  LOG_DEBUG("rebalance victim=%llu src=%zu dst=%zu gen_len=%zu src_free=%d dst_free=%d",
            static_cast<unsigned long long>(decision->victim_req_id), src, decision->dst_idx,
            gen_len, worker_stats_[src].capacity - worker_stats_[src].inflight,
            worker_stats_[decision->dst_idx].capacity - worker_stats_[decision->dst_idx].inflight);
  (void)migrate_locked_(decision->victim_req_id, decision->dst_idx);
}

void ClusterScheduler::begin_migrate_(Session& sess, uint64_t req_id, size_t dst_idx) {
  const size_t src_idx = sess.worker_idx;
  sess.phase = SessionPhase::Migrating;
  sess.migrate_dst_idx = dst_idx;
  sess.migrate_acks = 0;
  // dst inflight already reserved in migrate()
  telemetry::counters().migrates_started++;
  telemetry::trace(telemetry::TraceKind::MigrateBegin, req_id, static_cast<int>(src_idx),
                   static_cast<int64_t>(dst_idx));
  LOG_DEBUG("migrate begin req_id=%llu src=%zu dst=%zu", static_cast<unsigned long long>(req_id),
            src_idx, dst_idx);

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
  telemetry::counters().migrates_failed++;
  telemetry::trace(telemetry::TraceKind::Failed, req_id, static_cast<int>(src_idx));
  LOG_ERROR("migrate failed req_id=%llu src=%zu dst=%zu: %s",
            static_cast<unsigned long long>(req_id), src_idx, dst_idx, err.message().c_str());
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
    telemetry::trace(telemetry::TraceKind::Done, req_id, static_cast<int>(sess.worker_idx),
                     static_cast<int64_t>(sess.generated_tokens.size()));
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
  maybe_rebalance_();
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
    telemetry::trace(telemetry::TraceKind::Failed, ev.req_id, static_cast<int>(sess.worker_idx));
    LOG_WARN("prefill failed req_id=%llu worker=%zu: %s", static_cast<unsigned long long>(ev.req_id),
             sess.worker_idx, ev.s.message().c_str());
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
      telemetry::counters().kv_overflows++;
      telemetry::trace(telemetry::TraceKind::Done, ev.req_id, static_cast<int>(sess.worker_idx),
                       static_cast<int64_t>(sess.generated_tokens.size()));
    } else {
      sess.phase = SessionPhase::Failed;
      sess.s = ev.s;
      telemetry::trace(telemetry::TraceKind::Failed, ev.req_id,
                       static_cast<int>(sess.worker_idx));
      LOG_WARN("decode failed req_id=%llu worker=%zu: %s",
               static_cast<unsigned long long>(ev.req_id), sess.worker_idx,
               ev.s.message().c_str());
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
    telemetry::counters().migrates_completed++;
    telemetry::trace(telemetry::TraceKind::MigrateCommit, ev.req_id,
                     static_cast<int>(sess.worker_idx));
    LOG_DEBUG("migrate commit req_id=%llu home=%zu", static_cast<unsigned long long>(ev.req_id),
              sess.worker_idx);
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
