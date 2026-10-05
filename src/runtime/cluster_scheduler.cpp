#include "runtime/cluster_scheduler.h"

#include "comm/comm_factory.h"
#include "core/device.h"
#include "core/status.h"
#include "proto/worker.pb.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/ipc/proto_convert.h"
#include "runtime/policy/rebalance_policy.h"
#include "runtime/runner/runner.h"
#include "runtime/worker.h"
#include "telemetry/log.h"
#include "telemetry/recorder.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace {

mambaserve::Command make_release_cmd(uint64_t req_id) {
  mambaserve::Command cmd;
  cmd.mutable_release()->set_req_id(req_id);
  return cmd;
}

mambaserve::Command make_decode_cmd(uint64_t req_id, int32_t token) {
  mambaserve::Command cmd;
  auto* d = cmd.mutable_decode();
  d->set_req_id(req_id);
  d->set_token(token);
  return cmd;
}

mambaserve::Command make_migrate_cmd(uint64_t req_id, mambaserve::XferRole role,
                                     int peer_device_id) {
  mambaserve::Command cmd;
  auto* m = cmd.mutable_migrate();
  m->set_req_id(req_id);
  m->set_role(role);
  m->set_peer_device_id(peer_device_id);
  return cmd;
}

mambaserve::Command make_prefill_cmd(uint64_t req_id, std::span<const int32_t> tokens,
                                     const GenerateParams& params) {
  mambaserve::Command cmd;
  auto* pref = cmd.mutable_prefill();
  pref->set_req_id(req_id);
  pref->set_max_new_tokens(params.max_new_tokens);
  pref->set_eos_id(params.eos_id);
  for (int32_t t : tokens)
    pref->add_tokens(t);
  return cmd;
}

mambaserve::Command make_shutdown_cmd() {
  mambaserve::Command cmd;
  cmd.mutable_shutdown();
  return cmd;
}

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

Status ClusterScheduler::start(const ClusterConfig& cfg, std::vector<WorkerSlot> slots) {
  if (!workers_.empty())
    return Status::InvalidArgument("ClusterScheduler already has a loaded model");

  if (Status s = validate_cluster_config(cfg); !s.ok())
    return s;

  if (cfg.max_seq_length <= 0)
    return Status::InvalidArgument("max_seq_length must be > 0");

  if (slots.size() != static_cast<size_t>(cfg.n_workers))
    return Status::InvalidArgument("expected " + std::to_string(cfg.n_workers) +
                                   " workers, got " + std::to_string(slots.size()));
  for (size_t i = 0; i < slots.size(); ++i) {
    if (!slots[i].chan || (!slots[i].worker && !slots[i].process))
      return Status::InvalidArgument("worker " + std::to_string(i) +
                                     " needs a channel and a worker or process");
  }

  cfg_ = cfg;

  const int n_workers = cfg_.n_workers;
  auto fail = [&](const Status& st) {
    slots.clear(); // joins every worker thread / reaps every worker process
    clear_workers_();
    return st;
  };

  // Ready barrier: ingress is not running yet, so read each channel directly.
  // A worker always reports Ready (with its init status) before serving commands.
  for (WorkerSlot& slot : slots) {
    for (;;) {
      // block on each worker slot channel till its ready
      StatusOr<mambaserve::Envelope> env_or = slot.chan->recv();
      if (!env_or.ok())
        return fail(Status::RuntimeError("worker " + std::to_string(slot.device_id) +
                                         " exited before reporting ready"));
      const mambaserve::Envelope& env = env_or.value();
      
      // if it sent a non ready message we poll it again
      if (env.body_case() != mambaserve::Envelope::kEvent ||
          env.event().body_case() != mambaserve::Event::kReady) {
        LOG_WARN("worker %d sent a non-ready message before ready; dropped", slot.device_id);
        continue;
      }

      const Status st = from_proto(env.event().ready().status());
      if (!st.ok())
        return fail(st);
      break;
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
  LOG_INFO("cluster loaded: model=%s device=%s workers=%d slots=%d transport=%d mode=%s "
           "max_seq=%d",
           cfg_.model_dir.c_str(), cfg_.device == Device::GPU ? "GPU" : "CPU", n_workers,
           cfg_.num_slots, static_cast<int>(cfg_.transport),
           cfg_.worker_mode == WorkerMode::Process ? "process" : "thread", cfg_.max_seq_length);

  {
    StatusOr<std::unique_ptr<TransportControlPlane>> plane_or =
        create_transport_control_plane(cfg_);
    if (!plane_or.ok())
      return fail(plane_or.status());
    transport_plane_ = std::move(plane_or.value());
  }

  stop_ = false;
  ingress_thread_ = std::thread(&ClusterScheduler::ingress_loop_, this);
  event_thread_ = std::thread(&ClusterScheduler::event_loop_, this);

  // Ingress is live: let the plane bootstrap the workers (e.g. NCCL unique id, set NIXL slab).
  if (Status s = transport_plane_->on_cluster_ready(transport_sender_); !s.ok()) {
    shutdown();
    return s;
  }
  return Status::Ok();
}

void ClusterScheduler::shutdown() {
  {
    // stop the event loop to stop listening to extra events from workers
    std::lock_guard<std::mutex> lk(event_mu_);
    stop_ = true;
  }
  event_cv_.notify_all();

  for (size_t i = 0; i < workers_.size(); ++i) {
    if (!workers_[i].chan)
      continue;
    send_cmd_(i, make_shutdown_cmd());
    workers_[i].chan->close();
  }

  if (ingress_thread_.joinable())
    ingress_thread_.join();
  if (event_thread_.joinable())
    event_thread_.join();
 
  clear_workers_();
  transport_plane_.reset();
}

void ClusterScheduler::clear_workers_() noexcept{
  workers_.clear();
  worker_stats_.clear();
}

StatusOr<uint64_t> ClusterScheduler::submit(std::vector<int32_t> tokens, GenerateParams params) {
  if (tokens.empty())
    return Status::InvalidArgument("tokens cannot be empty");
  if (params.eos_id < 0)
    return Status::InvalidArgument("eos_id required");
  if (params.max_new_tokens <= 0)
    return Status::InvalidArgument("max_new_tokens must be > 0");
  if (workers_.empty())
    return Status::RuntimeError("no workers loaded; call start first");

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

  send_cmd_(worker_idx, make_prefill_cmd(req_id, tokens, params));
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
  return migrate_locked_(req_id, dst_idx);
}

void ClusterScheduler::send_cmd_(size_t worker_idx, mambaserve::Command cmd) {
  if (worker_idx >= workers_.size() || !workers_[worker_idx].chan)
    return;
  mambaserve::Envelope env;
  *env.mutable_cmd() = std::move(cmd);
  if (Status s = workers_[worker_idx].chan->send(env); !s.ok())
    LOG_ERROR("send_cmd to worker %zu failed: %s", worker_idx, s.message().c_str());
}

void ClusterScheduler::send_trsp_(size_t worker_idx, const mambaserve::TransportControl& trsp) {
  if (worker_idx >= workers_.size() || !workers_[worker_idx].chan)
    return;
  mambaserve::Envelope env;
  *env.mutable_transport() = trsp;
  if (Status s = workers_[worker_idx].chan->send(env); !s.ok())
    LOG_ERROR("send_trsp to worker %zu failed: %s", worker_idx, s.message().c_str());
}

void ClusterScheduler::Sender::send_transport(size_t worker_idx,
                                                 const mambaserve::TransportControl& msg) {
  sched.send_trsp_(worker_idx, msg);
}

void ClusterScheduler::Sender::broadcast_transport(const mambaserve::TransportControl& msg) {
  for (size_t i = 0; i < sched.workers_.size(); ++i)
    sched.send_trsp_(i, msg);
}

size_t ClusterScheduler::Sender::n_workers() const { return sched.workers_.size(); }

void ClusterScheduler::ingress_loop_() {
  for (;;) {
    {
      std::lock_guard<std::mutex> lk(event_mu_);
      if (stop_)
        return;
    }

    bool got = false;
    for (size_t i = 0; i < workers_.size(); ++i) {
      WorkerSlot& slot = workers_[i];
      if (!slot.chan)
        continue;
      StatusOr<mambaserve::Envelope> env_or = slot.chan->try_recv();
      if (!env_or.ok())
        continue;
      mambaserve::Envelope env = std::move(env_or.value());
      switch (env.body_case()) {
      case mambaserve::Envelope::kEvent:
        {
          std::lock_guard<std::mutex> lk(event_mu_);
          eventq_.push_back(std::move(*env.mutable_event()));
        }
        event_cv_.notify_one();
        got = true;
        break;
      case mambaserve::Envelope::kTransport:
        // backend protocol lives in the plane; handled inline, not via eventq_
        if (transport_plane_)
          transport_plane_->on_upstream(i, env.transport(), transport_sender_);
        got = true;
        break;
      case mambaserve::Envelope::kCmd:
      case mambaserve::Envelope::BODY_NOT_SET:
        break;
      }
    }

    if (!got)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void ClusterScheduler::event_loop_() {
  for (;;) {
    std::unique_lock<std::mutex> lk(event_mu_);
    event_cv_.wait(lk, [&]() { return !eventq_.empty() || stop_; });
    if (eventq_.empty() && stop_)
      return;
    // PrefillEvent/DecodeEvent hold a const token, so Event (their variant)
    // is move-constructible but not move-assignable; construct, don't assign.
    mambaserve::Event ev = std::move(eventq_.front());
    eventq_.pop_front();
    lk.unlock();
    handle_event_(std::move(ev));
  }
}

void ClusterScheduler::handle_event_(mambaserve::Event ev) {
  switch (ev.body_case()) {
  case mambaserve::Event::kPrefill:
    on_prefill_event_(ev.prefill());
    break;
  case mambaserve::Event::kDecode:
    on_decode_event_(ev.decode());
    break;
  case mambaserve::Event::kRelease:
    on_release_event_(ev.release());
    break;
  case mambaserve::Event::kMigrate:
    on_migrate_event_(ev.migrate());
    break;
  case mambaserve::Event::kReady:
    LOG_DEBUG("worker %llu ready", static_cast<unsigned long long>(ev.ready().worker_idx()));
    break;
  case mambaserve::Event::BODY_NOT_SET:
    break;
  }
}

void ClusterScheduler::continue_after_token_(Session& sess, uint64_t req_id, int32_t token) {
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
    send_cmd_(sess.worker_idx, make_release_cmd(req_id));
    return;
  }
  // if migration pending and decode phase is not done yet, then send migration requests to
  // both the workers using begin_migrate_
  if (sess.migrate_pending) {
    sess.migrate_pending = false;
    begin_migrate_(sess, req_id, sess.migrate_dst_idx);
    return;
  }
  sess.phase = SessionPhase::Decoding;
  send_cmd_(sess.worker_idx, make_decode_cmd(req_id, token));
  maybe_rebalance_();
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

  // route must exist before either worker can emit control traffic for this xfer
  transport_plane_->on_migrate_begin(req_id, src_idx, dst_idx);
  send_cmd_(dst_idx, make_migrate_cmd(req_id, mambaserve::XFER_RECV, src_dev));
  send_cmd_(src_idx, make_migrate_cmd(req_id, mambaserve::XFER_SEND, dst_dev));
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
  transport_plane_->on_migrate_end(req_id);
  send_cmd_(src_idx, make_release_cmd(req_id));
  send_cmd_(dst_idx, make_release_cmd(req_id));
}

void ClusterScheduler::cancel_pending_migrate_(Session& sess) {
  if (!sess.migrate_pending)
    return;
  worker_stats_[sess.migrate_dst_idx].inflight--;
  sess.migrate_pending = false;
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

void ClusterScheduler::on_prefill_event_(const mambaserve::PrefillEvent& ev) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(ev.req_id());
  if (it == sessions_.end())
    return; // unknown/late event
  Session& sess = it->second;
  Status st = from_proto(ev.status());

  if (!st.ok()) {
    sess.phase = SessionPhase::Failed;
    sess.s = st;
    telemetry::trace(telemetry::TraceKind::Failed, ev.req_id(), static_cast<int>(sess.worker_idx));
    LOG_WARN("prefill failed req_id=%llu worker=%zu: %s",
             static_cast<unsigned long long>(ev.req_id()), sess.worker_idx, st.message().c_str());
    send_cmd_(sess.worker_idx, make_release_cmd(ev.req_id()));
    return;
  }

  sess.generated_tokens.push_back(ev.token());
  continue_after_token_(sess, ev.req_id(), ev.token());
}

void ClusterScheduler::on_decode_event_(const mambaserve::DecodeEvent& ev) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(ev.req_id());
  if (it == sessions_.end())
    return; // unknown/late event
  Session& sess = it->second;

  // Late decode after migrate started: drop it; migrate owns the session.
  if (sess.phase == SessionPhase::Migrating)
    return;

  Status st = from_proto(ev.status());

  if (!st.ok()) {
    // KV cache overflow ends the session with a partial result, not a hard failure.
    if (st.code() == Code::kKvCacheOverflow) {
      sess.phase = SessionPhase::Done;
      telemetry::counters().kv_overflows++;
      telemetry::trace(telemetry::TraceKind::Done, ev.req_id(), static_cast<int>(sess.worker_idx),
                       static_cast<int64_t>(sess.generated_tokens.size()));
    } else {
      sess.phase = SessionPhase::Failed;
      sess.s = st;
      telemetry::trace(telemetry::TraceKind::Failed, ev.req_id(),
                       static_cast<int>(sess.worker_idx));
      LOG_WARN("decode failed req_id=%llu worker=%zu: %s",
               static_cast<unsigned long long>(ev.req_id()), sess.worker_idx, st.message().c_str());
    }
    send_cmd_(sess.worker_idx, make_release_cmd(ev.req_id()));
    return;
  }

  sess.generated_tokens.push_back(ev.token());
  continue_after_token_(sess, ev.req_id(), ev.token());
}

void ClusterScheduler::on_migrate_event_(const mambaserve::MigrateEvent& ev) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(ev.req_id());
  if (it == sessions_.end())
    return;
  Session& sess = it->second;
  if (sess.phase != SessionPhase::Migrating)
    return;

  Status st = from_proto(ev.status());
  if (!st.ok()) {
    fail_migrate_(sess, ev.req_id(), st);
    return;
  }

  sess.migrate_acks++;
  // acks done only by one side
  if (sess.migrate_acks < 2)
    return;

  // Both sides done — release src slot; commit home on ReleaseEvent.
  send_cmd_(sess.worker_idx, make_release_cmd(ev.req_id()));
}

void ClusterScheduler::on_release_event_(const mambaserve::ReleaseEvent& ev) {
  std::lock_guard<std::mutex> lk(session_mu_);
  auto it = sessions_.find(ev.req_id());
  if (it == sessions_.end())
    return; // unknown/late event
  Session& sess = it->second;
  const size_t worker_idx = static_cast<size_t>(ev.worker_idx());

  if (sess.phase == SessionPhase::Migrating) {
    // Src cleanup after successful xfer transfer — commit sticky home to dst.
    if (worker_idx != sess.worker_idx)
      return;
    worker_stats_[sess.worker_idx].inflight--;
    sess.worker_idx = sess.migrate_dst_idx;
    // dst inflight already reserved in begin_migrate_
    sess.phase = SessionPhase::Decoding;
    sess.migrate_acks = 0;
    telemetry::counters().migrates_completed++;
    telemetry::trace(telemetry::TraceKind::MigrateCommit, ev.req_id(),
                     static_cast<int>(sess.worker_idx));
    LOG_DEBUG("migrate commit req_id=%llu home=%zu", static_cast<unsigned long long>(ev.req_id()),
              sess.worker_idx);
    transport_plane_->on_migrate_end(ev.req_id());
    if (sess.generated_tokens.empty()) {
      sess.phase = SessionPhase::Failed;
      sess.s = Status::RuntimeError("migrate commit with empty token history");
      send_cmd_(sess.worker_idx, make_release_cmd(ev.req_id()));
      return;
    }
    const int32_t last = sess.generated_tokens.back();
    send_cmd_(sess.worker_idx, make_decode_cmd(ev.req_id(), last));
    return;
  }

  if (sess.phase == SessionPhase::Failed) {
    // Fail path releases both src and dst; debit whichever worker reported.
    if (worker_idx < worker_stats_.size())
      worker_stats_[worker_idx].inflight--;
    return;
  }

  // Normal Done/Failed terminal release.
  if (worker_idx < worker_stats_.size())
    worker_stats_[worker_idx].inflight--;
  else
    worker_stats_[sess.worker_idx].inflight--;
}
