#include "runtime/worker.h"

#include "comm/comm_agent.h"
#include "comm/comm_factory.h"
#include "core/status.h"
#include "ops/cpu/reductions.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/ipc/proto_convert.h"
#include "runtime/model_registry.h"
#include "telemetry/log.h"
#include "telemetry/nvtx.h"
#include "telemetry/recorder.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

StatusOr<int32_t> argmax_gpu_d2h(Tensor& logits) {
#ifdef MAMBASERVE_WITH_CUDA
  StatusOr<int64_t> n_or = logits.numel();
  if (!n_or.ok())
    return Status(n_or.status());
  const int64_t n = n_or.value();
  if (n <= 0)
    return Status::InvalidArgument("empty logits");

  std::vector<float> host(static_cast<size_t>(n));
  cudaError_t err = cudaMemcpy(host.data(), logits.buffer.ptr,
                               static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(err));
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

mambaserve::PrefillEvent make_prefill_event(uint64_t req_id, int32_t token, const Status& s) {
  mambaserve::PrefillEvent ev;
  ev.set_req_id(req_id);
  ev.set_token(token);
  *ev.mutable_status() = to_proto(s);
  return ev;
}

mambaserve::DecodeEvent make_decode_event(uint64_t req_id, int32_t token, const Status& s) {
  mambaserve::DecodeEvent ev;
  ev.set_req_id(req_id);
  ev.set_token(token);
  *ev.mutable_status() = to_proto(s);
  return ev;
}

mambaserve::MigrateEvent make_migrate_event(uint64_t req_id, mambaserve::XferRole role,
                                            const Status& s) {
  mambaserve::MigrateEvent ev;
  ev.set_req_id(req_id);
  ev.set_role(role);
  *ev.mutable_status() = to_proto(s);
  return ev;
}

mambaserve::ReleaseEvent make_release_event(uint64_t req_id, size_t worker_idx, const Status& s) {
  mambaserve::ReleaseEvent ev;
  ev.set_req_id(req_id);
  ev.set_worker_idx(worker_idx);
  *ev.mutable_status() = to_proto(s);
  return ev;
}

} // namespace

StatusOr<int32_t> Worker::sample_(Tensor& logits) {
  if (logits.buffer.device == Device::CPU)
    return argmax(logits);
  if (logits.buffer.device == Device::GPU)
    return argmax_gpu_d2h(logits);
  return Status::InvalidArgument("logits device not supported for sampling");
}

mambaserve::PrefillEvent Worker::prefill_(uint64_t req_id, std::span<const int32_t> tokens) {
  if (!pool_)
    return make_prefill_event(req_id, -1, Status::RuntimeError("cache pool not yet initialized"));
  if (!alloc_)
    return make_prefill_event(req_id, -1, Status::RuntimeError("allocator not yet initialized"));
  if (tokens.empty())
    return make_prefill_event(req_id, -1, Status::InvalidArgument("token_ids cannot be empty"));

  auto it = cache_handles_.find(req_id);
  if (it == cache_handles_.end())
    return make_prefill_event(
        req_id, -1,
        Status::NotFound("Could not find Cache Handle for req_id" + std::to_string(req_id)));

  StatusOr<std::span<LayerCacheView>> views = pool_->layer_views(it->second);
  if (!views.ok())
    return make_prefill_event(req_id, -1, views.status());

  StatusOr<Tensor> prefill = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc_.get());
    return runner_->prefill(tokens, views.value());
  }();
  if (!prefill.ok())
    return make_prefill_event(req_id, -1, prefill.status());

  if (Status s = pool_->set_seq_len(it->second, static_cast<int64_t>(tokens.size())); !s.ok())
    return make_prefill_event(req_id, -1, s);

  StatusOr<int32_t> sampled = sample_(prefill.value());
  if (!sampled.ok())
    return make_prefill_event(req_id, -1, sampled.status());

  return make_prefill_event(req_id, sampled.value(), Status::Ok());
}

mambaserve::DecodeEvent Worker::decode_(uint64_t req_id, const int32_t token) {
  if (!pool_)
    return make_decode_event(req_id, -1, Status::RuntimeError("cache pool not yet initialized"));
  if (!alloc_)
    return make_decode_event(req_id, -1, Status::RuntimeError("allocator not yet initialized"));

  auto it = cache_handles_.find(req_id);
  if (it == cache_handles_.end())
    return make_decode_event(
        req_id, -1,
        Status::NotFound("Could not find Cache Handle for req_id" + std::to_string(req_id)));

  StatusOr<std::span<LayerCacheView>> views = pool_->layer_views(it->second);
  if (!views.ok())
    return make_decode_event(req_id, -1, views.status());

  StatusOr<int64_t> past = pool_->seq_len(it->second);
  if (!past.ok())
    return make_decode_event(req_id, -1, past.status());

  StatusOr<Tensor> logits = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc_.get());
    return runner_->decode(token, views.value(), past.value());
  }();
  if (!logits.ok())
    return make_decode_event(req_id, -1, logits.status());

  if (Status s = pool_->set_seq_len(it->second, past.value() + 1); !s.ok())
    return make_decode_event(req_id, -1, s);

  StatusOr<int32_t> sampled = sample_(logits.value());
  if (!sampled.ok())
    return make_decode_event(req_id, -1, sampled.status());

  return make_decode_event(req_id, sampled.value(), Status::Ok());
}

Status Worker::register_(uint64_t req_id) {
  if (!pool_)
    return Status::RuntimeError("Cache Pool not yet initialized");
  if (!alloc_)
    return Status::RuntimeError("Allocator not yet initialized");
  if (cache_handles_.find(req_id) != cache_handles_.end())
    return Status::RuntimeError("Cache Handle already registered for request_id" +
                                std::to_string(req_id));

  StatusOr<CacheHandle> handle = pool_->acquire();
  if (!handle.ok())
    return handle.status();
  cache_handles_[req_id] = handle.value();
  return Status::Ok();
}

Status Worker::release_(uint64_t req_id) {
  if (!pool_)
    return Status::RuntimeError("Cache Pool not yet initialized");
  if (!alloc_)
    return Status::RuntimeError("Allocator not yet initialized");

  auto it = cache_handles_.find(req_id);
  if (it == cache_handles_.end())
    return Status::NotFound("Could not find Cache Handle for req_id" + std::to_string(req_id));

  Status s = pool_->release(it->second);
  cache_handles_.erase(it);
  return s;
}

std::optional<mambaserve::MigrateEvent> Worker::post_xfer_(const mambaserve::MigrateCmd& cmd,
                                                           int64_t seq_len) {
  const XferRole role = from_proto(cmd.role());
  auto hit = cache_handles_.find(cmd.req_id());
  if (hit == cache_handles_.end())
    return make_migrate_event(cmd.req_id(), cmd.role(),
                              Status::NotFound("migrate missing cache handle for req_id " +
                                               std::to_string(cmd.req_id())));

  StatusOr<void*> slot_or = pool_->slot_ptr(hit->second);
  if (!slot_or.ok())
    return make_migrate_event(cmd.req_id(), cmd.role(), slot_or.status());

  XferDesc desc{
      .local_ptr = slot_or.value(),
      .remote_ptr = nullptr,
      .bytes = pool_->slot_bytes(),
      .peer_device_id = cmd.peer_device_id(),
      .role = role,
      .xfer_id = cmd.req_id(),
      .seq_len = seq_len,
  };

  StatusOr<XferHandle> handle_or = comm_agent_->post(desc);
  if (!handle_or.ok())
    return make_migrate_event(cmd.req_id(), cmd.role(), handle_or.status());

  pending_transfers_[cmd.req_id()] = PendingXfer{.handle = handle_or.value(), .role = role};
  // Backend may need to tell its peer something (e.g. NIXL Recv slot announce).
  if (auto announce = comm_agent_->make_post_announce(desc))
    emit_transport_(std::move(*announce));
  telemetry::trace(telemetry::TraceKind::MigrateXferPosted, cmd.req_id(), static_cast<int>(index_),
                   role == XferRole::Recv ? 1 : 0, static_cast<int64_t>(desc.bytes));
  LOG_DEBUG("migrate xfer posted req_id=%llu worker=%zu role=%s peer=%d bytes=%zu",
            static_cast<unsigned long long>(cmd.req_id()), index_,
            role == XferRole::Recv ? "recv" : "send", cmd.peer_device_id(), desc.bytes);
  return std::nullopt;
}

std::optional<mambaserve::MigrateEvent> Worker::migrate_recv_(const mambaserve::MigrateCmd& cmd) {
  // Recv: register a fresh slot for the incoming cache, then post.
  if (Status s = register_(cmd.req_id()); !s.ok())
    return make_migrate_event(cmd.req_id(), cmd.role(), s);
  return post_xfer_(cmd, /*seq_len=*/0);
}

std::optional<mambaserve::MigrateEvent> Worker::migrate_send_(const mambaserve::MigrateCmd& cmd) {
  // Send: existing handle required; publish seq_len with the transfer.
  auto hit = cache_handles_.find(cmd.req_id());
  if (hit == cache_handles_.end())
    return make_migrate_event(cmd.req_id(), cmd.role(),
                              Status::NotFound("Send migrate missing cache handle for req_id " +
                                               std::to_string(cmd.req_id())));

  StatusOr<int64_t> sl = pool_->seq_len(hit->second);
  if (!sl.ok())
    return make_migrate_event(cmd.req_id(), cmd.role(), sl.status());
  return post_xfer_(cmd, sl.value());
}

std::optional<mambaserve::MigrateEvent> Worker::migrate_(const mambaserve::MigrateCmd& cmd) {
  if (!pool_)
    return make_migrate_event(cmd.req_id(), cmd.role(),
                              Status::RuntimeError("cache pool not yet initialized"));
  if (!alloc_)
    return make_migrate_event(cmd.req_id(), cmd.role(),
                              Status::RuntimeError("allocator not yet initialized"));
  if (!comm_agent_)
    return make_migrate_event(cmd.req_id(), cmd.role(),
                              Status::RuntimeError("comm agent not yet initialized"));

  if (cmd.role() == mambaserve::XFER_RECV)
    return migrate_recv_(cmd);
  return migrate_send_(cmd);
}

void Worker::emit_event_(mambaserve::Event ev) {
  if (!ipc_)
    return;
  mambaserve::Envelope env;
  *env.mutable_event() = std::move(ev);
  if (Status s = ipc_->send(env); !s.ok())
    LOG_ERROR("worker %zu emit failed: %s", index_, s.message().c_str());
}

void Worker::emit_transport_(mambaserve::TransportControl msg) {
  if (!ipc_)
    return;
  mambaserve::Envelope env;
  *env.mutable_transport() = std::move(msg);
  if (Status s = ipc_->send(env); !s.ok())
    LOG_ERROR("worker %zu emit transport failed: %s", index_, s.message().c_str());
}

void Worker::handle_transport_(const mambaserve::TransportControl& msg) {
  if (!comm_agent_)
    return;
  if (Status s = comm_agent_->handle_transport(msg); !s.ok())
    LOG_WARN("worker %zu transport control failed: %s", index_, s.message().c_str());
  // e.g. NCCL bootstrap ack; sent even when handling failed so the parent can react.
  if (auto reply = comm_agent_->take_transport_reply())
    emit_transport_(std::move(*reply));
}

Worker::Worker(WorkerBootstrap boot, WorkerRunMode mode)
    : index_(boot.index), device_id_(boot.device_id), cfg_(std::move(boot.cfg)),
      ipc_(std::move(boot.ipc)) {
  if (mode == WorkerRunMode::Thread)
    thread_ = std::thread(&Worker::loop_, this);
}

int Worker::run() {
  if (thread_.joinable())
    return 1; // thread-mode workers already run their own loop
  loop_();
  return init_ok_ ? 0 : 1;
}

Status Worker::init_() {
  if (device_id_ < 0)
    return Status::InvalidArgument("device id cannot be less than 0");

#ifdef MAMBASERVE_WITH_CUDA
  if (cfg_.device == Device::GPU) {
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess)
      return Status::RuntimeError(std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
  }
#endif

  ASSIGN_OR_RETURN(alloc_, create_device_allocator(cfg_.device, device_id_));

  if (cfg_.max_seq_length <= 0)
    return Status::InvalidArgument("max_seq_length must be > 0");
  ModelEntry entry;
  ASSIGN_OR_RETURN(entry, ModelRegistry::open(cfg_.model_dir, cfg_.max_seq_length));
  if (!entry.cfg)
    return Status::RuntimeError("ModelEntry has no config");

  ASSIGN_OR_RETURN(runner_, entry.create_runner(*alloc_));
  ASSIGN_OR_RETURN(
      pool_, create_cache_pool(*entry.cfg, alloc_.get(), cfg_.num_slots, create_cache_layout));
  // No shared transport state: backends coordinate through the parent's control plane,
  // so this works identically for a worker thread and a worker process.
  ASSIGN_OR_RETURN(comm_agent_, create_comm_agent(cfg_, device_id_, nullptr));

  // No-op for MemcpyPeer/NCCL; required for NIXL.
  return comm_agent_->register_slab(pool_->slab_ptr(), pool_->slab_bytes());
}

Worker::~Worker() {
  stop_ = true;
  if (ipc_)
    ipc_->close();
  if (thread_.joinable())
    thread_.join();
}

void Worker::poll_transfer_states_() {
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

  for (auto it = pending_transfers_.begin(); it != pending_transfers_.end();) {
    XferState state = comm_agent_->poll(it->second.handle);
    // Backend may need to tell its peer the transfer finished (e.g. MemcpyPeer
    // Send -> Recv). Sent before our MigrateEvent so the peer is never behind us.
    if (state != XferState::Pending) {
      if (auto done = comm_agent_->make_completion_announce(it->second.handle, state))
        emit_transport_(std::move(*done));
    }
    switch (state) {
    case XferState::Pending:
      ++it;
      break;
    case XferState::Done: {
      mambaserve::Event ev;
      if (it->second.role == XferRole::Recv) {
        auto hit = cache_handles_.find(it->first);
        if (hit != cache_handles_.end()) {
          const int64_t sl = comm_agent_->xfer_seq_len(it->second.handle);
          if (Status s = pool_->set_seq_len(hit->second, sl); !s.ok()) {
            xfer_done(it->first, it->second.role, false);
            *ev.mutable_migrate() = make_migrate_event(it->first, to_proto(it->second.role), s);
            emit_event_(std::move(ev));
            it = pending_transfers_.erase(it);
            break;
          }
        }
      }
      xfer_done(it->first, it->second.role, true);
      *ev.mutable_migrate() =
          make_migrate_event(it->first, to_proto(it->second.role), Status::Ok());
      emit_event_(std::move(ev));
      it = pending_transfers_.erase(it);
      break;
    }
    case XferState::Error:
      xfer_done(it->first, it->second.role, false);
      LOG_ERROR("transfer failed req_id=%llu worker=%zu",
                static_cast<unsigned long long>(it->first), index_);
      {
        mambaserve::Event ev;
        *ev.mutable_migrate() = make_migrate_event(
            it->first, to_proto(it->second.role),
            Status::RuntimeError("transfer failed for req_id: " + std::to_string(it->first)));
        emit_event_(std::move(ev));
      }
      it = pending_transfers_.erase(it);
      break;
    }
  }
}

void Worker::handle_command_(const mambaserve::Command& cmd) {
  mambaserve::Event ev;
  switch (cmd.body_case()) {

  case mambaserve::Command::kPrefill: {
    MS_NVTX_RANGE("prefill");
    const auto& c = cmd.prefill();
    const int w = static_cast<int>(index_);
    // track prefill start
    telemetry::trace(telemetry::TraceKind::PrefillStart, c.req_id(), w,
                     static_cast<int64_t>(c.tokens_size()));

    if (auto s = register_(c.req_id()); !s.ok()) {
      if (s.code() == Code::kOOM)
        telemetry::counters().slot_rejects++;
      LOG_WARN("prefill register failed req_id=%llu worker=%d: %s",
               static_cast<unsigned long long>(c.req_id()), w, s.message().c_str());

      // track prefill fail
      telemetry::trace(telemetry::TraceKind::PrefillEnd, c.req_id(), w, 0);

      *ev.mutable_prefill() = make_prefill_event(c.req_id(), -1, s);
      emit_event_(std::move(ev));
      return;
    }
    auto pref =
        prefill_(c.req_id(),
                 std::span<const int32_t>(c.tokens().data(), static_cast<size_t>(c.tokens_size())));
    // track prefill end
    telemetry::trace(telemetry::TraceKind::PrefillEnd, c.req_id(), w,
                     status_ok(pref.status()) ? 1 : 0);

    *ev.mutable_prefill() = std::move(pref);
    emit_event_(std::move(ev));
    return;
  }

  case mambaserve::Command::kDecode: {
    MS_NVTX_RANGE("decode");
    const auto& c = cmd.decode();
    const int w = static_cast<int>(index_);
    // track each decode start
    telemetry::trace(telemetry::TraceKind::DecodeStart, c.req_id(), w);

    auto dec = decode_(c.req_id(), c.token());

    // track each decode end
    telemetry::trace(telemetry::TraceKind::DecodeEnd, c.req_id(), w, 0,
                     status_ok(dec.status()) ? 1 : 0);

    *ev.mutable_decode() = std::move(dec);
    emit_event_(std::move(ev));
    return;
  }
  case mambaserve::Command::kRelease: {
    const auto& c = cmd.release();
    *ev.mutable_release() = make_release_event(c.req_id(), index_, release_(c.req_id()));
    emit_event_(std::move(ev));
    return;
  }
  case mambaserve::Command::kMigrate: {
    MS_NVTX_RANGE("migrate_post");
    const auto& c = cmd.migrate();
    if (auto m = migrate_(c)) {
      LOG_ERROR("migrate post failed req_id=%llu worker=%zu: %s",
                static_cast<unsigned long long>(c.req_id()), index_, m->status().message().c_str());

      *ev.mutable_migrate() = std::move(*m);
      emit_event_(std::move(ev));
    }
    return;
  }
  case mambaserve::Command::kShutdown:
    stop_ = true;
    return;
  case mambaserve::Command::kResetTraces: {
    telemetry::Recorder& rec = telemetry::Recorder::instance();
    rec.reset();
    rec.set_epoch_ns(cmd.reset_traces().epoch_ns());
    rec.enable(true);
    return;
  }
  case mambaserve::Command::kDumpTraces: {
    telemetry::Recorder& rec = telemetry::Recorder::instance();
    const std::vector<telemetry::TraceEvent> snap = rec.snapshot();
    mambaserve::Event batch_ev;
    auto* batch = batch_ev.mutable_trace_batch();
    batch->set_worker_idx(index_);
    batch->set_epoch_ns(rec.epoch_ns());
    for (const telemetry::TraceEvent& e : snap) {
      mambaserve::TraceEventMsg* m = batch->add_events();
      m->set_t_ns(e.t_ns);
      m->set_kind(static_cast<uint32_t>(e.kind));
      m->set_req_id(e.req_id);
      m->set_worker(e.worker);
      m->set_a(e.a);
      m->set_b(e.b);
    }
    auto* ctr = batch->mutable_counters();
    ctr->set_bytes_migrated(rec.counters.bytes_migrated.load());
    ctr->set_slot_rejects(rec.counters.slot_rejects.load());
    emit_event_(std::move(batch_ev));
    rec.clear_events();
    rec.counters.bytes_migrated = 0;
    rec.counters.slot_rejects = 0;
    return;
  }
  case mambaserve::Command::BODY_NOT_SET:
    LOG_WARN("worker %zu received empty command", index_);
    return;
  }
}

void Worker::loop_() {
  // Build everything on the executing thread so device context lives where inference runs.
  const Status init = init_();
  if (!init.ok())
    LOG_ERROR("worker %zu init failed: %s", index_, init.message().c_str());

  // Announce ready (or init failure) to the parent.
  {
    mambaserve::Event ev;
    auto* ready = ev.mutable_ready();
    ready->set_worker_idx(index_);
    *ready->mutable_status() = to_proto(init);
    emit_event_(std::move(ev));
  }
  if (!init.ok())
    return;
  init_ok_ = true;

  // Process-mode workers (Inline) own a private Recorder; enable it so Prefill/Decode/Xfer
  // events are captured until the parent syncs epochs via ResetTracesCmd. Thread-mode
  // workers share the parent's recorder, which cluster_bench enables itself.
  if (!thread_.joinable())
    telemetry::Recorder::instance().enable(true);

  // Backend registration info (e.g. NIXL agent metadata) goes up after Ready so the
  // parent's Ready barrier only ever sees Ready events.
  if (auto announce = comm_agent_->make_register_announce())
    emit_transport_(std::move(*announce));

  for (;;) {
    poll_transfer_states_();
    if (!ipc_)
      return;
    // Graceful stop: drain in-flight transfers, then exit.
    if (stop_ && pending_transfers_.empty())
      return;

    // While transfers are outstanding, don't block the poll loop on IPC.
    const bool draining = !pending_transfers_.empty();
    StatusOr<mambaserve::Envelope> env_or = draining ? ipc_->try_recv() : ipc_->recv();
    if (!env_or.ok()) {
      // Idle (Empty) or a hard error while transfers may still complete out-of-band:
      // keep polling unless we're stopping or there is nothing left to wait on.
      if (draining && (env_or.status().code() == Code::kEmpty || !stop_)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      return;
    }

    switch (env_or.value().body_case()) {
    case mambaserve::Envelope::kCmd:
      handle_command_(env_or.value().cmd());
      break;
    case mambaserve::Envelope::kTransport:
      handle_transport_(env_or.value().transport());
      break;
    case mambaserve::Envelope::kEvent:
    case mambaserve::Envelope::BODY_NOT_SET:
      break;
    }
  }
}