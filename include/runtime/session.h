#pragma once

#include "core/status.h"
#include "runtime/worker.h"

#include <cstdint>
#include <vector>

enum class SessionPhase { Prefilling, Decoding, Migrating, Done, Failed };

// Control-plane bookkeeping for one in-flight generate() request. Only the
// event-processing thread and poll()/submit() (under session_mu_) touch this.
struct Session {
  std::vector<int32_t> generated_tokens;
  GenerateParams params;
  SessionPhase phase = SessionPhase::Prefilling;
  Status s = Status::Ok();
  size_t prompt_len = 0;
  size_t seq_len = 0;
  size_t worker_idx = 0;
  void* slot_ptr = nullptr; // cluster level bookeeping of slot_ptr
  size_t migrate_dst_idx = 0;
  void* migrate_slot_ptr = nullptr; // cluter level bookeeping of slot_ptr during migration
  int migrate_acks = 0;

  // When set, the next decode boundary starts migrate instead of another Decode.
  bool migrate_pending = false;
};

// Lightweight session row for rebalance (avoids copying generated_tokens).
struct SessionView {
  uint64_t req_id = 0;
  size_t worker_idx = 0;
  SessionPhase phase = SessionPhase::Prefilling;
  bool migrate_pending = false;
  size_t gen_len = 0;
};