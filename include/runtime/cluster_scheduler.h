#pragma once

#include "core/status.h"
#include "io/config.h"
#include "runtime/model_registry.h"
#include "runtime/worker.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// Response is a snapshot of session state at poll() time: tokens generated so
// far, whether generation has finished (Done or Failed), and terminal status.
struct Response {
  uint64_t req_id = 0;
  std::vector<int32_t> tokens; // generated tokens so far
  int gen_seq_len = 0;
  bool done = false;
  Status s = Status::Ok();
};

struct WorkerSlot {
  int device_id = -1;
  std::unique_ptr<Worker> worker;
};

enum class SessionPhase { Prefilling, Decoding, Done, Failed };

// Control-plane bookkeeping for one in-flight generate() request. Only the
// event-processing thread and poll()/submit() (under session_mu_) touch this.
struct Session {
  std::vector<int32_t> generated_tokens;
  GenerateParams params;
  SessionPhase phase = SessionPhase::Prefilling;
  Status s = Status::Ok();
  size_t worker_idx = 0;
};

// takes in some sort of config , spins up that many threads of workers
// , a placement policy that gives returns a worker view for a given request 
// pushes cmds using workers->enqueue
// listens to worker events
class ClusterScheduler{
    private:
        // for event queue workers pushing events into
        std::mutex event_mu_;
        std::condition_variable event_cv_;
        std::deque<Event> eventq_;
        std::thread event_thread_;
        bool stop_ = false;

        // array of workers
        std::vector<WorkerSlot> workers_;

        // round-robin worker assignment at admit time. This is a placeholder:
        // the real least-loaded placement policy is a later step.
        std::atomic<size_t> next_worker_{0};

        // for simple request id generation
        std::atomic<uint64_t> req_id_counter_{0};

        // session_mu_ guards sessions_; submit()/poll() take it from the
        // caller's thread, the event loop takes it while applying events.
        mutable std::mutex session_mu_;
        std::unordered_map<uint64_t, Session> sessions_;

        StatusOr<WorkerSlot> create_worker_(const ModelEntry& entry, Device kind, int device_id,
                                            int num_slots);

        // drains eventq_ and applies PrefillDone/DecodeDone/Released/Migrated
        // to the matching session, driving the next command (decode/release).
        void event_loop_();
        void handle_event_(Event ev);
        void on_prefill_event_(const PrefillEvent& ev);
        void on_decode_event_(const DecodeEvent& ev);

    public:
        ClusterScheduler() = default;

        // submit token ids directly; tokenization is not wired into the
        // cluster scheduler yet (see runtime/tokenizer.h for the single-model
        // path used by main.cpp / benchmarks).
        StatusOr<uint64_t> submit(std::vector<int32_t> tokens, GenerateParams params);
        Response poll(uint64_t req_id) const;

        Status load_model(const std::string model_dir, Device kind, int n_devices);
        
        // shutdown all workers (waits for all requests to be done) then shutsdown
        void shutdown();

        // calls shutdown
        ~ClusterScheduler();
};
