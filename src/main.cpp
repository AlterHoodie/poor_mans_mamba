#include "core/device.h"
#include "core/status.h"
#include "runtime/cluster_config.h"
#include "runtime/cluster_scheduler.h"
#include "runtime/model_registry.h"
#include "runtime/tokenizer.h"
#include "worker_factory.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

bool wait_all_done(ClusterScheduler& sched, const std::vector<uint64_t>& req_ids,
                   std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    bool all_done = true;
    for (uint64_t id : req_ids) {
      if (!sched.poll(id).done) {
        all_done = false;
        break;
      }
    }
    if (all_done)
      return true;
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

// Bring up a thread-mode cluster, submit `n_prompts` encoded prompts, wait for Done.
Status run_cluster_demo(const std::string& model_dir, int max_seq_length, Device device,
                        int n_workers, int n_prompts, int max_new_tokens) {
  if (n_workers <= 0)
    return Status::InvalidArgument("n_workers must be > 0");
  if (n_prompts <= 0)
    return Status::InvalidArgument("n_prompts must be > 0");
  if (max_new_tokens <= 0)
    return Status::InvalidArgument("max_new_tokens must be > 0");

  // Host-side tokenizer only; each worker loads its own model replica.
  auto entry_or = ModelRegistry::open(model_dir, max_seq_length);
  if (!entry_or.ok())
    return entry_or.status();
  Tokenizer tokenizer(*entry_or.value().cfg, model_dir);
  auto eos_or = tokenizer.eos_id();
  if (!eos_or.ok())
    return eos_or.status();

  ClusterConfig cfg{
      .model_dir = model_dir,
      .device = device,
      .n_workers = n_workers,
      .num_slots = std::max(n_prompts, 1),
      .transport = TransportBackend::MemcpyPeer,
      .max_seq_length = max_seq_length,
      .worker_mode = WorkerMode::Thread,
  };

  StatusOr<std::vector<WorkerSlot>> workers_or = create_cluster_workers(cfg);
  if (!workers_or.ok())
    return workers_or.status();

  ClusterScheduler sched;
  if (Status s = sched.start(cfg, std::move(workers_or.value())); !s.ok())
    return s;

  GenerateParams params{.max_new_tokens = max_new_tokens, .eos_id = eos_or.value()};

  std::vector<uint64_t> req_ids;
  req_ids.reserve(static_cast<size_t>(n_prompts));
  for (int i = 0; i < n_prompts; ++i) {
    const std::string prompt = "Hi How are you? (" + std::to_string(i) + ")";
    auto ids_or = tokenizer.encode(prompt);
    if (!ids_or.ok())
      return ids_or.status();
    StatusOr<uint64_t> id_or = sched.submit(std::move(ids_or.value()), params);
    if (!id_or.ok())
      return id_or.status();
    req_ids.push_back(id_or.value());
    std::cout << "submitted req_id=" << id_or.value() << " prompt=\"" << prompt << "\"\n";
  }

  if (!wait_all_done(sched, req_ids, std::chrono::seconds(120)))
    return Status::RuntimeError("timed out waiting for prompts to finish");

  for (uint64_t id : req_ids) {
    Response r = sched.poll(id);
    if (!r.s.ok())
      return r.s;
    auto text_or = tokenizer.decode(r.tokens);
    if (!text_or.ok())
      return text_or.status();
    std::cout << "req_id=" << id << " worker=" << r.worker_idx
              << " gen_tokens=" << r.gen_seq_len << " text=\"" << text_or.value() << "\"\n";
  }

  sched.shutdown();
  return Status::Ok();
}

} // namespace

int main(int argc, char** argv) {
  // app [model_dir] [max_seq] [CPU|GPU] [n_workers] [n_prompts] [max_new_tokens]
  const std::string model_dir = (argc > 1) ? argv[1] : "models/mamba2-130m-hf";
  const int max_seq_length = (argc > 2) ? std::stoi(argv[2]) : 2048;
  const Device device =
      (argc > 3 && std::string_view(argv[3]) == "GPU") ? Device::GPU : Device::CPU;
  const int n_workers = (argc > 4) ? std::stoi(argv[4]) : 2;
  const int n_prompts = (argc > 5) ? std::stoi(argv[5]) : 4;
  const int max_new_tokens = (argc > 6) ? std::stoi(argv[6]) : 16;

  const Status s =
      run_cluster_demo(model_dir, max_seq_length, device, n_workers, n_prompts, max_new_tokens);
  if (!s.ok()) {
    std::cerr << s.message() << '\n';
    return 1;
  }
  return 0;
}
