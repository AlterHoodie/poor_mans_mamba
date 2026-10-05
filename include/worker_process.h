#pragma once

#include "core/status.h"
#include "runtime/cluster_config.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <sys/types.h>

// Process-mode workers: the parent re-executes its own binary (/proc/self/exe) with
// `--mambaserve-worker ...` and hands the child one end of a Unix socketpair as an
// inherited fd. The child runs a Worker inline and speaks the same Envelope protocol
// as a worker thread, so ClusterScheduler is agnostic to how the host created the worker
// (see create_cluster_workers in worker_factory.h).
//
// Any binary that can host process-mode workers (tests, benches) must call
// maybe_run_worker_process() first thing in main().

// RAII handle for a spawned worker process: the destructor waits for a graceful exit
// (after the parent has sent ShutdownCmd / closed the channel) and SIGKILLs a child
// that does not exit in time.
class WorkerProcess {
public:
  explicit WorkerProcess(pid_t pid) : pid_(pid) {}
  ~WorkerProcess();

  WorkerProcess(const WorkerProcess&) = delete;
  WorkerProcess& operator=(const WorkerProcess&) = delete;

  pid_t pid() const { return pid_; }

  // True once the child has exited and been reaped.
  bool wait_for_exit(std::chrono::milliseconds timeout);
  void kill();

  // Exit code of a reaped child (128 + signal if it was killed); -1 until reaped.
  int exit_code() const { return exit_code_; }

private:
  pid_t pid_ = -1;
  bool reaped_ = false;
  int exit_code_ = -1;
};

// Forks and execs a worker process for (index, device_id). `child_fd` is the child end
// of a socketpair (see make_process_socketpair); it is inherited by the child, and the
// caller keeps ownership of its own copy. Does not validate cfg: callers go through
// create_cluster_workers (which validates).
StatusOr<std::unique_ptr<WorkerProcess>> spawn_worker_process(const ClusterConfig& cfg,
                                                              size_t index, int device_id,
                                                              int child_fd);

// If argv says this invocation is a spawned worker, runs it to completion and returns
// the process exit code; otherwise returns nullopt and main() proceeds normally.
std::optional<int> maybe_run_worker_process(int argc, char** argv);
