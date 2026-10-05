#include "worker_process.h"

#include "runtime/ipc/process_comm.h"
#include "runtime/worker.h"
#include "telemetry/log.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr const char* kWorkerFlag = "--mambaserve-worker";

const char* device_name(Device d) {
  switch (d) {
  case Device::GPU:
    return "gpu";
  case Device::CPU:
    return "cpu";
  case Device::NA:
    break;
  }
  return "na";
}

const char* transport_name(TransportBackend t) {
  switch (t) {
  case TransportBackend::MemcpyPeer:
    return "memcpy";
  case TransportBackend::Nccl:
    return "nccl";
  case TransportBackend::Nixl:
    return "nixl";
  }
  return "memcpy";
}

bool parse_device(std::string_view s, Device* out) {
  if (s == "gpu")
    *out = Device::GPU;
  else if (s == "cpu")
    *out = Device::CPU;
  else
    return false;
  return true;
}

bool parse_transport(std::string_view s, TransportBackend* out) {
  if (s == "memcpy")
    *out = TransportBackend::MemcpyPeer;
  else if (s == "nccl")
    *out = TransportBackend::Nccl;
  else if (s == "nixl")
    *out = TransportBackend::Nixl;
  else
    return false;
  return true;
}

bool parse_int(std::string_view s, int* out) {
  if (s.empty())
    return false;
  char* end = nullptr;
  const std::string tmp(s);
  const long v = std::strtol(tmp.c_str(), &end, 10);
  if (end == tmp.c_str() || *end != '\0')
    return false;
  *out = static_cast<int>(v);
  return true;
}

} // namespace

WorkerProcess::~WorkerProcess() {
  if (pid_ <= 0 || reaped_)
    return;
  if (!wait_for_exit(std::chrono::seconds(10))) {
    LOG_WARN("worker process %d did not exit in time; killing", static_cast<int>(pid_));
    kill();
    int status = 0;
    while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
    }
    reaped_ = true;
  }
}

bool WorkerProcess::wait_for_exit(std::chrono::milliseconds timeout) {
  if (pid_ <= 0)
    return true;
  if (reaped_)
    return true;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    int status = 0;
    const pid_t r = ::waitpid(pid_, &status, WNOHANG);
    if (r == pid_) {
      exit_code_ = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
      reaped_ = true;
      return true;
    }
    if (r < 0 && errno == ECHILD) {
      reaped_ = true;
      return true;
    }
    if (r < 0 && errno != EINTR)
      return false;
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

void WorkerProcess::kill() {
  if (pid_ > 0 && !reaped_)
    ::kill(pid_, SIGKILL);
}

StatusOr<std::unique_ptr<WorkerProcess>> spawn_worker_process(const ClusterConfig& cfg,
                                                              size_t index, int device_id,
                                                              int child_fd) {
  if (child_fd < 0)
    return Status::InvalidArgument("child_fd must be >= 0");

  // Build argv before fork: the child may only call async-signal-safe functions.
  std::vector<std::string> args = {
      "mambaserve-worker",
      kWorkerFlag,
      "--index=" + std::to_string(index),
      "--device-id=" + std::to_string(device_id),
      "--ipc-fd=" + std::to_string(child_fd),
      "--model-dir=" + cfg.model_dir,
      std::string("--device=") + device_name(cfg.device),
      "--n-workers=" + std::to_string(cfg.n_workers),
      "--num-slots=" + std::to_string(cfg.num_slots),
      std::string("--transport=") + transport_name(cfg.transport),
      "--max-seq-length=" + std::to_string(cfg.max_seq_length),
  };
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (std::string& a : args)
    argv.push_back(a.data());
  argv.push_back(nullptr);

  // fork the parent process
  const pid_t pid = ::fork();
  if (pid < 0)
    return Status::RuntimeError(std::string("fork failed: ") + std::strerror(errno));
  if (pid == 0) {
    // Child: the channel fds are close-on-exec; keep only this worker's end across exec.
    const int flags = ::fcntl(child_fd, F_GETFD);
    if (flags < 0 || ::fcntl(child_fd, F_SETFD, flags & ~FD_CLOEXEC) < 0)
      ::_exit(126);
    // since the forked process is simple a copy of the parent process, replace it with worker process using execv
    ::execv("/proc/self/exe", argv.data());
    ::_exit(127);
  }
  return std::make_unique<WorkerProcess>(pid);
}

std::optional<int> maybe_run_worker_process(int argc, char** argv) {
  if (argc < 2 || std::string_view(argv[1]) != kWorkerFlag)
    return std::nullopt;

  ClusterConfig cfg;
  int index = -1;
  int device_id = -1;
  int ipc_fd = -1;
  bool ok = true;

  for (int i = 2; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    const size_t eq = arg.find('=');
    if (eq == std::string_view::npos) {
      ok = false;
      break;
    }
    const std::string_view key = arg.substr(0, eq);
    const std::string_view val = arg.substr(eq + 1);
    if (key == "--index")
      ok = parse_int(val, &index);
    else if (key == "--device-id")
      ok = parse_int(val, &device_id);
    else if (key == "--ipc-fd")
      ok = parse_int(val, &ipc_fd);
    else if (key == "--model-dir")
      cfg.model_dir = std::string(val);
    else if (key == "--device")
      ok = parse_device(val, &cfg.device);
    else if (key == "--n-workers")
      ok = parse_int(val, &cfg.n_workers);
    else if (key == "--num-slots")
      ok = parse_int(val, &cfg.num_slots);
    else if (key == "--transport")
      ok = parse_transport(val, &cfg.transport);
    else if (key == "--max-seq-length")
      ok = parse_int(val, &cfg.max_seq_length);
    else
      ok = false;
    if (!ok)
      break;
  }
  if (!ok || index < 0 || device_id < 0 || ipc_fd < 0) {
    std::fprintf(stderr, "mambaserve worker: invalid arguments\n");
    return 2;
  }

  // The worker is already process-isolated; the parent validated the full config
  // (including worker_mode) before spawning, so the child keeps the default.
  cfg.worker_mode = WorkerMode::Thread;

  // No PR_SET_PDEATHSIG: it fires when the forking *thread* exits. If the parent dies,
  // its socket end closes, the worker's recv() fails and the loop exits on its own.
  Worker worker(
      WorkerBootstrap{
          .index = static_cast<size_t>(index),
          .device_id = device_id,
          .cfg = std::move(cfg),
          .ipc = std::make_unique<ProcessChannel>(ipc_fd),
      },
      WorkerRunMode::Inline);
  return worker.run();
}
