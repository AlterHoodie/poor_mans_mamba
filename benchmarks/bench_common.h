#pragma once

// Shared helpers for the transport_bench / cluster_bench drivers: tiny CLI
// parser, statistics, directory helpers, and run-metadata capture.

#include "comm/comm_factory.h"
#include "core/device.h"
#include "runtime/cluster_config.h"
#include "telemetry/recorder.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace bench {

// "--key value" or bare "--flag" (stored as "1").
class Args {
public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) {
        positional_.push_back(a);
        continue;
      }
      a = a.substr(2);
      const auto eq = a.find('=');
      if (eq != std::string::npos) {
        kv_[a.substr(0, eq)] = a.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        kv_[a] = argv[++i];
      } else {
        kv_[a] = "1";
      }
    }
  }

  bool has(const std::string& k) const { return kv_.count(k) != 0; }
  std::string str(const std::string& k, const std::string& def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  int64_t i64(const std::string& k, int64_t def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : std::stoll(it->second);
  }
  double f64(const std::string& k, double def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : std::stod(it->second);
  }
  // Comma separated list of ints, e.g. "1,2,4".
  std::vector<int64_t> i64_list(const std::string& k, std::vector<int64_t> def) const {
    auto it = kv_.find(k);
    if (it == kv_.end())
      return def;
    std::vector<int64_t> out;
    std::stringstream ss(it->second);
    std::string tok;
    while (std::getline(ss, tok, ','))
      if (!tok.empty())
        out.push_back(std::stoll(tok));
    return out;
  }
  std::vector<std::string> str_list(const std::string& k, std::vector<std::string> def) const {
    auto it = kv_.find(k);
    if (it == kv_.end())
      return def;
    std::vector<std::string> out;
    std::stringstream ss(it->second);
    std::string tok;
    while (std::getline(ss, tok, ','))
      if (!tok.empty())
        out.push_back(tok);
    return out;
  }
  const std::map<std::string, std::string>& all() const { return kv_; }

private:
  std::map<std::string, std::string> kv_;
  std::vector<std::string> positional_;
};

inline double percentile(std::vector<double> v, double p) {
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p / 100.0 * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

inline double mean(const std::vector<double>& v) {
  if (v.empty())
    return 0.0;
  double s = 0;
  for (double x : v)
    s += x;
  return s / static_cast<double>(v.size());
}

inline double stddev(const std::vector<double>& v) {
  if (v.size() < 2)
    return 0.0;
  const double m = mean(v);
  double s = 0;
  for (double x : v)
    s += (x - m) * (x - m);
  return std::sqrt(s / static_cast<double>(v.size() - 1));
}

inline std::string timestamp_str() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
  return buf;
}

inline void make_dirs(const std::string& dir) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
}

inline std::optional<TransportBackend> parse_backend(const std::string& s) {
  if (s == "memcpy" || s == "memcpy_peer")
    return TransportBackend::MemcpyPeer;
  if (s == "nccl")
    return TransportBackend::Nccl;
  if (s == "nixl")
    return TransportBackend::Nixl;
  return std::nullopt;
}

inline const char* backend_name(TransportBackend b) {
  switch (b) {
  case TransportBackend::MemcpyPeer:
    return "memcpy";
  case TransportBackend::Nccl:
    return "nccl";
  case TransportBackend::Nixl:
    return "nixl";
  }
  return "?";
}

inline std::optional<Device> parse_device(const std::string& s) {
  if (s == "GPU" || s == "gpu")
    return Device::GPU;
  if (s == "CPU" || s == "cpu")
    return Device::CPU;
  return std::nullopt;
}

inline const char* device_name(Device d) { return d == Device::GPU ? "GPU" : "CPU"; }

// Environment that influences transport/path selection; recorded in meta.json.
inline std::vector<std::string> transport_env_names() {
  return {"MAMBASERVE_LOG", "UCX_TLS",           "UCX_NET_DEVICES", "UCX_LOG_LEVEL",
          "NCCL_DEBUG",     "NCCL_P2P_DISABLE",  "NCCL_P2P_LEVEL",  "NCCL_SHM_DISABLE",
          "NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL", "CUDA_VISIBLE_DEVICES"};
}

// Common metadata: build/host/toolchain info + CLI args + transport env.
inline telemetry::MetaKV common_meta(const std::string& driver, const Args& args) {
  telemetry::MetaKV kv;
  kv.emplace_back("driver", driver);
  kv.emplace_back("git_hash", telemetry::build_git_hash());
  kv.emplace_back("timestamp", timestamp_str());
  kv.emplace_back("hostname", telemetry::capture_command("hostname"));
  kv.emplace_back("kernel", telemetry::capture_command("uname -r"));
  kv.emplace_back("nproc", telemetry::capture_command("nproc"));
  kv.emplace_back("cpu_model",
                  telemetry::capture_command("lscpu | grep 'Model name' | sed 's/.*: *//'"));
  kv.emplace_back("nvidia_smi_gpus",
                  telemetry::capture_command("nvidia-smi --query-gpu=index,name,memory.total,"
                                             "driver_version,clocks.max.sm --format=csv,noheader"));
  kv.emplace_back("nvidia_smi_topo", telemetry::capture_command("nvidia-smi topo -m"));
  kv.emplace_back("nvcc_version",
                  telemetry::capture_command("nvcc --version | tail -n 1"));
  kv.emplace_back("nccl_version",
                  telemetry::capture_command(
                      "ldconfig -p | grep -m1 libnccl.so | sed 's/.*=> *//'"));
  kv.emplace_back("ucx_version",
                  telemetry::capture_command(
                      "command -v ucx_info >/dev/null && ucx_info -v 2>/dev/null | head -n 1"));
#if defined(MAMBASERVE_WITH_NCCL) && MAMBASERVE_WITH_NCCL
  kv.emplace_back("built_with_nccl", "1");
#else
  kv.emplace_back("built_with_nccl", "0");
#endif
#if defined(MAMBASERVE_WITH_NIXL) && MAMBASERVE_WITH_NIXL
  kv.emplace_back("built_with_nixl", "1");
#else
  kv.emplace_back("built_with_nixl", "0");
#endif
  for (const auto& [k, v] : args.all())
    kv.emplace_back("arg." + k, v);
  for (auto& e : telemetry::capture_env(transport_env_names()))
    kv.emplace_back("env." + e.first, e.second);
  kv.emplace_back("scope_note",
                  "intra-node; workers are threads of one process by default (cluster_bench "
                  "--worker-mode process spawns one process per worker); NIXL uses UCX over "
                  "local GPUs (cuda_ipc or host staged), not network RDMA");
  return kv;
}

} // namespace bench
