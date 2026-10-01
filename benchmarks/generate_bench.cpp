#include "core/device.h"
#include "core/status.h"
#include "core/tensor.h"
#include "ops/cpu/reductions.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/model_registry.h"
#include "runtime/runner/runner.h"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

struct BenchConfig {
  std::string model_dir = MAMBA_BENCH_MODEL_DIR;
  Device device = Device::CPU;
};

BenchConfig g_cfg;
std::unique_ptr<DeviceAllocator> g_alloc;
std::unique_ptr<CachePool> g_pool;
std::unique_ptr<Runner> g_runner;
ModelConfig g_model_cfg;
int32_t g_fill_token = 0;

constexpr int kMaxSeqLength = 2048 + 128;
constexpr int64_t kPromptLens[] = {32, 128, 512, 1024, 2048};
constexpr int64_t kGenLens[] = {1, 16, 64, 128};

void sync_device() {
#ifdef MAMBASERVE_WITH_CUDA
  if (g_cfg.device == Device::GPU) {
    cudaDeviceSynchronize();
  }
#endif
}

StatusOr<int32_t> sample_token(Tensor& logits) {
  if (logits.buffer.device == Device::CPU) {
    return argmax(logits);
  }

#ifdef MAMBASERVE_WITH_CUDA
  if (logits.buffer.device != Device::GPU) {
    return Status::InvalidArgument("logits device not supported for sampling");
  }
  StatusOr<int64_t> n_or = logits.numel();
  if (!n_or.ok())
    return Status(n_or.status());
  const int64_t n = n_or.value();
  if (n <= 0)
    return Status::InvalidArgument("empty logits");

  std::vector<float> host(static_cast<size_t>(n));
  cudaError_t err =
      cudaMemcpy(host.data(), logits.buffer.ptr, static_cast<size_t>(n) * sizeof(float),
                 cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return Status::InvalidArgument(std::string("cudaMemcpy D2H failed: ") +
                                   cudaGetErrorString(err));
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
  return Status::InvalidArgument("GPU sampling requires CUDA build");
#endif
}

double percentile_sorted_ms(std::vector<double> samples_s, double p) {
  if (samples_s.empty())
    return 0.0;
  std::sort(samples_s.begin(), samples_s.end());
  const double idx = (p / 100.0) * static_cast<double>(samples_s.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi)
    return samples_s[lo] * 1e3;
  const double w = idx - static_cast<double>(lo);
  return (samples_s[lo] * (1.0 - w) + samples_s[hi] * w) * 1e3;
}

std::vector<int32_t> make_prompt(int64_t prompt_len) {
  return std::vector<int32_t>(static_cast<size_t>(prompt_len), g_fill_token);
}

using Clock = std::chrono::steady_clock;

double elapsed_s(Clock::time_point t0, Clock::time_point t1) {
  return std::chrono::duration<double>(t1 - t0).count();
}

bool parse_args(int argc, char** argv, BenchConfig& cfg, int& bench_argc,
                char**& bench_argv) {
  static std::vector<char*> filtered;
  filtered.clear();
  filtered.push_back(argv[0]);

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--model_dir") {
      if (i + 1 >= argc) {
        std::cerr << "--model_dir requires a path\n";
        return false;
      }
      cfg.model_dir = argv[++i];
    } else if (arg.rfind("--model_dir=", 0) == 0) {
      cfg.model_dir = std::string(arg.substr(std::strlen("--model_dir=")));
    } else if (arg == "--device") {
      if (i + 1 >= argc) {
        std::cerr << "--device requires CPU or GPU\n";
        return false;
      }
      const std::string_view d(argv[++i]);
      if (d == "CPU" || d == "cpu") {
        cfg.device = Device::CPU;
      } else if (d == "GPU" || d == "gpu") {
        cfg.device = Device::GPU;
      } else {
        std::cerr << "unknown --device: " << d << " (use CPU or GPU)\n";
        return false;
      }
    } else if (arg.rfind("--device=", 0) == 0) {
      const std::string_view d = arg.substr(std::strlen("--device="));
      if (d == "CPU" || d == "cpu") {
        cfg.device = Device::CPU;
      } else if (d == "GPU" || d == "gpu") {
        cfg.device = Device::GPU;
      } else {
        std::cerr << "unknown --device: " << d << " (use CPU or GPU)\n";
        return false;
      }
    } else {
      filtered.push_back(argv[i]);
    }
  }

  bench_argc = static_cast<int>(filtered.size());
  bench_argv = filtered.data();
  return true;
}

Status load_model(const BenchConfig& cfg) {
#ifndef MAMBASERVE_WITH_CUDA
  if (cfg.device == Device::GPU) {
    return Status::InvalidArgument("GPU requested but build lacks MAMBASERVE_WITH_CUDA");
  }
#endif

  auto entry_or = ModelRegistry::open(cfg.model_dir, kMaxSeqLength);
  if (!entry_or.ok())
    return entry_or.status();
  ModelEntry entry = std::move(entry_or.value());
  g_model_cfg = *entry.cfg;

  g_fill_token = g_model_cfg.bos_token_id;
  if (g_fill_token < 0)
    g_fill_token = g_model_cfg.pad_token_id;
  if (g_fill_token < 0)
    g_fill_token = 0;

  auto alloc_or = create_device_allocator(cfg.device, /*device_id=*/0);
  if (!alloc_or.ok())
    return alloc_or.status();
  g_alloc = std::move(alloc_or.value());

  auto runner_or = entry.create_runner(*g_alloc);
  if (!runner_or.ok())
    return runner_or.status();
  g_runner = std::move(runner_or.value());

  auto pool_or = create_cache_pool(g_model_cfg, g_alloc.get(), /*num_slots=*/1, create_cache_layout);
  if (!pool_or.ok())
    return pool_or.status();
  g_pool = std::move(pool_or.value());
  return Status::Ok();
}

void BM_Prefill(benchmark::State& state) {
  const int64_t prompt_len = state.range(0);
  if (prompt_len > g_model_cfg.max_seq_length) {
    state.SkipWithError("prompt_len exceeds model max_seq_length");
    return;
  }

  const auto prompt = make_prompt(prompt_len);

  for (auto _ : state) {
    StatusOr<CacheHandle> handle_or = g_pool->acquire();
    if (!handle_or.ok()) {
      state.SkipWithError(handle_or.status().message().c_str());
      return;
    }
    CacheHandle handle = std::move(handle_or.value());
    StatusOr<std::span<LayerCacheView>> layers = g_pool->layer_views(handle);
    if (!layers.ok()) {
      (void)g_pool->release(handle);
      state.SkipWithError(layers.status().message().c_str());
      return;
    }

    sync_device();
    const auto t0 = Clock::now();
    StatusOr<Tensor> pref = [&]() -> StatusOr<Tensor> {
      AllocatorScope scope(g_alloc.get());
      return g_runner->prefill(prompt, layers.value());
    }();
    sync_device();
    if (!pref.ok()) {
      (void)g_pool->release(handle);
      state.SkipWithError(pref.status().message().c_str());
      return;
    }

    StatusOr<int32_t> tok = sample_token(pref.value());
    sync_device();
    const auto t1 = Clock::now();
    if (!tok.ok()) {
      (void)g_pool->release(handle);
      state.SkipWithError(tok.status().message().c_str());
      return;
    }

    const double ttft_s = elapsed_s(t0, t1);
    state.SetIterationTime(ttft_s);

    state.counters["ttft_ms"] = benchmark::Counter(ttft_s * 1e3);
    state.counters["prefill_tok_s"] =
        benchmark::Counter(static_cast<double>(prompt_len) / ttft_s);

    benchmark::DoNotOptimize(tok.value());

    state.PauseTiming();
    if (Status s = g_pool->release(handle); !s.ok()) {
      state.SkipWithError(s.message().c_str());
      return;
    }
    state.ResumeTiming();
  }

  state.SetLabel("prompt=" + std::to_string(prompt_len));
}

void BM_Generate(benchmark::State& state) {
  const int64_t prompt_len = state.range(0);
  const int64_t gen_len = state.range(1);
  if (prompt_len + gen_len > g_model_cfg.max_seq_length) {
    state.SkipWithError("prompt_len + gen_len exceeds model max_seq_length");
    return;
  }
  if (gen_len < 1) {
    state.SkipWithError("gen_len must be >= 1");
    return;
  }

  const auto prompt = make_prompt(prompt_len);

  for (auto _ : state) {
    std::vector<double> itls;
    itls.reserve(static_cast<size_t>(std::max<int64_t>(0, gen_len - 1)));

    StatusOr<CacheHandle> handle_or = g_pool->acquire();
    if (!handle_or.ok()) {
      state.SkipWithError(handle_or.status().message().c_str());
      return;
    }
    CacheHandle handle = std::move(handle_or.value());
    StatusOr<std::span<LayerCacheView>> layers = g_pool->layer_views(handle);
    if (!layers.ok()) {
      (void)g_pool->release(handle);
      state.SkipWithError(layers.status().message().c_str());
      return;
    }

    sync_device();
    const auto t_start = Clock::now();

    StatusOr<Tensor> pref = [&]() -> StatusOr<Tensor> {
      AllocatorScope scope(g_alloc.get());
      return g_runner->prefill(prompt, layers.value());
    }();
    sync_device();
    if (!pref.ok()) {
      (void)g_pool->release(handle);
      state.SkipWithError(pref.status().message().c_str());
      return;
    }

    StatusOr<int32_t> token = sample_token(pref.value());
    sync_device();
    const auto t_ttft = Clock::now();
    if (!token.ok()) {
      (void)g_pool->release(handle);
      state.SkipWithError(token.status().message().c_str());
      return;
    }

    const double ttft_s = elapsed_s(t_start, t_ttft);
    int32_t cur = token.value();
    double decode_s = 0.0;
    (void)g_pool->set_seq_len(handle, prompt_len);

    for (int64_t i = 1; i < gen_len; ++i) {
      const auto t0 = Clock::now();
      StatusOr<Tensor> dec = [&]() -> StatusOr<Tensor> {
        AllocatorScope scope(g_alloc.get());
        return g_runner->decode(cur, layers.value(), prompt_len + i - 1);
      }();
      sync_device();
      if (!dec.ok()) {
        (void)g_pool->release(handle);
        state.SkipWithError(dec.status().message().c_str());
        return;
      }
      StatusOr<int32_t> next = sample_token(dec.value());
      sync_device();
      const auto t1 = Clock::now();
      if (!next.ok()) {
        (void)g_pool->release(handle);
        state.SkipWithError(next.status().message().c_str());
        return;
      }
      const double step_s = elapsed_s(t0, t1);
      itls.push_back(step_s);
      decode_s += step_s;
      cur = next.value();
      (void)g_pool->set_seq_len(handle, prompt_len + i);
    }

    const double e2e_s = ttft_s + decode_s;
    state.SetIterationTime(e2e_s);

    state.counters["ttft_ms"] = benchmark::Counter(ttft_s * 1e3);
    state.counters["prefill_tok_s"] =
        benchmark::Counter(static_cast<double>(prompt_len) / ttft_s);
    state.counters["e2e_tok_s"] =
        benchmark::Counter(static_cast<double>(gen_len) / e2e_s);

    if (gen_len > 1 && decode_s > 0.0) {
      state.counters["decode_tok_s"] =
          benchmark::Counter(static_cast<double>(gen_len - 1) / decode_s);
      const double mean_itl_s = decode_s / static_cast<double>(gen_len - 1);
      state.counters["itl_mean_ms"] = benchmark::Counter(mean_itl_s * 1e3);
      state.counters["itl_p50_ms"] = benchmark::Counter(percentile_sorted_ms(itls, 50.0));
      state.counters["itl_p95_ms"] = benchmark::Counter(percentile_sorted_ms(itls, 95.0));
    }

    benchmark::DoNotOptimize(cur);

    state.PauseTiming();
    if (Status s = g_pool->release(handle); !s.ok()) {
      state.SkipWithError(s.message().c_str());
      return;
    }
    state.ResumeTiming();
  }

  state.SetLabel("prompt=" + std::to_string(prompt_len) +
                 " gen=" + std::to_string(gen_len));
}

void RegisterBenches() {
  for (int64_t p : kPromptLens) {
    if (p > g_model_cfg.max_seq_length)
      continue;
    benchmark::RegisterBenchmark("BM_Prefill", BM_Prefill)
        ->Arg(p)
        ->Unit(benchmark::kMillisecond)
        ->UseManualTime();
  }

  for (int64_t p : kPromptLens) {
    for (int64_t g : kGenLens) {
      if (p + g > g_model_cfg.max_seq_length)
        continue;
      benchmark::RegisterBenchmark("BM_Generate", BM_Generate)
          ->Args({p, g})
          ->Unit(benchmark::kMillisecond)
          ->UseManualTime();
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  int bench_argc = 0;
  char** bench_argv = nullptr;
  if (!parse_args(argc, argv, g_cfg, bench_argc, bench_argv)) {
    return 1;
  }

  if (Status s = load_model(g_cfg); !s.ok()) {
    std::cerr << "failed to load model: " << s.message() << '\n';
    return 1;
  }

  std::cout << "model_dir=" << g_cfg.model_dir
            << " device=" << (g_cfg.device == Device::GPU ? "GPU" : "CPU")
            << " model_type=" << g_model_cfg.model_type
            << " max_seq_length=" << g_model_cfg.max_seq_length << '\n';

  RegisterBenches();
  benchmark::Initialize(&bench_argc, bench_argv);
  if (benchmark::ReportUnrecognizedArguments(bench_argc, bench_argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
