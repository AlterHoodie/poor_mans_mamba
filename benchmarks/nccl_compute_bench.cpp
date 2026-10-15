// nccl_compute_bench: NCCL peer-xfer latency with GPUs idle vs SM-saturated by GEMM.
//
// Measures Recv-side completion of the same CommAgent NCCL path used for
// migration, once with no compute and once while both GPUs loop cublasSgemm on
// dedicated non-blocking streams (separate from NCCL's stream).
//
//   nccl_compute_bench --min-mb 1 --max-mb 256 --iters 20 --warmup 5 \
//                      --loads idle,gemm --modes uni --gemm-n 4096 \
//                      --out-dir results/nccl_compute
//
// Outputs (in --out-dir):
//   nccl_compute.csv      per (load, mode, size) latency / bandwidth summary
//   nccl_compute_raw.csv  one row per measured iteration
//   nccl_compute_meta.json

#include "bench_common.h"

#include "comm/comm_agent.h"
#include "comm/comm_factory.h"
#include "comm/nccl_transport_control.h"
#include "core/device.h"
#include "core/status.h"
#include "telemetry/log.h"
#include "telemetry/recorder.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

#if defined(MAMBASERVE_WITH_CUDA) && MAMBASERVE_WITH_CUDA
#include "ops/gpu/linear.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace {

using bench::Args;
using telemetry::steady_now_ns;

#if defined(MAMBASERVE_WITH_CUDA) && MAMBASERVE_WITH_CUDA

#define CUDA_OK(expr, what)                                                                        \
  do {                                                                                             \
    cudaError_t _e = (expr);                                                                       \
    if (_e != cudaSuccess) {                                                                       \
      std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(_e));                \
      return false;                                                                                \
    }                                                                                              \
  } while (0)

void* dev_alloc(int dev, size_t bytes) {
  if (cudaSetDevice(dev) != cudaSuccess)
    return nullptr;
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess)
    return nullptr;
  cudaMemset(p, 0, bytes);
  return p;
}

void dev_free(int dev, void* p) {
  if (!p)
    return;
  cudaSetDevice(dev);
  cudaFree(p);
}

bool dev_write(int dev, void* dst, const std::vector<uint8_t>& host) {
  CUDA_OK(cudaSetDevice(dev), "cudaSetDevice");
  CUDA_OK(cudaMemcpy(dst, host.data(), host.size(), cudaMemcpyHostToDevice), "H2D");
  return true;
}

bool dev_read(int dev, const void* src, std::vector<uint8_t>& host, size_t bytes) {
  host.resize(bytes);
  CUDA_OK(cudaSetDevice(dev), "cudaSetDevice");
  CUDA_OK(cudaMemcpy(host.data(), src, bytes, cudaMemcpyDeviceToHost), "D2H");
  return true;
}

// Continuous FP32 GEMM on a non-blocking stream (separate from NCCL's stream).
struct GemmLoad {
  int dev = 0;
  int n = 0; // square: M = N = K
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  float* A = nullptr;
  float* B = nullptr;
  float* C = nullptr;
  std::atomic<bool> stop{true};
  std::thread th;

  bool init(int device, int gemm_n) {
    dev = device;
    n = gemm_n;
    if (n <= 0)
      return false;
    CUDA_OK(cudaSetDevice(dev), "cudaSetDevice");
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "gemm stream");
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "cublasCreate failed on dev %d\n", dev);
      return false;
    }
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "cublasSetStream failed on dev %d\n", dev);
      return false;
    }
    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = elems * sizeof(float);
    CUDA_OK(cudaMalloc(&A, bytes), "gemm A");
    CUDA_OK(cudaMalloc(&B, bytes), "gemm B");
    CUDA_OK(cudaMalloc(&C, bytes), "gemm C");
    CUDA_OK(cudaMemset(A, 0, bytes), "memset A");
    CUDA_OK(cudaMemset(B, 0, bytes), "memset B");
    CUDA_OK(cudaMemset(C, 0, bytes), "memset C");
    return true;
  }

  void start() {
    stop.store(false, std::memory_order_release);
    th = std::thread([this] {
      cudaSetDevice(dev);
      // Batch a few launches then sync so the queue stays bounded while SMs
      // stay busy for the whole NCCL window.
      constexpr int kBatch = 4;
      while (!stop.load(std::memory_order_acquire)) {
        for (int i = 0; i < kBatch; ++i) {
          if (linear_f32(handle, A, B, C, n, n, n) != cudaSuccess) {
            std::fprintf(stderr, "gemm launch failed on dev %d\n", dev);
            stop.store(true, std::memory_order_release);
            break;
          }
        }
        cudaStreamSynchronize(stream);
      }
    });
  }

  void stop_join() {
    stop.store(true, std::memory_order_release);
    if (th.joinable())
      th.join();
    cudaSetDevice(dev);
    if (stream)
      cudaStreamSynchronize(stream);
  }

  void destroy() {
    stop_join();
    cudaSetDevice(dev);
    if (A)
      cudaFree(A);
    if (B)
      cudaFree(B);
    if (C)
      cudaFree(C);
    A = B = C = nullptr;
    if (handle) {
      cublasDestroy(handle);
      handle = nullptr;
    }
    if (stream) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }
};

// ---- endpoint: one thread per CommAgent (mirrors Worker's threading) -------

struct Endpoint {
  int dev = 0;
  std::unique_ptr<CommAgent> agent;
  Endpoint* peer = nullptr;
  void* slab = nullptr;
  size_t slab_bytes = 0;

  std::vector<XferDesc> descs;
  std::atomic<bool> has_job{false};
  std::atomic<bool> ready{false};
  std::atomic<bool>* go = nullptr;
  std::atomic<bool> finished{false};
  std::atomic<bool> stop{false};
  bool ok = true;
  int64_t t_start_ns = 0;
  int64_t t_end_ns = 0;
  int64_t t_recv_done_ns = 0;

  std::mutex mu;
  std::condition_variable cv;
  std::thread th;

  void start() { th = std::thread([this] { loop(); }); }

  void submit(std::vector<XferDesc> d, std::atomic<bool>* go_flag) {
    {
      std::lock_guard<std::mutex> lk(mu);
      descs = std::move(d);
      go = go_flag;
      finished = false;
      ready = false;
      has_job = true;
    }
    cv.notify_one();
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv.notify_one();
    if (th.joinable())
      th.join();
  }

private:
  void loop() {
    cudaSetDevice(dev);
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return has_job.load() || stop.load(); });
        if (stop && !has_job)
          return;
      }
      ready = true;
      while (!go->load(std::memory_order_acquire))
        std::this_thread::yield();

      ok = true;
      t_recv_done_ns = 0;
      t_start_ns = steady_now_ns();
      std::vector<XferHandle> handles;
      std::vector<bool> done;
      handles.reserve(descs.size());
      done.reserve(descs.size());
      size_t remaining = 0;
      for (XferDesc& d : descs) {
        StatusOr<PostResult> h = agent->post(d);
        if (!h.ok()) {
          std::fprintf(stderr, "post failed: %s\n", h.status().message().c_str());
          ok = false;
          break;
        }
        if (h.value().state == XferState::Error) {
          std::fprintf(stderr, "xfer error at post on dev %d\n", dev);
          ok = false;
          break;
        }
        if (h.value().state == XferState::Done) {
          handles.push_back(h.value().handle);
          done.push_back(true);
          if (d.role == XferRole::Recv)
            t_recv_done_ns = steady_now_ns();
          continue;
        }
        handles.push_back(h.value().handle);
        done.push_back(false);
        ++remaining;
      }
      const int64_t deadline = steady_now_ns() + 60'000'000'000LL;
      while (remaining > 0 && ok) {
        for (size_t i = 0; i < handles.size(); ++i) {
          if (done[i])
            continue;
          const XferState st = agent->poll(handles[i]);
          if (st == XferState::Pending)
            continue;
          if (st == XferState::Error) {
            std::fprintf(stderr, "xfer error on dev %d\n", dev);
            ok = false;
            break;
          }
          done[i] = true;
          --remaining;
          if (descs[i].role == XferRole::Recv)
            t_recv_done_ns = steady_now_ns();
        }
        if (steady_now_ns() > deadline) {
          std::fprintf(stderr, "xfer timeout on dev %d\n", dev);
          ok = false;
        }
      }
      t_end_ns = steady_now_ns();
      {
        std::lock_guard<std::mutex> lk(mu);
        has_job = false;
      }
      finished = true;
    }
  }
};

struct Pair {
  Endpoint ep[2];
  std::shared_ptr<TransportContext> ctx;
};

bool setup_pair(Pair& p, size_t slab_bytes) {
  ClusterConfig cfg;
  cfg.model_dir = "nccl_compute_bench";
  cfg.device = Device::GPU;
  cfg.n_workers = 2;
  cfg.num_slots = 1;
  cfg.transport = TransportBackend::Nccl;

  for (int i = 0; i < 2; ++i) {
    p.ep[i].dev = i;
    p.ep[i].slab_bytes = slab_bytes;
    p.ep[i].slab = dev_alloc(i, slab_bytes);
    if (!p.ep[i].slab) {
      std::fprintf(stderr, "failed to allocate %zu bytes on dev %d\n", slab_bytes, i);
      return false;
    }
  }

  auto ctx_or = create_transport_context(cfg);
  if (!ctx_or.ok()) {
    std::fprintf(stderr, "create_transport_context: %s\n", ctx_or.status().message().c_str());
    return false;
  }
  p.ctx = ctx_or.value();

  std::atomic<bool> fail{false};
  std::thread threads[2];
  for (int i = 0; i < 2; ++i) {
    threads[i] = std::thread([&, i] {
      cudaSetDevice(i);
      auto a = create_comm_agent(cfg, i, p.ctx);
      if (!a.ok()) {
        std::fprintf(stderr, "create_comm_agent(%d): %s\n", i, a.status().message().c_str());
        fail = true;
        return;
      }
      p.ep[i].agent = std::move(a.value());
    });
  }
  for (auto& th : threads)
    th.join();
  if (fail)
    return false;

#if defined(MAMBASERVE_WITH_NCCL) && MAMBASERVE_WITH_NCCL
  {
    auto plane_or = NcclTransportControl::create(cfg.n_workers);
    if (!plane_or.ok()) {
      std::fprintf(stderr, "NcclTransportControl::create: %s\n",
                   plane_or.status().message().c_str());
      return false;
    }
    NcclTransportControl& plane = *plane_or.value();
    std::thread boots[2];
    for (int i = 0; i < 2; ++i) {
      boots[i] = std::thread([&, i] {
        cudaSetDevice(i);
        Status s = p.ep[i].agent->handle_transport(plane.make_bootstrap(i));
        if (!s.ok()) {
          std::fprintf(stderr, "nccl bootstrap(%d): %s\n", i, s.message().c_str());
          fail = true;
        }
        (void)p.ep[i].agent->take_transport_reply();
      });
    }
    for (auto& th : boots)
      th.join();
    if (fail)
      return false;
  }
#else
  std::fprintf(stderr, "built without MAMBASERVE_WITH_NCCL\n");
  return false;
#endif

  for (int i = 0; i < 2; ++i) {
    Status s = p.ep[i].agent->register_slab(p.ep[i].slab, p.ep[i].slab_bytes);
    if (!s.ok()) {
      std::fprintf(stderr, "register_slab(%d): %s\n", i, s.message().c_str());
      return false;
    }
  }
  if (Status s = finalize_transport_peers(p.ctx); !s.ok()) {
    std::fprintf(stderr, "finalize_transport_peers: %s\n", s.message().c_str());
    return false;
  }

  p.ep[0].peer = &p.ep[1];
  p.ep[1].peer = &p.ep[0];
  for (int i = 0; i < 2; ++i)
    p.ep[i].start();
  return true;
}

void teardown_pair(Pair& p) {
  for (int i = 0; i < 2; ++i)
    p.ep[i].shutdown();
  for (int i = 0; i < 2; ++i)
    p.ep[i].agent.reset();
  for (int i = 0; i < 2; ++i) {
    dev_free(i, p.ep[i].slab);
    p.ep[i].slab = nullptr;
  }
  p.ctx.reset();
}

struct IterResult {
  bool ok = false;
  double latency_us = 0;
  double recv_latency_us = 0;
};

uint64_t g_next_xfer_id = 1;

IterResult run_iter(Pair& p, size_t bytes, size_t region_stride, bool bidir) {
  IterResult r;
  const uint64_t fwd = g_next_xfer_id++;
  const uint64_t rev = g_next_xfer_id++;

  char* s0 = static_cast<char*>(p.ep[0].slab);
  char* s1 = static_cast<char*>(p.ep[1].slab);

  std::vector<XferDesc> d0, d1;
  d0.push_back(XferDesc{.local_ptr = s0,
                        .bytes = bytes,
                        .peer_device_id = 1,
                        .role = XferRole::Send,
                        .xfer_id = fwd,
                        .seq_len = 1});
  d1.push_back(XferDesc{.local_ptr = s1,
                        .remote_ptr = s0,
                        .bytes = bytes,
                        .peer_device_id = 0,
                        .role = XferRole::Recv,
                        .xfer_id = fwd,
                        .seq_len = 1});
  if (bidir) {
    d0.push_back(XferDesc{.local_ptr = s0 + region_stride,
                          .remote_ptr = s1 + region_stride,
                          .bytes = bytes,
                          .peer_device_id = 1,
                          .role = XferRole::Recv,
                          .xfer_id = rev,
                          .seq_len = 1});
    d1.push_back(XferDesc{.local_ptr = s1 + region_stride,
                          .bytes = bytes,
                          .peer_device_id = 0,
                          .role = XferRole::Send,
                          .xfer_id = rev,
                          .seq_len = 1});
  }

  std::atomic<bool> go{false};
  p.ep[0].submit(std::move(d0), &go);
  p.ep[1].submit(std::move(d1), &go);
  while (!p.ep[0].ready || !p.ep[1].ready)
    std::this_thread::yield();
  go.store(true, std::memory_order_release);
  while (!p.ep[0].finished || !p.ep[1].finished)
    std::this_thread::yield();

  if (!p.ep[0].ok || !p.ep[1].ok)
    return r;
  const int64_t start = std::min(p.ep[0].t_start_ns, p.ep[1].t_start_ns);
  const int64_t end = std::max(p.ep[0].t_end_ns, p.ep[1].t_end_ns);
  const int64_t recv_end = std::max(p.ep[0].t_recv_done_ns, p.ep[1].t_recv_done_ns);
  r.ok = true;
  r.latency_us = static_cast<double>(end - start) / 1e3;
  r.recv_latency_us = static_cast<double>(recv_end - start) / 1e3;
  return r;
}

bool verify(Pair& p, size_t bytes, size_t region_stride, bool bidir) {
  std::vector<uint8_t> got;
  std::vector<uint8_t> expect(bytes);
  for (size_t i = 0; i < bytes; ++i)
    expect[i] = static_cast<uint8_t>((i * 131 + 7) % 251);
  if (!dev_read(1, p.ep[1].slab, got, bytes) || got != expect)
    return false;
  if (bidir) {
    for (size_t i = 0; i < bytes; ++i)
      expect[i] = static_cast<uint8_t>((i * 17 + 3) % 241);
    if (!dev_read(0, static_cast<char*>(p.ep[0].slab) + region_stride, got, bytes) ||
        got != expect)
      return false;
  }
  return true;
}

bool prime_buffers(Pair& p, size_t bytes, size_t region_stride, bool bidir) {
  std::vector<uint8_t> src(bytes), zero(bytes, 0);
  for (size_t i = 0; i < bytes; ++i)
    src[i] = static_cast<uint8_t>((i * 131 + 7) % 251);
  if (!dev_write(0, p.ep[0].slab, src) || !dev_write(1, p.ep[1].slab, zero))
    return false;
  if (bidir) {
    for (size_t i = 0; i < bytes; ++i)
      src[i] = static_cast<uint8_t>((i * 17 + 3) % 241);
    if (!dev_write(1, static_cast<char*>(p.ep[1].slab) + region_stride, src) ||
        !dev_write(0, static_cast<char*>(p.ep[0].slab) + region_stride, zero))
      return false;
  }
  return true;
}

#endif // MAMBASERVE_WITH_CUDA

} // namespace

int main(int argc, char** argv) {
  Args args(argc, argv);
  if (args.has("help")) {
    std::puts("nccl_compute_bench --min-mb 1 --max-mb 256 --iters 20 --warmup 5\n"
              "  --loads idle,gemm --modes uni,bidir --gemm-n 4096 --out-dir DIR");
    return 0;
  }

#if !(defined(MAMBASERVE_WITH_CUDA) && MAMBASERVE_WITH_CUDA)
  std::fprintf(stderr, "nccl_compute_bench requires MAMBASERVE_WITH_CUDA=ON\n");
  return 1;
#elif !(defined(MAMBASERVE_WITH_NCCL) && MAMBASERVE_WITH_NCCL)
  std::fprintf(stderr, "nccl_compute_bench requires MAMBASERVE_WITH_NCCL=ON\n");
  return 1;
#else
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 2) {
    std::fprintf(stderr, "nccl_compute_bench needs >= 2 CUDA devices (found %d)\n", n_dev);
    return 1;
  }

  const auto min_mb = args.f64("min-mb", 1.0);
  const auto max_mb = args.f64("max-mb", 256.0);
  const int iters = static_cast<int>(args.i64("iters", 20));
  const int warmup = static_cast<int>(args.i64("warmup", 5));
  const int gemm_n = static_cast<int>(args.i64("gemm-n", 4096));
  const std::string out_dir =
      args.str("out-dir", "results/nccl_compute_" + bench::timestamp_str());
  const auto modes = args.str_list("modes", {"uni"});
  const auto loads = args.str_list("loads", {"idle", "gemm"});
  bench::make_dirs(out_dir);

  std::vector<size_t> sizes;
  for (double mb = min_mb; mb <= max_mb * 1.0001; mb *= 2.0)
    sizes.push_back(static_cast<size_t>(mb * 1024.0 * 1024.0));
  if (sizes.empty()) {
    std::fprintf(stderr, "no sizes to run\n");
    return 1;
  }
  const size_t max_bytes = sizes.back();
  const size_t region_stride = max_bytes;

  std::ofstream summary(out_dir + "/nccl_compute.csv");
  std::ofstream raw(out_dir + "/nccl_compute_raw.csv");
  summary << "backend,load,mode,bytes,iters,verified,gemm_n,"
             "lat_us_median,lat_us_mean,lat_us_std,lat_us_min,lat_us_p95,"
             "recv_lat_us_median,recv_lat_us_mean,recv_lat_us_p95,"
             "gbps_median,gbps_best\n";
  raw << "backend,load,mode,bytes,iter,latency_us,recv_latency_us\n";

  telemetry::MetaKV meta = bench::common_meta("nccl_compute_bench", args);
  meta.emplace_back("gemm_n", std::to_string(gemm_n));
  meta.emplace_back("sizes_bytes", [&] {
    std::string s;
    for (size_t b : sizes)
      s += std::to_string(b) + ",";
    return s;
  }());
  if (Status s = telemetry::write_meta_json(out_dir + "/nccl_compute_meta.json", meta); !s.ok())
    std::fprintf(stderr, "warning: %s\n", s.message().c_str());

  std::fprintf(stderr, "[nccl] setting up pair...\n");
  Pair pair;
  if (!setup_pair(pair, 2 * region_stride)) {
    std::fprintf(stderr, "[nccl] setup failed\n");
    teardown_pair(pair);
    return 2;
  }

  GemmLoad gemm[2];
  bool need_gemm = false;
  for (const std::string& load : loads) {
    if (load == "gemm")
      need_gemm = true;
  }
  if (need_gemm) {
    for (int i = 0; i < 2; ++i) {
      if (!gemm[i].init(i, gemm_n)) {
        std::fprintf(stderr, "[gemm] init failed on dev %d (n=%d)\n", i, gemm_n);
        for (int j = 0; j < 2; ++j)
          gemm[j].destroy();
        teardown_pair(pair);
        return 2;
      }
    }
    std::fprintf(stderr, "[gemm] ready n=%d on both devices\n", gemm_n);
  }

  int failures = 0;
  for (const std::string& load : loads) {
    const bool use_gemm = (load == "gemm");
    if (load != "idle" && load != "gemm") {
      std::fprintf(stderr, "unknown load '%s' (want idle|gemm)\n", load.c_str());
      ++failures;
      continue;
    }

    for (const std::string& mode : modes) {
      const bool bidir = (mode == "bidir");
      for (size_t bytes : sizes) {
        if (!prime_buffers(pair, bytes, region_stride, bidir)) {
          std::fprintf(stderr, "[%s/%s] prime failed\n", load.c_str(), mode.c_str());
          ++failures;
          break;
        }

        auto run_timed = [&](bool measure, int iter_idx, std::vector<double>* lat,
                             std::vector<double>* rlat) -> bool {
          if (use_gemm) {
            gemm[0].start();
            gemm[1].start();
            // Let a few GEMMs land before posting NCCL.
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
          }
          IterResult r = run_iter(pair, bytes, region_stride, bidir);
          if (use_gemm) {
            gemm[0].stop_join();
            gemm[1].stop_join();
          }
          if (!r.ok)
            return false;
          if (measure && lat && rlat) {
            lat->push_back(r.latency_us);
            rlat->push_back(r.recv_latency_us);
            raw << "nccl," << load << ',' << mode << ',' << bytes << ',' << iter_idx << ','
                << r.latency_us << ',' << r.recv_latency_us << '\n';
          }
          return true;
        };

        bool verified = false;
        bool ok = true;
        for (int i = 0; i < warmup && ok; ++i) {
          ok = run_timed(false, i, nullptr, nullptr);
          if (i == 0 && ok)
            verified = verify(pair, bytes, region_stride, bidir);
        }
        if (!ok) {
          std::fprintf(stderr, "[%s/%s/%zu] warmup failed\n", load.c_str(), mode.c_str(), bytes);
          ++failures;
          break;
        }
        if (!verified)
          std::fprintf(stderr, "[%s/%s/%zu] WARNING: data verification failed\n", load.c_str(),
                       mode.c_str(), bytes);

        std::vector<double> lat, rlat;
        for (int i = 0; i < iters; ++i) {
          if (!run_timed(true, i, &lat, &rlat)) {
            ok = false;
            break;
          }
        }
        if (!ok || lat.empty()) {
          std::fprintf(stderr, "[%s/%s/%zu] measured iteration failed\n", load.c_str(),
                       mode.c_str(), bytes);
          ++failures;
          break;
        }

        const double total_bytes = static_cast<double>(bytes) * (bidir ? 2 : 1);
        const double med = bench::percentile(lat, 50);
        const double best = *std::min_element(lat.begin(), lat.end());
        const double rmed = bench::percentile(rlat, 50);
        summary << "nccl," << load << ',' << mode << ',' << bytes << ',' << iters << ','
                << (verified ? 1 : 0) << ',' << gemm_n << ',' << med << ',' << bench::mean(lat)
                << ',' << bench::stddev(lat) << ',' << best << ',' << bench::percentile(lat, 95)
                << ',' << rmed << ',' << bench::mean(rlat) << ',' << bench::percentile(rlat, 95)
                << ',' << total_bytes / (rmed * 1e3) << ',' << total_bytes / (best * 1e3) << '\n';
        summary.flush();
        std::fprintf(stderr,
                     "[nccl/%s/%s] %8.1f MB  recv_p50 %10.1f us  %7.2f GB/s%s\n", load.c_str(),
                     mode.c_str(), static_cast<double>(bytes) / (1024.0 * 1024.0), rmed,
                     total_bytes / (rmed * 1e3), verified ? "" : "  (UNVERIFIED)");
      }
    }
  }

  for (int i = 0; i < 2; ++i)
    gemm[i].destroy();
  teardown_pair(pair);

  std::fprintf(stderr, "wrote %s/{nccl_compute.csv,nccl_compute_raw.csv,nccl_compute_meta.json}\n",
               out_dir.c_str());
  return failures == 0 ? 0 : 2;
#endif
}
