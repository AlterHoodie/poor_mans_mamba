// transport_bench: model-free microbenchmark of the CommAgent backends.
//
// Moves a buffer between "worker 0" and "worker 1" through the same
// CommAgent interface the Worker uses (Recv posts the destination, Send
// posts the source, completion is observed on the Recv side because NCCL and
// NIXL Send agents ack at post time).
//
//   transport_bench --device GPU --backends memcpy,nccl,nixl --min-mb 1 --max-mb 256 \
//                   --iters 20 --warmup 5 --modes uni,bidir --out-dir results/transport
//
// Outputs (in --out-dir):
//   transport.csv        per (backend, mode, size) latency / bandwidth summary
//   transport_raw.csv    one row per measured iteration
//   transport_setup.csv  one-time setup costs per backend (agent init, register, finalize)
//   transport_meta.json  run metadata (topology, versions, env, args)
//
// The "cuda_p2p_raw" pseudo-backend is a plain cudaMemcpyPeerAsync baseline.

#include "bench_common.h"

#include "comm/comm_agent.h"
#include "comm/comm_factory.h"
#include "core/device.h"
#include "core/status.h"
#include "telemetry/log.h"
#include "telemetry/recorder.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

#ifdef MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

using bench::Args;
using telemetry::steady_now_ns;

#define CUDA_OK(expr, what)                                                                        \
  do {                                                                                             \
    cudaError_t _e = (expr);                                                                       \
    if (_e != cudaSuccess) {                                                                       \
      std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(_e));                \
      return false;                                                                                \
    }                                                                                              \
  } while (0)

// ---- device memory helpers -------------------------------------------------

void* dev_alloc(Device kind, int dev, size_t bytes) {
  if (kind == Device::CPU) {
    void* p = nullptr;
    if (posix_memalign(&p, 4096, bytes) != 0)
      return nullptr;
    std::memset(p, 0, bytes);
    return p;
  }
#ifdef MAMBASERVE_WITH_CUDA
  if (cudaSetDevice(dev) != cudaSuccess)
    return nullptr;
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess)
    return nullptr;
  cudaMemset(p, 0, bytes);
  return p;
#else
  (void)dev;
  return nullptr;
#endif
}

void dev_free(Device kind, int dev, void* p) {
  if (!p)
    return;
  if (kind == Device::CPU) {
    std::free(p);
    return;
  }
#ifdef MAMBASERVE_WITH_CUDA
  cudaSetDevice(dev);
  cudaFree(p);
#else
  (void)dev;
#endif
}

bool dev_write(Device kind, int dev, void* dst, const std::vector<uint8_t>& host) {
  if (kind == Device::CPU) {
    std::memcpy(dst, host.data(), host.size());
    return true;
  }
#ifdef MAMBASERVE_WITH_CUDA
  CUDA_OK(cudaSetDevice(dev), "cudaSetDevice");
  CUDA_OK(cudaMemcpy(dst, host.data(), host.size(), cudaMemcpyHostToDevice), "H2D");
  return true;
#else
  (void)dev;
  return false;
#endif
}

bool dev_read(Device kind, int dev, const void* src, std::vector<uint8_t>& host, size_t bytes) {
  host.resize(bytes);
  if (kind == Device::CPU) {
    std::memcpy(host.data(), src, bytes);
    return true;
  }
#ifdef MAMBASERVE_WITH_CUDA
  CUDA_OK(cudaSetDevice(dev), "cudaSetDevice");
  CUDA_OK(cudaMemcpy(host.data(), src, bytes, cudaMemcpyDeviceToHost), "D2H");
  return true;
#else
  (void)dev;
  return false;
#endif
}

// ---- endpoint: one thread per CommAgent (mirrors Worker's threading) -------

struct Endpoint {
  int dev = 0;
  Device kind = Device::GPU;
  std::unique_ptr<CommAgent> agent;
  void* slab = nullptr;
  size_t slab_bytes = 0;

  // job
  std::vector<XferDesc> descs;
  std::atomic<bool> has_job{false};
  std::atomic<bool> ready{false};
  std::atomic<bool>* go = nullptr;
  std::atomic<bool> finished{false};
  std::atomic<bool> stop{false};
  bool ok = true;
  int64_t t_start_ns = 0;
  int64_t t_end_ns = 0;
  int64_t t_recv_done_ns = 0; // when the last Recv desc on this endpoint completed

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
#ifdef MAMBASERVE_WITH_CUDA
    if (kind == Device::GPU)
      cudaSetDevice(dev);
#endif
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
      handles.reserve(descs.size());
      for (XferDesc& d : descs) {
        StatusOr<XferHandle> h = agent->post(d);
        if (!h.ok()) {
          std::fprintf(stderr, "post failed: %s\n", h.status().message().c_str());
          ok = false;
          break;
        }
        handles.push_back(h.value());
      }
      std::vector<bool> done(handles.size(), false);
      size_t remaining = ok ? handles.size() : 0;
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

struct SetupTimes {
  double create_agents_us = 0;
  double register_slab_us = 0;
  double finalize_us = 0;
};

// Builds both endpoints. Agents are created on separate threads because NCCL's
// communicator init blocks until every rank joins.
bool setup_pair(Pair& p, TransportBackend backend, Device kind, size_t slab_bytes,
                SetupTimes& t) {
  ClusterConfig cfg;
  cfg.model_dir = "transport_bench";
  cfg.device = kind;
  cfg.n_workers = 2;
  cfg.num_slots = 1;
  cfg.transport = backend;

  for (int i = 0; i < 2; ++i) {
    p.ep[i].dev = i;
    p.ep[i].kind = kind;
    p.ep[i].slab_bytes = slab_bytes;
    p.ep[i].slab = dev_alloc(kind, i, slab_bytes);
    if (!p.ep[i].slab) {
      std::fprintf(stderr, "failed to allocate %zu bytes on dev %d\n", slab_bytes, i);
      return false;
    }
  }

  int64_t t0 = steady_now_ns();
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
#ifdef MAMBASERVE_WITH_CUDA
      if (kind == Device::GPU)
        cudaSetDevice(i);
#endif
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
  t.create_agents_us = static_cast<double>(steady_now_ns() - t0) / 1e3;

  t0 = steady_now_ns();
  for (int i = 0; i < 2; ++i) {
    Status s = p.ep[i].agent->register_slab(p.ep[i].slab, p.ep[i].slab_bytes);
    if (!s.ok()) {
      std::fprintf(stderr, "register_slab(%d): %s\n", i, s.message().c_str());
      return false;
    }
  }
  t.register_slab_us = static_cast<double>(steady_now_ns() - t0) / 1e3;

  t0 = steady_now_ns();
  if (Status s = finalize_transport_peers(p.ctx); !s.ok()) {
    std::fprintf(stderr, "finalize_transport_peers: %s\n", s.message().c_str());
    return false;
  }
  t.finalize_us = static_cast<double>(steady_now_ns() - t0) / 1e3;

  for (int i = 0; i < 2; ++i)
    p.ep[i].start();
  return true;
}

void teardown_pair(Pair& p, Device kind) {
  for (int i = 0; i < 2; ++i)
    p.ep[i].shutdown();
  for (int i = 0; i < 2; ++i)
    p.ep[i].agent.reset();
  for (int i = 0; i < 2; ++i) {
    dev_free(kind, i, p.ep[i].slab);
    p.ep[i].slab = nullptr;
  }
  p.ctx.reset();
}

struct IterResult {
  bool ok = false;
  double latency_us = 0;      // first post -> last completion on either endpoint
  double recv_latency_us = 0; // first post -> last Recv completion
};

uint64_t g_next_xfer_id = 1;

// One iteration. Region 0 of each slab is the forward buffer (0 -> 1); region 1
// is the reverse buffer (1 -> 0). Post order is fixed (dev0: send then recv;
// dev1: recv then send) so a single in-order NCCL stream per rank cannot
// deadlock.
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
                        .bytes = bytes,
                        .peer_device_id = 0,
                        .role = XferRole::Recv,
                        .xfer_id = fwd});
  if (bidir) {
    d0.push_back(XferDesc{.local_ptr = s0 + region_stride,
                          .bytes = bytes,
                          .peer_device_id = 1,
                          .role = XferRole::Recv,
                          .xfer_id = rev});
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

bool verify(Pair& p, Device kind, size_t bytes, size_t region_stride, bool bidir) {
  std::vector<uint8_t> got;
  std::vector<uint8_t> expect(bytes);
  for (size_t i = 0; i < bytes; ++i)
    expect[i] = static_cast<uint8_t>((i * 131 + 7) % 251);
  if (!dev_read(kind, 1, p.ep[1].slab, got, bytes) || got != expect)
    return false;
  if (bidir) {
    for (size_t i = 0; i < bytes; ++i)
      expect[i] = static_cast<uint8_t>((i * 17 + 3) % 241);
    if (!dev_read(kind, 0, static_cast<char*>(p.ep[0].slab) + region_stride, got, bytes) ||
        got != expect)
      return false;
  }
  return true;
}

// Fill sources with patterns and clear destinations.
bool prime_buffers(Pair& p, Device kind, size_t bytes, size_t region_stride, bool bidir) {
  std::vector<uint8_t> src(bytes), zero(bytes, 0);
  for (size_t i = 0; i < bytes; ++i)
    src[i] = static_cast<uint8_t>((i * 131 + 7) % 251);
  if (!dev_write(kind, 0, p.ep[0].slab, src) || !dev_write(kind, 1, p.ep[1].slab, zero))
    return false;
  if (bidir) {
    for (size_t i = 0; i < bytes; ++i)
      src[i] = static_cast<uint8_t>((i * 17 + 3) % 241);
    if (!dev_write(kind, 1, static_cast<char*>(p.ep[1].slab) + region_stride, src) ||
        !dev_write(kind, 0, static_cast<char*>(p.ep[0].slab) + region_stride, zero))
      return false;
  }
  return true;
}

// ---- raw cudaMemcpyPeerAsync baseline -------------------------------------

#ifdef MAMBASERVE_WITH_CUDA
struct RawP2P {
  void* buf[2] = {nullptr, nullptr};
  cudaStream_t stream[2] = {nullptr, nullptr};
  cudaEvent_t start[2] = {nullptr, nullptr};
  cudaEvent_t stop[2] = {nullptr, nullptr};
  bool peer_access = false;
};

bool raw_setup(RawP2P& r, size_t bytes) {
  int n = 0;
  CUDA_OK(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
  if (n < 2) {
    std::fprintf(stderr, "cuda_p2p_raw needs 2 GPUs (found %d)\n", n);
    return false;
  }
  int can01 = 0, can10 = 0;
  cudaDeviceCanAccessPeer(&can01, 0, 1);
  cudaDeviceCanAccessPeer(&can10, 1, 0);
  r.peer_access = can01 && can10;
  for (int i = 0; i < 2; ++i) {
    CUDA_OK(cudaSetDevice(i), "cudaSetDevice");
    if (r.peer_access) {
      cudaError_t e = cudaDeviceEnablePeerAccess(1 - i, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
        return false;
      (void)cudaGetLastError();
    }
    CUDA_OK(cudaMalloc(&r.buf[i], bytes), "cudaMalloc");
    CUDA_OK(cudaStreamCreateWithFlags(&r.stream[i], cudaStreamNonBlocking), "stream");
    CUDA_OK(cudaEventCreate(&r.start[i]), "event");
    CUDA_OK(cudaEventCreate(&r.stop[i]), "event");
  }
  return true;
}

void raw_teardown(RawP2P& r) {
  for (int i = 0; i < 2; ++i) {
    cudaSetDevice(i);
    if (r.buf[i])
      cudaFree(r.buf[i]);
    if (r.stream[i])
      cudaStreamDestroy(r.stream[i]);
    if (r.start[i])
      cudaEventDestroy(r.start[i]);
    if (r.stop[i])
      cudaEventDestroy(r.stop[i]);
  }
}

// Returns wall latency in us (host timed around both streams' completion).
double raw_iter(RawP2P& r, size_t bytes, bool bidir) {
  const int64_t t0 = steady_now_ns();
  cudaMemcpyPeerAsync(r.buf[1], 1, r.buf[0], 0, bytes, r.stream[0]);
  if (bidir)
    cudaMemcpyPeerAsync(r.buf[0], 0, r.buf[1], 1, bytes, r.stream[1]);
  cudaStreamSynchronize(r.stream[0]);
  if (bidir)
    cudaStreamSynchronize(r.stream[1]);
  return static_cast<double>(steady_now_ns() - t0) / 1e3;
}
#endif

} // namespace

int main(int argc, char** argv) {
  Args args(argc, argv);
  if (args.has("help")) {
    std::puts("transport_bench --device GPU|CPU --backends memcpy,nccl,nixl,cuda_p2p_raw\n"
              "  --min-mb 1 --max-mb 256 --iters 20 --warmup 5 --modes uni,bidir\n"
              "  --out-dir DIR");
    return 0;
  }

  const Device kind = bench::parse_device(args.str("device", "GPU")).value_or(Device::GPU);
  const auto min_mb = args.f64("min-mb", 1.0);
  const auto max_mb = args.f64("max-mb", 256.0);
  const int iters = static_cast<int>(args.i64("iters", 20));
  const int warmup = static_cast<int>(args.i64("warmup", 5));
  const std::string out_dir = args.str("out-dir", "results/transport_" + bench::timestamp_str());
  const auto modes = args.str_list("modes", {"uni", "bidir"});
  const auto backends = args.str_list("backends", {"memcpy", "nccl", "nixl", "cuda_p2p_raw"});
  bench::make_dirs(out_dir);

  std::vector<size_t> sizes;
  for (double mb = min_mb; mb <= max_mb * 1.0001; mb *= 2.0)
    sizes.push_back(static_cast<size_t>(mb * 1024.0 * 1024.0));
  if (sizes.empty()) {
    std::fprintf(stderr, "no sizes to run\n");
    return 1;
  }
  const size_t max_bytes = sizes.back();

  std::ofstream summary(out_dir + "/transport.csv");
  std::ofstream raw(out_dir + "/transport_raw.csv");
  std::ofstream setup_csv(out_dir + "/transport_setup.csv");
  summary << "backend,device,mode,bytes,iters,verified,lat_us_median,lat_us_mean,lat_us_std,"
             "lat_us_min,lat_us_p95,recv_lat_us_median,gbps_median,gbps_best\n";
  raw << "backend,device,mode,bytes,iter,latency_us,recv_latency_us\n";
  setup_csv << "backend,device,phase,us\n";

  telemetry::MetaKV meta = bench::common_meta("transport_bench", args);
  meta.emplace_back("sizes_bytes", [&] {
    std::string s;
    for (size_t b : sizes)
      s += std::to_string(b) + ",";
    return s;
  }());
  if (Status s = telemetry::write_meta_json(out_dir + "/transport_meta.json", meta); !s.ok())
    std::fprintf(stderr, "warning: %s\n", s.message().c_str());

  int failures = 0;

  for (const std::string& bname : backends) {
    // ---- raw CUDA baseline ----
    if (bname == "cuda_p2p_raw") {
#ifdef MAMBASERVE_WITH_CUDA
      if (kind != Device::GPU)
        continue;
      RawP2P r;
      if (!raw_setup(r, max_bytes)) {
        std::fprintf(stderr, "[cuda_p2p_raw] setup failed, skipping\n");
        ++failures;
        raw_teardown(r);
        continue;
      }
      std::fprintf(stderr, "[cuda_p2p_raw] peer_access=%d\n", r.peer_access ? 1 : 0);
      for (const std::string& mode : modes) {
        const bool bidir = (mode == "bidir");
        for (size_t bytes : sizes) {
          for (int i = 0; i < warmup; ++i)
            raw_iter(r, bytes, bidir);
          std::vector<double> lat;
          for (int i = 0; i < iters; ++i) {
            const double us = raw_iter(r, bytes, bidir);
            lat.push_back(us);
            raw << "cuda_p2p_raw,GPU," << mode << ',' << bytes << ',' << i << ',' << us << ",\n";
          }
          const double total_bytes = static_cast<double>(bytes) * (bidir ? 2 : 1);
          const double med = bench::percentile(lat, 50);
          const double best = *std::min_element(lat.begin(), lat.end());
          summary << "cuda_p2p_raw,GPU," << mode << ',' << bytes << ',' << iters << ",na," << med
                  << ',' << bench::mean(lat) << ',' << bench::stddev(lat) << ',' << best << ','
                  << bench::percentile(lat, 95) << ",," << total_bytes / (med * 1e3) << ','
                  << total_bytes / (best * 1e3) << '\n';
        }
      }
      raw_teardown(r);
#endif
      continue;
    }

    // ---- CommAgent backends ----
    const auto backend = bench::parse_backend(bname);
    if (!backend) {
      std::fprintf(stderr, "unknown backend '%s'\n", bname.c_str());
      ++failures;
      continue;
    }
    std::fprintf(stderr, "[%s] setting up...\n", bname.c_str());
    Pair pair;
    SetupTimes st;
    // Two regions per slab so bidirectional runs use disjoint buffers.
    const size_t region_stride = max_bytes;
    if (!setup_pair(pair, *backend, kind, 2 * region_stride, st)) {
      std::fprintf(stderr, "[%s] setup failed (backend not built/available?), skipping\n",
                   bname.c_str());
      ++failures;
      teardown_pair(pair, kind);
      continue;
    }
    setup_csv << bname << ',' << bench::device_name(kind) << ",create_agents," << st.create_agents_us
              << '\n'
              << bname << ',' << bench::device_name(kind) << ",register_slab," << st.register_slab_us
              << '\n'
              << bname << ',' << bench::device_name(kind) << ",finalize_peers," << st.finalize_us
              << '\n';
    setup_csv.flush();

    for (const std::string& mode : modes) {
      const bool bidir = (mode == "bidir");
      for (size_t bytes : sizes) {
        if (!prime_buffers(pair, kind, bytes, region_stride, bidir)) {
          std::fprintf(stderr, "[%s] prime failed\n", bname.c_str());
          ++failures;
          break;
        }
        bool verified = false;
        bool ok = true;
        for (int i = 0; i < warmup && ok; ++i) {
          IterResult r = run_iter(pair, bytes, region_stride, bidir);
          ok = r.ok;
          if (i == 0 && ok)
            verified = verify(pair, kind, bytes, region_stride, bidir);
        }
        if (!ok) {
          std::fprintf(stderr, "[%s/%s/%zu] warmup failed\n", bname.c_str(), mode.c_str(), bytes);
          ++failures;
          break;
        }
        if (!verified)
          std::fprintf(stderr, "[%s/%s/%zu] WARNING: data verification failed\n", bname.c_str(),
                       mode.c_str(), bytes);

        std::vector<double> lat, rlat;
        for (int i = 0; i < iters; ++i) {
          IterResult r = run_iter(pair, bytes, region_stride, bidir);
          if (!r.ok) {
            ok = false;
            break;
          }
          lat.push_back(r.latency_us);
          rlat.push_back(r.recv_latency_us);
          raw << bname << ',' << bench::device_name(kind) << ',' << mode << ',' << bytes << ','
              << i << ',' << r.latency_us << ',' << r.recv_latency_us << '\n';
        }
        if (!ok || lat.empty()) {
          std::fprintf(stderr, "[%s/%s/%zu] measured iteration failed\n", bname.c_str(),
                       mode.c_str(), bytes);
          ++failures;
          break;
        }
        const double total_bytes = static_cast<double>(bytes) * (bidir ? 2 : 1);
        const double med = bench::percentile(lat, 50);
        const double best = *std::min_element(lat.begin(), lat.end());
        summary << bname << ',' << bench::device_name(kind) << ',' << mode << ',' << bytes << ','
                << iters << ',' << (verified ? 1 : 0) << ',' << med << ',' << bench::mean(lat)
                << ',' << bench::stddev(lat) << ',' << best << ',' << bench::percentile(lat, 95)
                << ',' << bench::percentile(rlat, 50) << ',' << total_bytes / (med * 1e3) << ','
                << total_bytes / (best * 1e3) << '\n';
        summary.flush();
        std::fprintf(stderr, "[%s/%s] %8.1f MB  median %10.1f us  %7.2f GB/s%s\n", bname.c_str(),
                     mode.c_str(), static_cast<double>(bytes) / (1024.0 * 1024.0), med,
                     total_bytes / (med * 1e3), verified ? "" : "  (UNVERIFIED)");
      }
    }
    teardown_pair(pair, kind);
  }

  std::fprintf(stderr, "wrote %s/{transport.csv,transport_raw.csv,transport_setup.csv}\n",
               out_dir.c_str());
  return failures == 0 ? 0 : 2;
}
