#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/status.h"
#include "ops/common/shapes.h"
#include "ops/common/tensor_checks.h"
#include "ops/cuda_backend.cuh"
#include "ops/gpu/attention.cuh"
#include "ops/gpu/conv.cuh"
#include "ops/gpu/element_wise.cuh"
#include "ops/gpu/embed.cuh"
#include "ops/gpu/linear.cuh"
#include "ops/gpu/mamba2_scan.cuh"
#include "ops/gpu/rms_norm.cuh"

namespace {
// y = x * W^T  with W: [N, K]
Status gpu_linear(cublasHandle_t handle, const Tensor& x, const Tensor& W, Tensor& y) {
  if (Status s = require_f32(x, "x", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(W, "W", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(y, "y", Device::GPU); !s.ok())
    return s;

  StatusOr<std::vector<int64_t>> expected = linear_output_shape(x, W);
  if (!expected.ok())
    return Status(expected.status());
  if (!same_shape(y.shape, expected.value()))
    return Status::InvalidArgument("out shape does not match linear result");

  int64_t K = 0;
  ASSIGN_OR_RETURN(K, x.last_dim());
  const int64_t N = W.shape[0];
  int64_t numel = 0;
  ASSIGN_OR_RETURN(numel, x.numel());
  const int M = static_cast<int>(numel / K);

  cudaError_t err =
      linear_f32(handle, static_cast<const float*>(x.buffer.ptr),
                 static_cast<const float*>(W.buffer.ptr), static_cast<float*>(y.buffer.ptr), M,
                 static_cast<int>(N), static_cast<int>(K));
  if (err != cudaSuccess)
    return Status::RuntimeError(std::string("linear_f32 / cublasSgemm failed: ") +
                                cudaGetErrorString(err));
  return Status::Ok();
}

StatusOr<Tensor> alloc_gpu_f32(std::vector<int64_t> shape) {
  StatusOr<Tensor> t = allocate_f32_tensor(std::move(shape));
  if (!t.ok())
    return Status(t.status());
  if (t.value().buffer.device != Device::GPU)
    return Status::RuntimeError("allocate_f32_tensor did not use GPU allocator");
  return std::move(t.value());
}

int launch_grid(int n, int block) { return (n + block - 1) / block; }
} // namespace

CUDABackend::CUDABackend(int device_id) : OpsBackend(Device::GPU), device_id_(device_id) {}

CUDABackend::~CUDABackend() {
  if (!handle_)
    return;
  if (cudaSetDevice(device_id_) == cudaSuccess)
    cublasDestroy(handle_);
  handle_ = nullptr;
}

Status CUDABackend::bind_device_() const {
  cudaError_t err = cudaSetDevice(device_id_);
  if (err != cudaSuccess)
    return Status::RuntimeError(std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
  return Status::Ok();
}

Status CUDABackend::ensure_cublas_() const {
  if (Status s = bind_device_(); !s.ok())
    return s;
  if (handle_)
    return Status::Ok();
  if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS)
    return Status::RuntimeError("cublasCreate failed");
  return Status::Ok();
}

Status CUDABackend::rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) const {
  if (Status s = bind_device_(); !s.ok())
    return s;
    if (Status s = require_f32(x, "x", Device::GPU); !s.ok())
    return s;
    if (Status s = require_f32(weight, "weight", Device::GPU); !s.ok())
      return s;
    if (Status s = require_f32(out, "out", Device::GPU); !s.ok())
      return s;
    if (Status s = validate_rms_norm_shapes(x, weight, out); !s.ok())
      return s;

    int64_t N;
    ASSIGN_OR_RETURN(N, x.last_dim());

    int64_t numel;
    ASSIGN_OR_RETURN(numel, x.numel());

    const int rows = static_cast<int>(numel / N);
    const int n = static_cast<int>(N);

    ::rms_norm<<<rows, block_size_>>>(static_cast<const float*>(x.buffer.ptr),
                                      static_cast<float*>(out.buffer.ptr),
                                      static_cast<const float*>(weight.buffer.ptr), n, eps);

    return Status::Ok();
}

Status CUDABackend::add(const Tensor& a, const Tensor& b, Tensor& out) const {
  if (Status s = bind_device_(); !s.ok())
    return s;
    if (Status s = require_f32(a, "a", Device::GPU); !s.ok())
    return s;
    if (Status s = require_f32(b, "b", Device::GPU); !s.ok())
        return s;
    if (Status s = require_f32(out, "out", Device::GPU); !s.ok())
        return s;
    if (!same_shape(a.shape, b.shape)) {
        return Status::InvalidArgument("shape mismatch between a, b");
    }
    if (!same_shape(a.shape, out.shape)) {
        return Status::InvalidArgument("shape mismatch  between a/b, out");
    }

    int64_t numel = 0;
    ASSIGN_OR_RETURN(numel, a.numel());
    const int n = static_cast<int>(numel);
    const int grid = (n + block_size_ - 1) / block_size_;

    ::add<<<grid, block_size_>>>(static_cast<const float*>(a.buffer.ptr),
                                 static_cast<const float*>(b.buffer.ptr),
                                 static_cast<float*>(out.buffer.ptr), n);
    return Status::Ok();
}

Status CUDABackend::scale(Tensor& x, float s) const {
  if (Status s_bind = bind_device_(); !s_bind.ok())
    return s_bind;
    if (Status st = require_f32(x, "x", Device::GPU); !st.ok())
    return st;

    int64_t numel = 0;
    ASSIGN_OR_RETURN(numel, x.numel());
    const int n = static_cast<int>(numel);
    const int grid = (n + block_size_ - 1) / block_size_;

    ::scale<<<grid, block_size_>>>(static_cast<float*>(x.buffer.ptr), s, n);
    return Status::Ok();
}

Status CUDABackend::linear(const Tensor& x, const Tensor& W, Tensor& out) const {
  if (Status s = ensure_cublas_(); !s.ok())
    return s;
  return gpu_linear(handle_, x, W, out);
}

Status CUDABackend::embedding_lookup(const Tensor& table, const int32_t* tokens, int64_t n_tokens,
                                     float scale, Tensor& out) const {
  if (tokens == nullptr || n_tokens <= 0)
    return Status::InvalidArgument("embedding_lookup requires at least one token");
  if (Status s = bind_device_(); !s.ok())
    return s;
  if (Status s = require_f32(table, "table", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::GPU); !s.ok())
    return s;
  if (table.size() != 2)
    return Status::InvalidArgument("embedding table must be rank 2 [vocab, hidden]");

  const int64_t vocab = table.shape[0];
  const int64_t hidden = table.shape[1];
  const int T = static_cast<int>(n_tokens);
  if (out.size() != 2 || out.shape[0] != n_tokens || out.shape[1] != hidden)
    return Status::InvalidArgument("out shape must be [T, hidden]");

  for (int64_t i = 0; i < n_tokens; ++i) {
    const int32_t id = tokens[i];
    if (id < 0 || static_cast<int64_t>(id) >= vocab)
      return Status::InvalidArgument("token id out of range");
  }

  int32_t* d_tokens = nullptr;
  const size_t tok_bytes = static_cast<size_t>(T) * sizeof(int32_t);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_tokens), tok_bytes);
  if (err != cudaSuccess)
    return Status::OOM("cudaMalloc tokens for embedding_lookup failed");

  err = cudaMemcpy(d_tokens, tokens, tok_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_tokens);
    return Status::RuntimeError(std::string("cudaMemcpy H2D tokens failed: ") +
                                   cudaGetErrorString(err));
  }

  err = embedding_lookup_f32(static_cast<const float*>(table.buffer.ptr), d_tokens,
                             static_cast<float*>(out.buffer.ptr), T, static_cast<int>(hidden),
                             static_cast<int>(vocab), scale, block_size_);
  cudaFree(d_tokens);
  if (err != cudaSuccess)
    return Status::RuntimeError(std::string("embedding_lookup_f32 failed: ") +
                                   cudaGetErrorString(err));
  return Status::Ok();
}

Status CUDABackend::take_last_row(const Tensor& hidden, Tensor& out) const {
  if (Status s = bind_device_(); !s.ok())
    return s;
  if (Status s = require_f32(hidden, "hidden", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::GPU); !s.ok())
    return s;
  if (hidden.empty())
    return Status::InvalidArgument("hidden is empty");

  if (hidden.size() == 1) {
    if (!same_shape(out.shape, hidden.shape))
      return Status::InvalidArgument("out shape must match rank-1 hidden");
    cudaError_t err =
        cudaMemcpy(out.buffer.ptr, hidden.buffer.ptr, hidden.buffer.bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
      return Status::RuntimeError(std::string("cudaMemcpy D2D failed: ") +
                                     cudaGetErrorString(err));
    return Status::Ok();
  }

  int64_t rows = 0;
  int64_t cols = 0;
  ASSIGN_OR_RETURN(rows, hidden.rows());
  ASSIGN_OR_RETURN(cols, hidden.cols());
  if (rows <= 0)
    return Status::InvalidArgument("hidden has no rows");
  if (out.size() != 1 || out.shape[0] != cols)
    return Status::InvalidArgument("out must be shape {H}");

  const float* src = static_cast<const float*>(hidden.buffer.ptr) +
                     static_cast<size_t>(rows - 1) * static_cast<size_t>(cols);
  cudaError_t err = cudaMemcpy(out.buffer.ptr, src, static_cast<size_t>(cols) * sizeof(float),
                               cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess)
    return Status::RuntimeError(std::string("cudaMemcpy D2D last row failed: ") +
                                   cudaGetErrorString(err));
  return Status::Ok();
}

Status CUDABackend::mlp_f32(const Tensor& normed_hidden, const MlpWeights& w, const MlpConfig& mlp,
                            const ScaleConfig* scales, Tensor& out) const {
  (void)mlp;
  if (Status s = ensure_cublas_(); !s.ok())
    return s;
  if (Status s = require_f32(normed_hidden, "normed_hidden", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::GPU); !s.ok())
    return s;
  if (!same_shape(out.shape, normed_hidden.shape))
    return Status::InvalidArgument("out shape must match hidden");

  const float gate_mult = scales ? scales->mlp[0] : 1.f;
  const float down_mult = scales ? scales->mlp[1] : 1.f;

  std::vector<int64_t> up_shape;
  ASSIGN_OR_RETURN(up_shape, linear_output_shape(normed_hidden, w.up_proj));
  StatusOr<Tensor> up_or = allocate_f32_tensor(std::move(up_shape));
  if (!up_or.ok())
    return Status(up_or.status());
  Tensor up = std::move(up_or.value());

  std::vector<int64_t> gate_shape;
  ASSIGN_OR_RETURN(gate_shape, linear_output_shape(normed_hidden, w.gate_proj));
  StatusOr<Tensor> gate_or = allocate_f32_tensor(std::move(gate_shape));
  if (!gate_or.ok())
    return Status(gate_or.status());
  Tensor gate = std::move(gate_or.value());

  if (Status s = gpu_linear(handle_, normed_hidden, w.up_proj, up); !s.ok())
    return s;
  if (Status s = gpu_linear(handle_, normed_hidden, w.gate_proj, gate); !s.ok())
    return s;

  int64_t mid_numel = 0;
  ASSIGN_OR_RETURN(mid_numel, up.numel());
  const int n = static_cast<int>(mid_numel);
  const int grid = (n + block_size_ - 1) / block_size_;

  if (gate_mult != 1.f) {
    if (Status s = scale(gate, gate_mult); !s.ok())
      return s;
  }

  ::silu<<<grid, block_size_>>>(static_cast<float*>(gate.buffer.ptr), n);

  ::mul<<<grid, block_size_>>>(static_cast<const float*>(up.buffer.ptr),
                               static_cast<const float*>(gate.buffer.ptr),
                               static_cast<float*>(up.buffer.ptr), n);

  if (Status s = gpu_linear(handle_, up, w.down_proj, out); !s.ok())
    return s;

  if (down_mult != 1.f) {
    if (Status s = scale(out, down_mult); !s.ok())
      return s;
  }

  return Status::Ok();
}

Status CUDABackend::mamba2_mixer_f32(const Tensor& normed_hidden, const Mamba2MixerWeights& w,
                                     const SsmConfig& ssm, const ScaleConfig* scales,
                                     float rms_norm_eps, LayerCacheView& cache,
                                     Tensor& out) const {
  if (Status s = require_f32(normed_hidden, "normed_hidden", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::GPU); !s.ok())
    return s;

  int64_t T = 1;
  int64_t hidden_dim = 0;
  if (normed_hidden.size() == 1) {
    hidden_dim = normed_hidden.shape[0];
    T = 1;
  } else if (normed_hidden.size() >= 2) {
    ASSIGN_OR_RETURN(T, normed_hidden.batch_size());
    ASSIGN_OR_RETURN(hidden_dim, normed_hidden.last_dim());
  } else {
    return Status::InvalidArgument("invalid hidden rank");
  }

  const int I = ssm.d_inner;
  const int G = ssm.n_groups;
  const int N = ssm.d_state;
  const int H = ssm.n_heads;
  const int Dh = ssm.d_head;
  const int K = ssm.d_conv;
  const int GN = G * N;
  const int conv_dim = I + 2 * GN;
  const int proj_size = I + conv_dim + H;

  const float in_scale = scales ? scales->ssm_in : 1.f;
  const float mz = scales ? scales->ssm_chunk[0] : 1.f;
  const float mx = scales ? scales->ssm_chunk[1] : 1.f;
  const float mB = scales ? scales->ssm_chunk[2] : 1.f;
  const float mC = scales ? scales->ssm_chunk[3] : 1.f;
  const float mdt = scales ? scales->ssm_chunk[4] : 1.f;

  if (!same_shape(out.shape, normed_hidden.shape))
    return Status::InvalidArgument("out shape must match hidden");
  if (w.in_proj.size() != 2 || w.in_proj.shape[0] != proj_size ||
      w.in_proj.shape[1] != hidden_dim)
    return Status::InvalidArgument("in_proj must be [proj_size, hidden]");
  if (cache.conv.ptr == nullptr || cache.ssm.ptr == nullptr)
    return Status::InvalidArgument("cache view is null");
  const size_t expect_conv =
      static_cast<size_t>(conv_dim) * static_cast<size_t>(K - 1) * sizeof(float);
  const size_t expect_ssm =
      static_cast<size_t>(H) * static_cast<size_t>(Dh) * static_cast<size_t>(N) * sizeof(float);
  if (cache.conv.bytes < expect_conv || cache.ssm.bytes < expect_ssm)
    return Status::InvalidArgument("cache view too small for mixer dims");
  if (ssm.gated_rms_norm && w.mixer_norm == nullptr)
    return Status::InvalidArgument("mixer_norm required when gated_rms_norm is true");
  if (Status s = require_f32(w.in_proj, "in_proj", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(w.out_proj, "out_proj", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(w.conv1d, "conv1d", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(w.A_log, "A_log", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(w.D, "D", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(w.dt_bias, "dt_bias", Device::GPU); !s.ok())
    return s;
  if (ssm.use_conv_bias) {
    if (Status s = require_f32(w.conv1d_bias, "conv1d_bias", Device::GPU); !s.ok())
      return s;
  }
  if (ssm.gated_rms_norm) {
    if (Status s = require_f32(*w.mixer_norm, "mixer_norm", Device::GPU); !s.ok())
      return s;
  }

  if (Status s = ensure_cublas_(); !s.ok())
    return s;

  StatusOr<Tensor> scaled_or = alloc_gpu_f32(normed_hidden.shape);
  if (!scaled_or.ok())
    return Status(scaled_or.status());
  Tensor scaled_hidden = std::move(scaled_or.value());
  if (in_scale != 1.f) {
    if (cudaMemcpy(scaled_hidden.buffer.ptr, normed_hidden.buffer.ptr, normed_hidden.buffer.bytes,
                   cudaMemcpyDeviceToDevice) != cudaSuccess)
      return Status::RuntimeError("cudaMemcpy scaled_hidden failed");
    if (Status s = scale(scaled_hidden, in_scale); !s.ok())
      return s;
  } else {
    if (cudaMemcpy(scaled_hidden.buffer.ptr, normed_hidden.buffer.ptr, normed_hidden.buffer.bytes,
                   cudaMemcpyDeviceToDevice) != cudaSuccess)
      return Status::RuntimeError("cudaMemcpy scaled_hidden failed");
  }

  StatusOr<std::vector<int64_t>> projected_shape_or = linear_output_shape(scaled_hidden, w.in_proj);
  if (!projected_shape_or.ok())
    return Status(projected_shape_or.status());
  StatusOr<Tensor> projected_or = alloc_gpu_f32(std::move(projected_shape_or).value());
  if (!projected_or.ok())
    return Status(projected_or.status());
  Tensor projected = std::move(projected_or.value());
  if (Status s = gpu_linear(handle_, scaled_hidden, w.in_proj, projected); !s.ok())
    return s;
  if (projected.size() == 1)
    projected.shape = {1, projected.shape[0]};

  StatusOr<Tensor> gate_or = alloc_gpu_f32({T, static_cast<int64_t>(I)});
  StatusOr<Tensor> xBC_or = alloc_gpu_f32({T, static_cast<int64_t>(conv_dim)});
  StatusOr<Tensor> dt_or = alloc_gpu_f32({T, static_cast<int64_t>(H)});
  if (!gate_or.ok())
    return Status(gate_or.status());
  if (!xBC_or.ok())
    return Status(xBC_or.status());
  if (!dt_or.ok())
    return Status(dt_or.status());
  Tensor gate = std::move(gate_or.value());
  Tensor xBC = std::move(xBC_or.value());
  Tensor dt = std::move(dt_or.value());

  {
    const int n = static_cast<int>(T) * proj_size;
    ::mixer_split_proj<<<launch_grid(n, block_size_), block_size_>>>(
        static_cast<const float*>(projected.buffer.ptr), static_cast<float*>(gate.buffer.ptr),
        static_cast<float*>(xBC.buffer.ptr), static_cast<float*>(dt.buffer.ptr),
        static_cast<int>(T), I, GN, H, mz, mx, mB, mC, mdt);
  }

  if (w.conv1d.numel().ok() && w.conv1d.numel().value() != static_cast<int64_t>(conv_dim) * K)
    return Status::InvalidArgument("conv1d weight numel mismatch");

  StatusOr<Tensor> xBC_out_or = alloc_gpu_f32({T, static_cast<int64_t>(conv_dim)});
  if (!xBC_out_or.ok())
    return Status(xBC_out_or.status());
  Tensor xBC_out = std::move(xBC_out_or.value());

  const float* conv_b =
      ssm.use_conv_bias ? static_cast<const float*>(w.conv1d_bias.buffer.ptr) : nullptr;
  ::causal_conv1d<<<launch_grid(conv_dim, block_size_), block_size_>>>(
      static_cast<const float*>(xBC.buffer.ptr), T, conv_dim, K,
      static_cast<const float*>(w.conv1d.buffer.ptr), conv_b, static_cast<float*>(cache.conv.ptr),
      static_cast<float*>(xBC_out.buffer.ptr), true);

  StatusOr<Tensor> x_or = alloc_gpu_f32({T, static_cast<int64_t>(I)});
  StatusOr<Tensor> B_or = alloc_gpu_f32({T, static_cast<int64_t>(GN)});
  StatusOr<Tensor> C_or = alloc_gpu_f32({T, static_cast<int64_t>(GN)});
  if (!x_or.ok() || !B_or.ok() || !C_or.ok())
    return Status::OOM("mixer split alloc failed");
  Tensor x = std::move(x_or.value());
  Tensor B = std::move(B_or.value());
  Tensor C = std::move(C_or.value());

  {
    const int n = static_cast<int>(T) * conv_dim;
    ::mixer_split_xbc<<<launch_grid(n, block_size_), block_size_>>>(
        static_cast<const float*>(xBC_out.buffer.ptr), static_cast<float*>(x.buffer.ptr),
        static_cast<float*>(B.buffer.ptr), static_cast<float*>(C.buffer.ptr), static_cast<int>(T),
        I, GN);
  }

  StatusOr<Tensor> y_or = alloc_gpu_f32({T, static_cast<int64_t>(I)});
  if (!y_or.ok())
    return Status(y_or.status());
  Tensor y = std::move(y_or.value());

  ::mamba2_ssm_scan<<<launch_grid(H * Dh, block_size_), block_size_>>>(
      static_cast<const float*>(x.buffer.ptr), static_cast<const float*>(B.buffer.ptr),
      static_cast<const float*>(C.buffer.ptr), static_cast<const float*>(dt.buffer.ptr),
      static_cast<const float*>(w.A_log.buffer.ptr), static_cast<const float*>(w.D.buffer.ptr),
      static_cast<const float*>(w.dt_bias.buffer.ptr), static_cast<float*>(cache.ssm.ptr),
      static_cast<float*>(y.buffer.ptr), static_cast<int>(T), H, Dh, N, G);

  {
    int64_t n64 = 0;
    ASSIGN_OR_RETURN(n64, y.numel());
    const int n = static_cast<int>(n64);
    const int grid = launch_grid(n, block_size_);
    ::silu<<<grid, block_size_>>>(static_cast<float*>(gate.buffer.ptr), n);
    ::mul<<<grid, block_size_>>>(static_cast<const float*>(y.buffer.ptr),
                                 static_cast<const float*>(gate.buffer.ptr),
                                 static_cast<float*>(y.buffer.ptr), n);
    if (ssm.gated_rms_norm) {
      if (Status s = rms_norm(y, *w.mixer_norm, rms_norm_eps, y); !s.ok())
        return s;
    }
  }

  if (w.out_proj.size() != 2 || w.out_proj.shape[0] != hidden_dim || w.out_proj.shape[1] != I)
    return Status::InvalidArgument("out_proj must be [hidden, intermediate]");

  if (normed_hidden.size() == 1)
    y.shape = {I};
  return gpu_linear(handle_, y, w.out_proj, out);
}

Status CUDABackend::attention_f32(const Tensor& normed_hidden, const AttnWeights& w,
                                  const AttnConfig& attn, const ScaleConfig* scales,
                                  int max_seq_length, int hidden_size, LayerCacheView& cache,
                                  int64_t past_len, Tensor& out) const {
  if (Status s = require_f32(normed_hidden, "normed_hidden", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::GPU); !s.ok())
    return s;
  if (cache.k.ptr == nullptr || cache.v.ptr == nullptr)
    return Status::InvalidArgument("attn cache view is null");
  if (past_len < 0)
    return Status::InvalidArgument("past_len must be non-negative");

  const int64_t D = hidden_size;
  const int Hq = attn.n_q_heads;
  const int Hkv = attn.n_kv_heads;
  const int Dh = attn.head_dim;
  if (Hq <= 0 || Hkv <= 0 || Dh <= 0 || Hq % Hkv != 0)
    return Status::InvalidArgument("invalid attention head config");
  const int64_t kv_dim = static_cast<int64_t>(Hkv) * Dh;
  const int64_t q_dim = static_cast<int64_t>(Hq) * Dh;
  const float key_mult = scales ? scales->key : 1.f;

  int64_t T = 1;
  if (normed_hidden.size() == 1) {
    if (normed_hidden.shape[0] != D)
      return Status::InvalidArgument("hidden dim mismatch");
  } else if (normed_hidden.size() >= 2) {
    ASSIGN_OR_RETURN(T, normed_hidden.batch_size());
    int64_t last = 0;
    ASSIGN_OR_RETURN(last, normed_hidden.last_dim());
    if (last != D)
      return Status::InvalidArgument("hidden last dim mismatch");
  } else {
    return Status::InvalidArgument("invalid hidden rank");
  }

  if (!same_shape(out.shape, normed_hidden.shape))
    return Status::InvalidArgument("out shape must match hidden");
  if (past_len + T > max_seq_length)
    return Status::KvCacheOverflow("KV cache overflow");

  const size_t expect_kv =
      static_cast<size_t>(max_seq_length) * static_cast<size_t>(kv_dim) * sizeof(float);
  if (cache.k.bytes < expect_kv || cache.v.bytes < expect_kv)
    return Status::InvalidArgument("KV cache view too small");

  if (w.q_proj.size() != 2 || w.q_proj.shape[0] != q_dim || w.q_proj.shape[1] != D)
    return Status::InvalidArgument("q_proj must be [Hq*Dh, D]");
  if (w.k_proj.size() != 2 || w.k_proj.shape[0] != kv_dim || w.k_proj.shape[1] != D)
    return Status::InvalidArgument("k_proj must be [Hkv*Dh, D]");
  if (w.v_proj.size() != 2 || w.v_proj.shape[0] != kv_dim || w.v_proj.shape[1] != D)
    return Status::InvalidArgument("v_proj must be [Hkv*Dh, D]");
  if (w.o_proj.size() != 2 || w.o_proj.shape[0] != D || w.o_proj.shape[1] != q_dim)
    return Status::InvalidArgument("o_proj must be [D, Hq*Dh]");

  if (Status s = require_f32(w.q_proj, "q_proj", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(w.k_proj, "k_proj", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(w.v_proj, "v_proj", Device::GPU); !s.ok())
    return s;
  if (Status s = require_f32(w.o_proj, "o_proj", Device::GPU); !s.ok())
    return s;

  if (Status s = ensure_cublas_(); !s.ok())
    return s;

  StatusOr<std::vector<int64_t>> q_shape_or = linear_output_shape(normed_hidden, w.q_proj);
  StatusOr<std::vector<int64_t>> k_shape_or = linear_output_shape(normed_hidden, w.k_proj);
  StatusOr<std::vector<int64_t>> v_shape_or = linear_output_shape(normed_hidden, w.v_proj);
  if (!q_shape_or.ok())
    return Status(q_shape_or.status());
  if (!k_shape_or.ok())
    return Status(k_shape_or.status());
  if (!v_shape_or.ok())
    return Status(v_shape_or.status());

  StatusOr<Tensor> Q_or = alloc_gpu_f32(std::move(q_shape_or).value());
  StatusOr<Tensor> K_or = alloc_gpu_f32(std::move(k_shape_or).value());
  StatusOr<Tensor> V_or = alloc_gpu_f32(std::move(v_shape_or).value());
  if (!Q_or.ok())
    return Status(Q_or.status());
  if (!K_or.ok())
    return Status(K_or.status());
  if (!V_or.ok())
    return Status(V_or.status());
  Tensor Q = std::move(Q_or.value());
  Tensor K = std::move(K_or.value());
  Tensor V = std::move(V_or.value());

  if (Status s = gpu_linear(handle_, normed_hidden, w.q_proj, Q); !s.ok())
    return s;
  if (Status s = gpu_linear(handle_, normed_hidden, w.k_proj, K); !s.ok())
    return s;
  if (Status s = gpu_linear(handle_, normed_hidden, w.v_proj, V); !s.ok())
    return s;

  if (Q.size() == 1)
    Q.shape = {1, Q.shape[0]};
  if (K.size() == 1)
    K.shape = {1, K.shape[0]};
  if (V.size() == 1)
    V.shape = {1, V.shape[0]};

  if (key_mult != 1.f) {
    if (Status s = scale(K, key_mult); !s.ok())
      return s;
  }

  {
    const int q_rows = static_cast<int>(T) * Hq;
    const int k_rows = static_cast<int>(T) * Hkv;
    ::apply_rope<<<launch_grid(q_rows, block_size_), block_size_>>>(
        static_cast<float*>(Q.buffer.ptr), static_cast<int>(T), Hq, Dh, static_cast<int>(past_len),
        attn.rope_theta);
    ::apply_rope<<<launch_grid(k_rows, block_size_), block_size_>>>(
        static_cast<float*>(K.buffer.ptr), static_cast<int>(T), Hkv, Dh, static_cast<int>(past_len),
        attn.rope_theta);
  }

  {
    const int n = static_cast<int>(T * kv_dim);
    ::kv_append<<<launch_grid(n, block_size_), block_size_>>>(
        static_cast<const float*>(K.buffer.ptr), static_cast<const float*>(V.buffer.ptr),
        static_cast<float*>(cache.k.ptr), static_cast<float*>(cache.v.ptr), static_cast<int>(T),
        static_cast<int>(kv_dim), static_cast<int>(past_len));
  }

  StatusOr<Tensor> attn_out_or = alloc_gpu_f32({T, q_dim});
  if (!attn_out_or.ok())
    return Status(attn_out_or.status());
  Tensor attn_out = std::move(attn_out_or.value());

  const float attn_scale = 1.f / std::sqrt(static_cast<float>(Dh));
  const size_t shmem = static_cast<size_t>(past_len + T) * sizeof(float);
  ::causal_attention_fused<<<static_cast<int>(T) * Hq, block_size_, shmem>>>(
      static_cast<const float*>(Q.buffer.ptr), static_cast<const float*>(cache.k.ptr),
      static_cast<const float*>(cache.v.ptr), static_cast<float*>(attn_out.buffer.ptr),
      static_cast<int>(T), Hq, Hkv, Dh, static_cast<int>(past_len), attn_scale);

  if (normed_hidden.size() == 1)
    attn_out.shape = {q_dim};
  return gpu_linear(handle_, attn_out, w.o_proj, out);
}

const OpsBackend& cuda_ops(int device_id) {
  if (device_id < 0)
    device_id = 0;
  static std::mutex mu;
  static std::unordered_map<int, std::unique_ptr<CUDABackend>> backends;
  std::lock_guard<std::mutex> lk(mu);
  auto& slot = backends[device_id];
  if (!slot)
    slot = std::make_unique<CUDABackend>(device_id);
  return *slot;
}