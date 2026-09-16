#include "ops/cpu/mamba2_mixer.h"

#include "ops/cpu/element_wise.h"
#include "ops/cpu/linear.h"
#include "ops/cpu/map.h"
#include "ops/cpu/rms_norm.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

inline float softplus(float x) {
  if (x > 20.f)
    return x;
  if (x < -20.f)
    return std::exp(x);
  return std::log1p(std::exp(x));
}

inline float silu(float x) { return x / (1.f + std::exp(-x)); }

} // namespace

Status causal_conv1d_f32(const float* x, int64_t T, int64_t conv_dim, int kernel,
                         const float* weight, const float* bias, float* cache, float* y,
                         bool apply_silu) {
  if (T <= 0 || conv_dim <= 0 || kernel <= 1) {
    return Status::InvalidArgument("invalid causal_conv1d dims");
  }
  if (x == nullptr || weight == nullptr || cache == nullptr || y == nullptr) {
    return Status::InvalidArgument("null pointer in causal_conv1d");
  }

  const int K = kernel;
  const int left = K - 1;

  // for each token
  for (int64_t t = 0; t < T; ++t) {
    // for each channel in that token (we process each channel independently)
    for (int64_t c = 0; c < conv_dim; ++c) {
      // get the cache for this particular channel
      float* crow = cache + static_cast<size_t>(c) * static_cast<size_t>(left);
      // current input (token, channel)
      const float xt =
          x[static_cast<size_t>(t) * static_cast<size_t>(conv_dim) + static_cast<size_t>(c)];
      // corresponding weight for that channel
      const float* wt = weight + static_cast<size_t>(c) * static_cast<size_t>(K);
      float acc = bias ? bias[c] : 0.f;
      // weighted sum for that particular (token, channel)
      for (int k = 0; k < left; ++k)
        acc += wt[k] * crow[k];
      // add current
      acc += wt[left] * xt;
      // conditionally apply silu
      if (apply_silu)
        acc = silu(acc);
      // store result into output
      y[static_cast<size_t>(t) * static_cast<size_t>(conv_dim) + static_cast<size_t>(c)] = acc;

      // update cache
      // Roll left context and append current token
      for (int i = 0; i < left - 1; ++i)
        crow[i] = crow[i + 1];
      if (left > 0)
        crow[left - 1] = xt;
    }
  }
  return Status::Ok();
}

Status mamba2_ssm_scan_f32(const float* x, const float* B, const float* C, const float* dt,
                           const float* A_log, const float* D, const float* dt_bias, float* state,
                           float* y, int64_t T, const Mamba2Config& cfg) {
  const int H = cfg.num_heads;       // number of heads to be split into
  const int Dh = cfg.head_dim;       // dimension of each split head
  const int N = cfg.state_size;      // state dimension
  const int G = cfg.n_groups;        // how are Bi,Ci pairs are shared among all the heads
  const int heads_per_group = H / G; // number of heads each group attends to
  const int64_t intermediate = static_cast<int64_t>(H) * Dh; // full intermediate state
  const int64_t BC_width = static_cast<int64_t>(G) * N;

  if (T <= 0)
    return Status::InvalidArgument("ssm scan requires T > 0");

  for (int64_t t = 0; t < T; ++t) {
    const float* x_t = x + t * intermediate;
    const float* B_t = B + t * BC_width;
    const float* C_t = C + t * BC_width;
    const float* dt_t = dt + t * H;
    float* y_t = y + t * intermediate;

    for (int h = 0; h < H; ++h) {
      const int g = h / heads_per_group;
      const float dt_h = softplus(dt_t[h] + dt_bias[h]);
      const float A_h = -std::exp(A_log[h]);
      const float dA = std::exp(dt_h * A_h);
      const float D_h = D[h];

      for (int d = 0; d < Dh; ++d) {
        const float x_hd = x_t[h * Dh + d];
        float y_hd = 0.f;
        for (int n = 0; n < N; ++n) {
          const float B_hn = B_t[g * N + n];
          const float C_hn = C_t[g * N + n];
          float& s = state[(static_cast<size_t>(h) * Dh + d) * N + n];
          s = s * dA + dt_h * B_hn * x_hd;
          y_hd += s * C_hn;
        }
        y_t[h * Dh + d] = y_hd + D_h * x_hd;
      }
    }
  }
  return Status::Ok();
}

Status rms_norm_gated_f32(Tensor& y, const Tensor& gate, const Tensor& weight, float eps) {
  if (Status s = require_f32_cpu(y, "y"); !s.ok())
    return s;
  if (Status s = require_f32_cpu(gate, "gate"); !s.ok())
    return s;
  if (Status s = require_f32_cpu(weight, "weight"); !s.ok())
    return s;
  if (!same_shape(y.shape, gate.shape)) {
    return Status::InvalidArgument("y and gate shape mismatch");
  }

  // HF MambaRMSNormGated: y = y * silu(gate); then RMSNorm(y)
  StatusOr<Tensor> gate_act = silu(gate);
  if (!gate_act.ok())
    return Status(gate_act.status());
  if (Status s = mul(y, gate_act.value(), y); !s.ok())
    return s;
  return rms_norm_inplace(y, weight, eps);
}

Status mamba2_mixer_f32(const Tensor& normed_hidden, const Mamba2LayerWeights& w,
                        const Mamba2Config& cfg, LayerCacheView& cache, Tensor& out) {
  if (Status s = require_f32_cpu(normed_hidden, "normed_hidden"); !s.ok())
    return s;
  if (Status s = require_f32_cpu(out, "out"); !s.ok())
    return s;

  const int64_t D = cfg.hidden_size;                                    // token dimension
  const int64_t I = static_cast<int64_t>(cfg.hidden_size) * cfg.expand; // expanded dimension
  const int64_t G = cfg.n_groups;         // defines how a Bi,Ci pair is shared among multiple heads
  const int64_t N = cfg.state_size;       // state dimension
  const int64_t H = cfg.num_heads;        // number of heads the head dimension is split into
  const int64_t conv_dim = I + 2 * G * N; // convolution on xBC
  const int64_t proj_size = I + conv_dim + H; // d_mlp == 0 for standard Mamba2

  int64_t T = 1;
  if (normed_hidden.size() == 1) {
    if (normed_hidden.shape[0] != D) {
      return Status::InvalidArgument("hidden dim mismatch");
    }
    T = 1;
  } else if (normed_hidden.size() >= 2) {
    ASSIGN_OR_RETURN(T, normed_hidden.batch_size());
    int64_t last = 0;
    ASSIGN_OR_RETURN(last, normed_hidden.last_dim());
    if (last != D)
      return Status::InvalidArgument("hidden last dim mismatch");
  } else {
    return Status::InvalidArgument("invalid hidden rank");
  }

  if (!same_shape(out.shape, normed_hidden.shape)) {
    return Status::InvalidArgument("out shape must match hidden");
  }

  if (w.in_proj.size() != 2 || w.in_proj.shape[0] != proj_size || w.in_proj.shape[1] != D) {
    return Status::InvalidArgument("in_proj must be [proj_size, hidden]");
  }
  if (cache.conv.ptr == nullptr || cache.ssm.ptr == nullptr) {
    return Status::InvalidArgument("cache view is null");
  }
  const size_t expect_conv =
      static_cast<size_t>(conv_dim) * static_cast<size_t>(cfg.conv_kernel - 1) * sizeof(float);
  const size_t expect_ssm = static_cast<size_t>(H) * static_cast<size_t>(cfg.head_dim) *
                            static_cast<size_t>(N) * sizeof(float);
  if (cache.conv.bytes < expect_conv || cache.ssm.bytes < expect_ssm) {
    return Status::InvalidArgument("cache view too small for mixer dims");
  }

  // projected: [T, proj_size] = gate[I] | xBC[conv_dim] | dt[H]
  StatusOr<Tensor> projected = linear(normed_hidden, w.in_proj);
  if (!projected.ok())
    return Status(projected.status());

  // Ensure projected is [T, proj_size]
  if (projected.value().size() == 1) {
    projected.value().shape = {1, projected.value().shape[0]};
  }

  StatusOr<Tensor> gate = allocate_f32_tensor({T, I});
  StatusOr<Tensor> xBC = allocate_f32_tensor({T, conv_dim});
  StatusOr<Tensor> dt = allocate_f32_tensor({T, H});
  if (!gate.ok())
    return Status(gate.status());
  if (!xBC.ok())
    return Status(xBC.status());
  if (!dt.ok())
    return Status(dt.status());

  {
    const auto P = as_mat_f32(projected.value());
    auto g = as_mat_f32(gate.value());
    auto xbc = as_mat_f32(xBC.value());
    auto d = as_mat_f32(dt.value());
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t i = 0; i < I; ++i)
        g(t, i) = P(t, i);
      for (int64_t i = 0; i < conv_dim; ++i)
        xbc(t, i) = P(t, I + i);
      for (int64_t i = 0; i < H; ++i)
        d(t, i) = P(t, I + conv_dim + i);
    }
  }

  // Conv weight: HF [conv_dim, 1, K] -> stored as [conv_dim, K] after load, or [conv_dim*K]
  if (w.conv1d.numel().ok() && w.conv1d.numel().value() != conv_dim * cfg.conv_kernel) {
    return Status::InvalidArgument("conv1d weight numel mismatch");
  }
  const float* conv_w = static_cast<const float*>(w.conv1d.buffer.ptr);
  const float* conv_b =
      cfg.use_conv_bias ? static_cast<const float*>(w.conv1d_bias.buffer.ptr) : nullptr;

  StatusOr<Tensor> xBC_out = allocate_f32_tensor({T, conv_dim});
  if (!xBC_out.ok())
    return Status(xBC_out.status());

  // convulate expanded input
  if (Status s =
          causal_conv1d_f32(static_cast<const float*>(xBC.value().buffer.ptr), T, conv_dim,
                            cfg.conv_kernel, conv_w, conv_b, static_cast<float*>(cache.conv.ptr),
                            static_cast<float*>(xBC_out.value().buffer.ptr), /*silu=*/true);
      !s.ok()) {
    return s;
  }

  // TODO: convert creation of xBC to making it view directly into the projectec tensor
  StatusOr<Tensor> x = allocate_f32_tensor({T, I});
  StatusOr<Tensor> B = allocate_f32_tensor({T, G * N});
  StatusOr<Tensor> C = allocate_f32_tensor({T, G * N});
  if (!x.ok() || !B.ok() || !C.ok())
    return Status::OOM("mixer split alloc failed");

  {
    const auto src = as_mat_f32(xBC_out.value());
    auto xx = as_mat_f32(x.value());
    auto bb = as_mat_f32(B.value());
    auto cc = as_mat_f32(C.value());
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t i = 0; i < I; ++i)
        xx(t, i) = src(t, i);
      for (int64_t i = 0; i < G * N; ++i) {
        bb(t, i) = src(t, I + i);
        cc(t, i) = src(t, I + G * N + i);
      }
    }
  }

  StatusOr<Tensor> y = allocate_f32_tensor({T, I});
  if (!y.ok())
    return Status(y.status());

  // do actual SSM update
  if (Status s = mamba2_ssm_scan_f32(
          static_cast<const float*>(x.value().buffer.ptr),
          static_cast<const float*>(B.value().buffer.ptr),
          static_cast<const float*>(C.value().buffer.ptr),
          static_cast<const float*>(dt.value().buffer.ptr),
          static_cast<const float*>(w.A_log.buffer.ptr), static_cast<const float*>(w.D.buffer.ptr),
          static_cast<const float*>(w.dt_bias.buffer.ptr), static_cast<float*>(cache.ssm.ptr),
          static_cast<float*>(y.value().buffer.ptr), T, cfg);
      !s.ok()) {
    return s;
  }

  // Match gate/y shapes for gated RMSNorm (both [T,I] or flatten consistently)
  if (Status s = rms_norm_gated_f32(y.value(), gate.value(), w.mixer_norm, cfg.layer_norm_epsilon);
      !s.ok()) {
    return s;
  }

  // out_proj: [hidden, intermediate]
  if (w.out_proj.size() != 2 || w.out_proj.shape[0] != D || w.out_proj.shape[1] != I) {
    return Status::InvalidArgument("out_proj must be [hidden, intermediate]");
  }

  // Flatten y to match out layout when hidden was rank-1
  if (normed_hidden.size() == 1) {
    y.value().shape = {I};
    return linear(y.value(), w.out_proj, out);
  }
  return linear(y.value(), w.out_proj, out);
}
