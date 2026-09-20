#include "ops/cpu_backend.h"

#include "ops/cpu/element_wise.h"
#include "ops/cpu/linear.h"
#include "ops/cpu/map.h"
#include "ops/cpu/rms_norm.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

void CPUBackend::apply_rope_inplace_(float* q, float* k, int64_t T, int Hq, int Hkv, int Dh,
                                     int64_t past_len, float rope_theta) const {
  const int half = Dh / 2;
  std::vector<float> inv_freq(static_cast<size_t>(half));
  for (int i = 0; i < half; ++i) {
    inv_freq[static_cast<size_t>(i)] =
        1.f / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(Dh));
  }

  auto rope_one = [&](float* x, int heads) {
    for (int64_t t = 0; t < T; ++t) {
      const float pos = static_cast<float>(past_len + t);
      for (int h = 0; h < heads; ++h) {
        float* row =
            x + (static_cast<size_t>(t) * static_cast<size_t>(heads) + static_cast<size_t>(h)) *
                    static_cast<size_t>(Dh);
        // rotate_half: [x1 | x2] -> [-x2 | x1], then x*cos + rot*sin
        for (int i = 0; i < half; ++i) {
          const float freq = pos * inv_freq[static_cast<size_t>(i)];
          const float c = std::cos(freq);
          const float s = std::sin(freq);
          const float x1 = row[i];
          const float x2 = row[half + i];
          row[i] = x1 * c - x2 * s;
          row[half + i] = x2 * c + x1 * s;
        }
      }
    }
  };

  rope_one(q, Hq);
  rope_one(k, Hkv);
}

Status CPUBackend::attention_f32(const Tensor& normed_hidden, const AttnWeights& w,
                                 const AttnConfig& attn, const ScaleConfig* scales,
                                 int max_seq_length, int hidden_size, LayerCacheView& cache,
                                 int64_t past_len, Tensor& out) const {
  if (Status s = require_f32(normed_hidden, "normed_hidden", Device::CPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::CPU); !s.ok())
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
  const int n_rep = Hq / Hkv;
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

  StatusOr<Tensor> Q = ::linear(normed_hidden, w.q_proj);
  StatusOr<Tensor> K = ::linear(normed_hidden, w.k_proj);
  StatusOr<Tensor> V = ::linear(normed_hidden, w.v_proj);
  if (!Q.ok())
    return Status(Q.status());
  if (!K.ok())
    return Status(K.status());
  if (!V.ok())
    return Status(V.status());

  if (Q.value().size() == 1)
    Q.value().shape = {1, Q.value().shape[0]};
  if (K.value().size() == 1)
    K.value().shape = {1, K.value().shape[0]};
  if (V.value().size() == 1)
    V.value().shape = {1, V.value().shape[0]};

  {
    auto kmat = as_mat_f32(K.value());
    kmat *= key_mult;
  }

  float* q_ptr = static_cast<float*>(Q.value().buffer.ptr);
  float* k_ptr = static_cast<float*>(K.value().buffer.ptr);
  float* v_ptr = static_cast<float*>(V.value().buffer.ptr);

  apply_rope_inplace_(q_ptr, k_ptr, T, Hq, Hkv, Dh, past_len, attn.rope_theta);

  // Append into cache: layout [S, Hkv*Dh]
  float* cache_k = static_cast<float*>(cache.k.ptr);
  float* cache_v = static_cast<float*>(cache.v.ptr);
  for (int64_t t = 0; t < T; ++t) {
    std::memcpy(kv_row_(cache_k, past_len + t, kv_dim), k_ptr + t * kv_dim,
                static_cast<size_t>(kv_dim) * sizeof(float));
    std::memcpy(kv_row_(cache_v, past_len + t, kv_dim), v_ptr + t * kv_dim,
                static_cast<size_t>(kv_dim) * sizeof(float));
  }

  const int64_t S = past_len + T;
  const float scale = 1.f / std::sqrt(static_cast<float>(Dh));

  StatusOr<Tensor> attn_out = allocate_f32_tensor({T, q_dim});
  if (!attn_out.ok())
    return Status(attn_out.status());
  float* out_ptr = static_cast<float*>(attn_out.value().buffer.ptr);

  std::vector<float> scores(static_cast<size_t>(S));
  std::vector<float> probs(static_cast<size_t>(S));

  for (int64_t t = 0; t < T; ++t) {
    const int64_t q_pos = past_len + t; // absolute position of this query
    for (int hq = 0; hq < Hq; ++hq) {
      const int hkv = hq / n_rep;
      const float* q_hd = q_ptr + (t * Hq + hq) * Dh;

      // scores over keys 0..q_pos
      float max_score = -INFINITY;
      for (int64_t s = 0; s <= q_pos; ++s) {
        const float* k_hd = kv_row_(cache_k, s, kv_dim) + hkv * Dh;
        float dot = 0.f;
        for (int d = 0; d < Dh; ++d)
          dot += q_hd[d] * k_hd[d];
        const float sc = dot * scale;
        scores[static_cast<size_t>(s)] = sc;
        if (sc > max_score)
          max_score = sc;
      }

      float sum = 0.f;
      for (int64_t s = 0; s <= q_pos; ++s) {
        const float e = std::exp(scores[static_cast<size_t>(s)] - max_score);
        probs[static_cast<size_t>(s)] = e;
        sum += e;
      }
      const float inv = 1.f / sum;
      for (int64_t s = 0; s <= q_pos; ++s)
        probs[static_cast<size_t>(s)] *= inv;

      float* o_hd = out_ptr + (t * Hq + hq) * Dh;
      for (int d = 0; d < Dh; ++d)
        o_hd[d] = 0.f;
      for (int64_t s = 0; s <= q_pos; ++s) {
        const float* v_hd = kv_row_(cache_v, s, kv_dim) + hkv * Dh;
        const float p = probs[static_cast<size_t>(s)];
        for (int d = 0; d < Dh; ++d)
          o_hd[d] += p * v_hd[d];
      }
    }
  }

  if (normed_hidden.size() == 1) {
    attn_out.value().shape = {q_dim};
    return linear(attn_out.value(), w.o_proj, out);
  }
  return linear(attn_out.value(), w.o_proj, out);
}

Status CPUBackend::mamba2_ssm_scan_f32_(const float* x, const float* B, const float* C,
                                        const float* dt, const float* A_log, const float* D,
                                        const float* dt_bias, float* state, float* y, int64_t T,
                                        const SsmConfig& ssm) const {
  const int H = ssm.n_heads;
  const int Dh = ssm.d_head;
  const int N = ssm.d_state;
  const int G = ssm.n_groups;
  const int heads_per_group = H / G;
  const int64_t intermediate = static_cast<int64_t>(H) * Dh;
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

Status CPUBackend::causal_conv1d_f32_(const float* x, int64_t T, int64_t conv_dim, int kernel,
                                      const float* weight, const float* bias, float* cache,
                                      float* y, bool apply_silu) const {
  if (T <= 0 || conv_dim <= 0 || kernel <= 1) {
    return Status::InvalidArgument("invalid causal_conv1d dims");
  }
  if (x == nullptr || weight == nullptr || cache == nullptr || y == nullptr) {
    return Status::InvalidArgument("null pointer in causal_conv1d");
  }

  const int K = kernel;
  const int left = K - 1;

  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < conv_dim; ++c) {
      float* crow = cache + static_cast<size_t>(c) * static_cast<size_t>(left);
      const float xt =
          x[static_cast<size_t>(t) * static_cast<size_t>(conv_dim) + static_cast<size_t>(c)];
      const float* wt = weight + static_cast<size_t>(c) * static_cast<size_t>(K);
      float acc = bias ? bias[c] : 0.f;
      for (int k = 0; k < left; ++k)
        acc += wt[k] * crow[k];
      acc += wt[left] * xt;
      if (apply_silu)
        acc = silu(acc);
      y[static_cast<size_t>(t) * static_cast<size_t>(conv_dim) + static_cast<size_t>(c)] = acc;

      for (int i = 0; i < left - 1; ++i)
        crow[i] = crow[i + 1];
      if (left > 0)
        crow[left - 1] = xt;
    }
  }
  return Status::Ok();
}

Status CPUBackend::rms_norm_gated_f32_(Tensor& y, const Tensor& gate, const Tensor& weight,
                                       float eps) const {
  if (Status s = require_f32(y, "y", Device::CPU); !s.ok())
    return s;
  if (Status s = require_f32(gate, "gate", Device::CPU); !s.ok())
    return s;
  if (Status s = require_f32(weight, "weight", Device::CPU); !s.ok())
    return s;
  if (!same_shape(y.shape, gate.shape)) {
    return Status::InvalidArgument("y and gate shape mismatch");
  }

  StatusOr<Tensor> gate_act = silu(gate);
  if (!gate_act.ok())
    return Status(gate_act.status());
  if (Status s = mul(y, gate_act.value(), y); !s.ok())
    return s;
  return rms_norm_inplace(y, weight, eps);
}

Status CPUBackend::mamba2_mixer_f32(const Tensor& normed_hidden, const Mamba2MixerWeights& w,
                                    const SsmConfig& ssm, const ScaleConfig* scales,
                                    float rms_norm_eps, LayerCacheView& cache, Tensor& out) const {
  if (Status s = require_f32(normed_hidden, "normed_hidden", Device::CPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::CPU); !s.ok())
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

  const int64_t I = ssm.d_inner;
  const int64_t G = ssm.n_groups;
  const int64_t N = ssm.d_state;
  const int64_t H = ssm.n_heads;
  const int64_t K = ssm.d_conv;
  const int64_t conv_dim = I + 2 * G * N;
  const int64_t proj_size = I + conv_dim + H;

  const float in_scale = scales ? scales->ssm_in : 1.f;
  const float mz = scales ? scales->ssm_chunk[0] : 1.f;
  const float mx = scales ? scales->ssm_chunk[1] : 1.f;
  const float mB = scales ? scales->ssm_chunk[2] : 1.f;
  const float mC = scales ? scales->ssm_chunk[3] : 1.f;
  const float mdt = scales ? scales->ssm_chunk[4] : 1.f;

  if (!same_shape(out.shape, normed_hidden.shape)) {
    return Status::InvalidArgument("out shape must match hidden");
  }
  if (w.in_proj.size() != 2 || w.in_proj.shape[0] != proj_size ||
      w.in_proj.shape[1] != hidden_dim) {
    return Status::InvalidArgument("in_proj must be [proj_size, hidden]");
  }
  if (cache.conv.ptr == nullptr || cache.ssm.ptr == nullptr) {
    return Status::InvalidArgument("cache view is null");
  }
  const size_t expect_conv =
      static_cast<size_t>(conv_dim) * static_cast<size_t>(K - 1) * sizeof(float);
  const size_t expect_ssm = static_cast<size_t>(H) * static_cast<size_t>(ssm.d_head) *
                            static_cast<size_t>(N) * sizeof(float);
  if (cache.conv.bytes < expect_conv || cache.ssm.bytes < expect_ssm) {
    return Status::InvalidArgument("cache view too small for mixer dims");
  }
  if (ssm.gated_rms_norm && w.mixer_norm == nullptr) {
    return Status::InvalidArgument("mixer_norm required when gated_rms_norm is true");
  }

  // Optional input scale
  StatusOr<Tensor> scaled_hidden = allocate_f32_tensor(normed_hidden.shape);
  if (!scaled_hidden.ok())
    return Status(scaled_hidden.status());
  if (in_scale != 1.f) {
    const auto src = as_vec_f32(normed_hidden);
    auto dst = as_vec_f32(scaled_hidden.value());
    dst = src * in_scale;
  } else {
    std::memcpy(scaled_hidden.value().buffer.ptr, normed_hidden.buffer.ptr,
                normed_hidden.buffer.bytes);
  }

  StatusOr<Tensor> projected = ::linear(scaled_hidden.value(), w.in_proj);
  if (!projected.ok())
    return Status(projected.status());

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
    const int64_t GN = G * N;
    const auto P = as_mat_f32(projected.value());
    auto g = as_mat_f32(gate.value());
    auto xbc = as_mat_f32(xBC.value());
    auto d = as_mat_f32(dt.value());
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t i = 0; i < I; ++i)
        g(t, i) = P(t, i) * mz;
      for (int64_t i = 0; i < I; ++i)
        xbc(t, i) = P(t, I + i) * mx;
      for (int64_t i = 0; i < GN; ++i)
        xbc(t, I + i) = P(t, 2 * I + i) * mB;
      for (int64_t i = 0; i < GN; ++i)
        xbc(t, I + GN + i) = P(t, 2 * I + GN + i) * mC;
      for (int64_t i = 0; i < H; ++i)
        d(t, i) = P(t, I + conv_dim + i) * mdt;
    }
  }

  if (w.conv1d.numel().ok() && w.conv1d.numel().value() != conv_dim * K) {
    return Status::InvalidArgument("conv1d weight numel mismatch");
  }
  const float* conv_w = static_cast<const float*>(w.conv1d.buffer.ptr);
  const float* conv_b =
      ssm.use_conv_bias ? static_cast<const float*>(w.conv1d_bias.buffer.ptr) : nullptr;

  StatusOr<Tensor> xBC_out = allocate_f32_tensor({T, conv_dim});
  if (!xBC_out.ok())
    return Status(xBC_out.status());

  if (Status s = causal_conv1d_f32_(static_cast<const float*>(xBC.value().buffer.ptr), T, conv_dim,
                                    K, conv_w, conv_b, static_cast<float*>(cache.conv.ptr),
                                    static_cast<float*>(xBC_out.value().buffer.ptr), /*silu=*/true);
      !s.ok()) {
    return s;
  }

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

  if (Status s = mamba2_ssm_scan_f32_(
          static_cast<const float*>(x.value().buffer.ptr),
          static_cast<const float*>(B.value().buffer.ptr),
          static_cast<const float*>(C.value().buffer.ptr),
          static_cast<const float*>(dt.value().buffer.ptr),
          static_cast<const float*>(w.A_log.buffer.ptr), static_cast<const float*>(w.D.buffer.ptr),
          static_cast<const float*>(w.dt_bias.buffer.ptr), static_cast<float*>(cache.ssm.ptr),
          static_cast<float*>(y.value().buffer.ptr), T, ssm);
      !s.ok()) {
    return s;
  }

  if (ssm.gated_rms_norm) {
    if (Status s = rms_norm_gated_f32_(y.value(), gate.value(), *w.mixer_norm, rms_norm_eps);
        !s.ok()) {
      return s;
    }
  } else {
    StatusOr<Tensor> gate_act = silu(gate.value());
    if (!gate_act.ok())
      return Status(gate_act.status());
    if (Status s = mul(y.value(), gate_act.value(), y.value()); !s.ok())
      return s;
  }

  if (w.out_proj.size() != 2 || w.out_proj.shape[0] != hidden_dim || w.out_proj.shape[1] != I) {
    return Status::InvalidArgument("out_proj must be [hidden, intermediate]");
  }

  if (normed_hidden.size() == 1) {
    y.value().shape = {I};
    return linear(y.value(), w.out_proj, out);
  }
  return linear(y.value(), w.out_proj, out);
}

Status CPUBackend::mlp_f32(const Tensor& normed_hidden, const MlpWeights& w, const MlpConfig& mlp,
                           const ScaleConfig* scales, Tensor& out) const {
  (void)mlp;
  if (Status s = require_f32(normed_hidden, "normed_hidden", Device::CPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::CPU); !s.ok())
    return s;
  if (!same_shape(out.shape, normed_hidden.shape))
    return Status::InvalidArgument("out shape must match hidden");

  const float gate_mult = scales ? scales->mlp[0] : 1.f;
  const float down_mult = scales ? scales->mlp[1] : 1.f;

  StatusOr<Tensor> up = ::linear(normed_hidden, w.up_proj);
  StatusOr<Tensor> gate = ::linear(normed_hidden, w.gate_proj);
  if (!up.ok())
    return Status(up.status());
  if (!gate.ok())
    return Status(gate.status());

  {
    auto g = as_vec_f32(gate.value());
    g *= gate_mult;
  }
  StatusOr<Tensor> gate_act = silu(gate.value());
  if (!gate_act.ok())
    return Status(gate_act.status());
  if (Status s = mul(up.value(), gate_act.value(), up.value()); !s.ok())
    return s;

  if (Status s = linear(up.value(), w.down_proj, out); !s.ok())
    return s;

  auto o = as_vec_f32(out);
  o *= down_mult;
  return Status::Ok();
}

Status CPUBackend::rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) const {
  return ::rms_norm(x, weight, eps, out);
}

Status CPUBackend::add(const Tensor& a, const Tensor& b, Tensor& out) const {
  return ::add(a, b, out);
}

Status CPUBackend::scale(Tensor& x, float s) const {
  if (Status st = require_f32(x, "x", Device::CPU); !st.ok())
    return st;
  as_vec_f32(x) *= s;
  return Status::Ok();
}

Status CPUBackend::linear(const Tensor& x, const Tensor& W, Tensor& out) const {
  return ::linear(x, W, out);
}

Status CPUBackend::embedding_lookup(const Tensor& table, const int32_t* tokens, int64_t n_tokens,
                                    float scale, Tensor& out) const {
  if (tokens == nullptr || n_tokens <= 0)
    return Status::InvalidArgument("embedding_lookup requires at least one token");
  if (Status s = require_f32(table, "table", Device::CPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::CPU); !s.ok())
    return s;
  if (table.size() != 2)
    return Status::InvalidArgument("embedding table must be rank 2 [vocab, hidden]");

  const int64_t vocab = table.shape[0];
  const int64_t hidden = table.shape[1];
  if (out.size() != 2 || out.shape[0] != n_tokens || out.shape[1] != hidden) {
    return Status::InvalidArgument("out shape must be [T, hidden]");
  }

  const auto emb = as_mat_f32(table);
  auto dest = as_mat_f32(out);
  for (int64_t i = 0; i < n_tokens; ++i) {
    const int32_t id = tokens[i];
    if (id < 0 || static_cast<int64_t>(id) >= vocab)
      return Status::InvalidArgument("token id out of range");
    dest.row(static_cast<Eigen::Index>(i)) =
        emb.row(static_cast<Eigen::Index>(id)) * scale;
  }
  return Status::Ok();
}

Status CPUBackend::take_last_row(const Tensor& hidden, Tensor& out) const {
  if (Status s = require_f32(hidden, "hidden", Device::CPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::CPU); !s.ok())
    return s;
  if (hidden.empty())
    return Status::InvalidArgument("hidden is empty");

  if (hidden.size() == 1) {
    if (!same_shape(out.shape, hidden.shape))
      return Status::InvalidArgument("out shape must match rank-1 hidden");
    std::memcpy(out.buffer.ptr, hidden.buffer.ptr, hidden.buffer.bytes);
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

  const auto src = as_mat_f32(hidden);
  auto dst = as_vec_f32(out);
  dst = src.row(rows - 1);
  return Status::Ok();
}