#pragma once

#include "core/status.h"

#include <array>
#include <optional>
#include <string>

enum class ArchLayout { MambaOnly, ParallelHybrid };

struct SsmConfig {
  int n_heads = 0;
  int d_head = 0;
  int d_state = 0;
  int d_conv = 0;
  int d_inner = 0; // intermediate SSM width
  int n_groups = 0;
  int chunk_size = 0;
  bool use_conv_bias = true;
  bool use_proj_bias = false;
  bool gated_rms_norm = true; // true: rms(y*silu(z)); false: y*silu(z)
};

struct AttnConfig {
  int n_q_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  bool bias = false;
  float dropout = 0.f;
  float rope_theta = 10000.f;
};

struct MlpConfig {
  int intermediate_size = 0;
  bool bias = false;
};

struct ScaleConfig {
  std::array<float, 5> ssm_chunk{{1.f, 1.f, 1.f, 1.f, 1.f}}; // z, x, B, C, dt
  float ssm_in = 1.f;
  float ssm_out = 1.f;
  float attn_in = 1.f;
  float attn_out = 1.f;
  float key = 1.f;
  std::array<float, 2> mlp{{1.f, 1.f}}; // gate/up, down
  float embedding = 1.f;
  float lm_head = 1.f;
};

struct ModelConfig {
  ArchLayout layout = ArchLayout::MambaOnly;
  std::string model_type;

  int hidden_size = 0;
  int vocab_size = 0;
  bool tie_word_embeddings = false;
  int num_hidden_layers = 0;
  int max_seq_length = 0;

  int bos_token_id = 0;
  int eos_token_id = -1;
  int pad_token_id = 0;

  float rms_norm_eps = 1e-5f;
  std::string hidden_act;

  SsmConfig ssm;
  std::optional<AttnConfig> attn;
  std::optional<MlpConfig> mlp;
  std::optional<ScaleConfig> scales;
};

// Reads config.json model_type without exposing JSON types to callers.
StatusOr<std::string> read_model_type(const std::string& model_dir);
