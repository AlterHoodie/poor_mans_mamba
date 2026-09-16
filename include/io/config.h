#pragma once

#include "core/status.h"

#include <string>
#include <vector>

enum class ModelKind { kMamba2, kFalconH1 };

class ModelConfig {
public:
  virtual ~ModelConfig() = default;

  virtual ModelKind kind() const = 0;

  // model identity
  std::string model_type;

  // embedding
  int hidden_size = 0;
  int vocab_size = 0;
  bool tie_word_embeddings = false;

  // number of layers
  int num_hidden_layers = 0;

  // Runtime sequence limit used for cache sizing.
  int max_seq_length = 0;

  // tokens
  int bos_token_id = 0;
  int eos_token_id = -1;
  int pad_token_id = 0;

  // others
  std::string transformers_version;
  bool use_cache = true;
};

class Mamba2Config final : public ModelConfig {
public:
  ModelKind kind() const override;

  // Mamba2 SSM Mixer
  int expand = 0;
  int conv_kernel = 0;
  int state_size = 0;
  int head_dim = 0;
  int num_heads = 0;
  int n_groups = 0;
  int chunk_size = 0;
  int time_step_rank = 0;

  // timesteps
  float time_step_max = 0.f;
  float time_step_min = 0.f;
  float time_step_floor = 0.f;

  // Norm & activations
  bool rms_norm = true;
  float layer_norm_epsilon = 0.f;
  std::string hiddent_act;
  bool residual_in_fp32 = true;
  bool rescale_prenorm_residual = false;

  // linear & conv bias flags
  bool use_bias = false;
  bool use_conv_bias = true;

  // others
  float initializer_range = 0.f;
};

class FalconH1Config final : public ModelConfig {
public:
  ModelKind kind() const override;

  // Attention
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  bool attention_bias = false;
  float attention_dropout = 0.f;
  float attention_in_multiplier = 1.f;
  float attention_out_multiplier = 1.f;
  float key_multiplier = 1.f;

  // Mamba / SSM
  int mamba_n_heads = 0; // number of heads
  int mamba_d_head = 0;  // number of channels for each head
  int mamba_d_state = 0; // how many state dimensions each channel has
  int mamba_d_conv = 0;
  int mamba_d_ssm = 0; // intermediate SSM width; may differ from expand*hidden
  int mamba_expand = 0;
  int mamba_n_groups = 0;
  int mamba_chunk_size = 0;
  bool mamba_conv_bias = true;
  bool mamba_proj_bias = false;
  bool mamba_rms_norm = false;
  bool mamba_norm_before_gate = false;
  bool mamba_use_mlp = true;

  // μP scales: [z, x, B, C, dt] — same for every layer
  std::vector<float> ssm_multipliers;
  float ssm_in_multiplier = 1.f;
  float ssm_out_multiplier = 1.f;

  // MLP
  int intermediate_size = 0;
  int mlp_expansion_factor = 0;
  bool mlp_bias = false;
  std::vector<float> mlp_multipliers; // typically [gate/up, down]

  // Norm / act / RoPE / head
  float rms_norm_eps = 1e-5f;
  std::string hidden_act;
  float rope_theta = 10000.f;
  int max_position_embeddings = 0;
  int num_logits_to_keep = 1;
  bool projectors_bias = false;

  // Misc scales
  float embedding_multiplier = 1.f;
  float lm_head_multiplier = 1.f;
  float initializer_range = 0.02f;
  std::string torch_dtype;
};

// Reads config.json model_type without exposing JSON types to callers.
StatusOr<std::string> read_model_type(const std::string& model_dir);
