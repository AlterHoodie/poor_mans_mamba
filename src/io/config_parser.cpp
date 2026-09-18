#include "io/config_parser.h"

#include "core/status.h"
#include "io/utils.h"

#include <vector>

namespace {

Status parse_common_fields(const json& config_json, ModelConfig& cfg) {
  ASSIGN_OR_RETURN(cfg.model_type, json_at<std::string>(config_json, "model_type"));
  ASSIGN_OR_RETURN(cfg.hidden_size, json_at<int>(config_json, "hidden_size"));
  ASSIGN_OR_RETURN(cfg.vocab_size, json_at<int>(config_json, "vocab_size"));
  ASSIGN_OR_RETURN(cfg.tie_word_embeddings, json_at<bool>(config_json, "tie_word_embeddings"));
  ASSIGN_OR_RETURN(cfg.num_hidden_layers, json_at<int>(config_json, "num_hidden_layers"));
  ASSIGN_OR_RETURN(cfg.bos_token_id, json_at<int>(config_json, "bos_token_id"));
  ASSIGN_OR_RETURN(cfg.eos_token_id, json_at<int>(config_json, "eos_token_id"));
  ASSIGN_OR_RETURN(cfg.pad_token_id, json_at<int>(config_json, "pad_token_id"));
  return Status::Ok();
}

Status validate_ssm(const SsmConfig& ssm) {
  if (ssm.n_heads <= 0 || ssm.d_head <= 0 || ssm.d_state <= 0 || ssm.d_conv <= 1 ||
      ssm.d_inner <= 0 || ssm.n_groups <= 0) {
    return Status::InvalidArgument("invalid ssm dimensions");
  }
  if (ssm.n_heads % ssm.n_groups != 0) {
    return Status::InvalidArgument("n_heads must be divisible by n_groups");
  }
  if (ssm.d_inner != ssm.n_heads * ssm.d_head) {
    return Status::InvalidArgument("d_inner must equal n_heads * d_head");
  }
  return Status::Ok();
}

} // namespace

StatusOr<std::unique_ptr<ModelConfig>> Mamba2ConfigParser::parse(const std::string& model_dir,
                                                                 int max_seq_length) const {
  if (max_seq_length <= 0) {
    return Status::InvalidArgument("max_seq_length must be positive");
  }

  const std::string config_path = model_dir + "/config.json";
  auto config_json_status = json_load(config_path);
  if (!config_json_status.ok()) {
    return Status(config_json_status.status());
  }

  json config_json = config_json_status.value();
  auto cfg = std::make_unique<ModelConfig>();
  cfg->layout = ArchLayout::MambaOnly;

  if (Status s = parse_common_fields(config_json, *cfg); !s.ok())
    return s;
  cfg->max_seq_length = max_seq_length;

  int expand = 0;
  ASSIGN_OR_RETURN(expand, json_at<int>(config_json, "expand"));
  ASSIGN_OR_RETURN(cfg->ssm.d_conv, json_at<int>(config_json, "conv_kernel"));
  ASSIGN_OR_RETURN(cfg->ssm.d_state, json_at<int>(config_json, "state_size"));
  ASSIGN_OR_RETURN(cfg->ssm.d_head, json_at<int>(config_json, "head_dim"));
  ASSIGN_OR_RETURN(cfg->ssm.n_heads, json_at<int>(config_json, "num_heads"));
  ASSIGN_OR_RETURN(cfg->ssm.n_groups, json_at<int>(config_json, "n_groups"));
  ASSIGN_OR_RETURN(cfg->ssm.chunk_size, json_at<int>(config_json, "chunk_size"));
  ASSIGN_OR_RETURN(cfg->ssm.use_conv_bias, json_at<bool>(config_json, "use_conv_bias"));
  ASSIGN_OR_RETURN(cfg->ssm.use_proj_bias, json_at<bool>(config_json, "use_bias"));
  ASSIGN_OR_RETURN(cfg->rms_norm_eps, json_at<float>(config_json, "layer_norm_epsilon"));
  ASSIGN_OR_RETURN(cfg->hidden_act, json_at<std::string>(config_json, "hidden_act"));

  cfg->ssm.d_inner = expand * cfg->hidden_size;
  cfg->ssm.gated_rms_norm = true;

  if (Status s = validate_ssm(cfg->ssm); !s.ok())
    return s;

  return cfg;
}

StatusOr<std::unique_ptr<ModelConfig>> FalconH1ConfigParser::parse(const std::string& model_dir,
                                                                   int max_seq_length) const {
  if (max_seq_length <= 0) {
    return Status::InvalidArgument("max_seq_length must be positive");
  }

  const std::string config_path = model_dir + "/config.json";
  auto config_json_status = json_load(config_path);
  if (!config_json_status.ok()) {
    return Status(config_json_status.status());
  }

  json config_json = config_json_status.value();
  auto cfg = std::make_unique<ModelConfig>();
  cfg->layout = ArchLayout::ParallelHybrid;

  if (Status s = parse_common_fields(config_json, *cfg); !s.ok())
    return s;
  cfg->max_seq_length = max_seq_length;

  if (cfg->model_type != "falcon_h1") {
    return Status::InvalidArgument("expected model_type falcon_h1, got " + cfg->model_type);
  }

  AttnConfig attn{};
  ASSIGN_OR_RETURN(attn.n_q_heads, json_at<int>(config_json, "num_attention_heads"));
  ASSIGN_OR_RETURN(attn.n_kv_heads, json_at<int>(config_json, "num_key_value_heads"));
  ASSIGN_OR_RETURN(attn.head_dim, json_at<int>(config_json, "head_dim"));
  ASSIGN_OR_RETURN(attn.bias, json_at<bool>(config_json, "attention_bias"));
  ASSIGN_OR_RETURN(attn.dropout, json_at<float>(config_json, "attention_dropout"));
  ASSIGN_OR_RETURN(attn.rope_theta, json_at<float>(config_json, "rope_theta"));
  cfg->attn = attn;

  ASSIGN_OR_RETURN(cfg->ssm.n_heads, json_at<int>(config_json, "mamba_n_heads"));
  ASSIGN_OR_RETURN(cfg->ssm.d_head, json_at<int>(config_json, "mamba_d_head"));
  ASSIGN_OR_RETURN(cfg->ssm.d_state, json_at<int>(config_json, "mamba_d_state"));
  ASSIGN_OR_RETURN(cfg->ssm.d_conv, json_at<int>(config_json, "mamba_d_conv"));
  ASSIGN_OR_RETURN(cfg->ssm.d_inner, json_at<int>(config_json, "mamba_d_ssm"));
  ASSIGN_OR_RETURN(cfg->ssm.n_groups, json_at<int>(config_json, "mamba_n_groups"));
  ASSIGN_OR_RETURN(cfg->ssm.chunk_size, json_at<int>(config_json, "mamba_chunk_size"));
  ASSIGN_OR_RETURN(cfg->ssm.use_conv_bias, json_at<bool>(config_json, "mamba_conv_bias"));
  ASSIGN_OR_RETURN(cfg->ssm.use_proj_bias, json_at<bool>(config_json, "mamba_proj_bias"));
  ASSIGN_OR_RETURN(cfg->ssm.gated_rms_norm, json_at<bool>(config_json, "mamba_rms_norm"));

  ScaleConfig scales{};
  std::vector<float> ssm_mult;
  ASSIGN_OR_RETURN(ssm_mult, json_at<std::vector<float>>(config_json, "ssm_multipliers"));
  if (ssm_mult.size() != 5) {
    return Status::InvalidArgument("ssm_multipliers must have length 5 [z,x,B,C,dt]");
  }
  for (size_t i = 0; i < 5; ++i)
    scales.ssm_chunk[i] = ssm_mult[i];

  ASSIGN_OR_RETURN(scales.ssm_in, json_at<float>(config_json, "ssm_in_multiplier"));
  ASSIGN_OR_RETURN(scales.ssm_out, json_at<float>(config_json, "ssm_out_multiplier"));
  ASSIGN_OR_RETURN(scales.attn_in, json_at<float>(config_json, "attention_in_multiplier"));
  ASSIGN_OR_RETURN(scales.attn_out, json_at<float>(config_json, "attention_out_multiplier"));
  ASSIGN_OR_RETURN(scales.key, json_at<float>(config_json, "key_multiplier"));
  ASSIGN_OR_RETURN(scales.embedding, json_at<float>(config_json, "embedding_multiplier"));
  ASSIGN_OR_RETURN(scales.lm_head, json_at<float>(config_json, "lm_head_multiplier"));

  std::vector<float> mlp_mult;
  ASSIGN_OR_RETURN(mlp_mult, json_at<std::vector<float>>(config_json, "mlp_multipliers"));
  if (mlp_mult.size() != 2) {
    return Status::InvalidArgument("mlp_multipliers must have length 2");
  }
  scales.mlp[0] = mlp_mult[0];
  scales.mlp[1] = mlp_mult[1];
  cfg->scales = scales;

  MlpConfig mlp{};
  ASSIGN_OR_RETURN(mlp.intermediate_size, json_at<int>(config_json, "intermediate_size"));
  ASSIGN_OR_RETURN(mlp.bias, json_at<bool>(config_json, "mlp_bias"));
  cfg->mlp = mlp;

  ASSIGN_OR_RETURN(cfg->rms_norm_eps, json_at<float>(config_json, "rms_norm_eps"));
  ASSIGN_OR_RETURN(cfg->hidden_act, json_at<std::string>(config_json, "hidden_act"));

  if (Status s = validate_ssm(cfg->ssm); !s.ok())
    return s;

  return cfg;
}
