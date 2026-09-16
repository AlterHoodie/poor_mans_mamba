#include "io/config_parser.h"

#include "core/status.h"
#include "io/utils.h"

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
  ASSIGN_OR_RETURN(cfg.transformers_version,
                   json_at<std::string>(config_json, "transformers_version"));
  ASSIGN_OR_RETURN(cfg.use_cache, json_at<bool>(config_json, "use_cache"));
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
  auto cfg = std::make_unique<Mamba2Config>();

  if (Status s = parse_common_fields(config_json, *cfg); !s.ok())
    return s;
  cfg->max_seq_length = max_seq_length;

  ASSIGN_OR_RETURN(cfg->expand, json_at<int>(config_json, "expand"));
  ASSIGN_OR_RETURN(cfg->conv_kernel, json_at<int>(config_json, "conv_kernel"));
  ASSIGN_OR_RETURN(cfg->state_size, json_at<int>(config_json, "state_size"));
  ASSIGN_OR_RETURN(cfg->head_dim, json_at<int>(config_json, "head_dim"));
  ASSIGN_OR_RETURN(cfg->num_heads, json_at<int>(config_json, "num_heads"));
  ASSIGN_OR_RETURN(cfg->n_groups, json_at<int>(config_json, "n_groups"));
  ASSIGN_OR_RETURN(cfg->chunk_size, json_at<int>(config_json, "chunk_size"));
  ASSIGN_OR_RETURN(cfg->time_step_rank, json_at<int>(config_json, "time_step_rank"));
  ASSIGN_OR_RETURN(cfg->time_step_max, json_at<float>(config_json, "time_step_max"));
  ASSIGN_OR_RETURN(cfg->time_step_min, json_at<float>(config_json, "time_step_min"));
  ASSIGN_OR_RETURN(cfg->time_step_floor, json_at<float>(config_json, "time_step_floor"));
  ASSIGN_OR_RETURN(cfg->rms_norm, json_at<bool>(config_json, "rms_norm"));
  ASSIGN_OR_RETURN(cfg->layer_norm_epsilon, json_at<float>(config_json, "layer_norm_epsilon"));
  ASSIGN_OR_RETURN(cfg->hiddent_act, json_at<std::string>(config_json, "hidden_act"));
  ASSIGN_OR_RETURN(cfg->residual_in_fp32, json_at<bool>(config_json, "residual_in_fp32"));
  ASSIGN_OR_RETURN(cfg->rescale_prenorm_residual,
                   json_at<bool>(config_json, "rescale_prenorm_residual"));
  ASSIGN_OR_RETURN(cfg->use_bias, json_at<bool>(config_json, "use_bias"));
  ASSIGN_OR_RETURN(cfg->use_conv_bias, json_at<bool>(config_json, "use_conv_bias"));
  ASSIGN_OR_RETURN(cfg->initializer_range, json_at<float>(config_json, "initializer_range"));

  return std::unique_ptr<ModelConfig>(std::move(cfg));
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
  auto cfg = std::make_unique<FalconH1Config>();

  if (Status s = parse_common_fields(config_json, *cfg); !s.ok())
    return s;
  cfg->max_seq_length = max_seq_length;

  // Attention
  ASSIGN_OR_RETURN(cfg->num_attention_heads, json_at<int>(config_json, "num_attention_heads"));
  ASSIGN_OR_RETURN(cfg->num_key_value_heads, json_at<int>(config_json, "num_key_value_heads"));
  ASSIGN_OR_RETURN(cfg->head_dim, json_at<int>(config_json, "head_dim"));
  ASSIGN_OR_RETURN(cfg->attention_bias, json_at<bool>(config_json, "attention_bias"));
  ASSIGN_OR_RETURN(cfg->attention_dropout, json_at<float>(config_json, "attention_dropout"));
  ASSIGN_OR_RETURN(cfg->attention_in_multiplier,
                   json_at<float>(config_json, "attention_in_multiplier"));
  ASSIGN_OR_RETURN(cfg->attention_out_multiplier,
                   json_at<float>(config_json, "attention_out_multiplier"));
  ASSIGN_OR_RETURN(cfg->key_multiplier, json_at<float>(config_json, "key_multiplier"));

  // Mamba / SSM
  ASSIGN_OR_RETURN(cfg->mamba_n_heads, json_at<int>(config_json, "mamba_n_heads"));
  ASSIGN_OR_RETURN(cfg->mamba_d_head, json_at<int>(config_json, "mamba_d_head"));
  ASSIGN_OR_RETURN(cfg->mamba_d_state, json_at<int>(config_json, "mamba_d_state"));
  ASSIGN_OR_RETURN(cfg->mamba_d_conv, json_at<int>(config_json, "mamba_d_conv"));
  ASSIGN_OR_RETURN(cfg->mamba_d_ssm, json_at<int>(config_json, "mamba_d_ssm"));
  ASSIGN_OR_RETURN(cfg->mamba_expand, json_at<int>(config_json, "mamba_expand"));
  ASSIGN_OR_RETURN(cfg->mamba_n_groups, json_at<int>(config_json, "mamba_n_groups"));
  ASSIGN_OR_RETURN(cfg->mamba_chunk_size, json_at<int>(config_json, "mamba_chunk_size"));
  ASSIGN_OR_RETURN(cfg->mamba_conv_bias, json_at<bool>(config_json, "mamba_conv_bias"));
  ASSIGN_OR_RETURN(cfg->mamba_proj_bias, json_at<bool>(config_json, "mamba_proj_bias"));
  ASSIGN_OR_RETURN(cfg->mamba_rms_norm, json_at<bool>(config_json, "mamba_rms_norm"));
  ASSIGN_OR_RETURN(cfg->mamba_norm_before_gate,
                   json_at<bool>(config_json, "mamba_norm_before_gate"));
  ASSIGN_OR_RETURN(cfg->mamba_use_mlp, json_at<bool>(config_json, "mamba_use_mlp"));

  ASSIGN_OR_RETURN(cfg->ssm_multipliers,
                   json_at<std::vector<float>>(config_json, "ssm_multipliers"));
  ASSIGN_OR_RETURN(cfg->ssm_in_multiplier, json_at<float>(config_json, "ssm_in_multiplier"));
  ASSIGN_OR_RETURN(cfg->ssm_out_multiplier, json_at<float>(config_json, "ssm_out_multiplier"));

  // MLP
  ASSIGN_OR_RETURN(cfg->intermediate_size, json_at<int>(config_json, "intermediate_size"));
  ASSIGN_OR_RETURN(cfg->mlp_expansion_factor, json_at<int>(config_json, "mlp_expansion_factor"));
  ASSIGN_OR_RETURN(cfg->mlp_bias, json_at<bool>(config_json, "mlp_bias"));
  ASSIGN_OR_RETURN(cfg->mlp_multipliers,
                   json_at<std::vector<float>>(config_json, "mlp_multipliers"));

  // Norm / act / RoPE / head
  ASSIGN_OR_RETURN(cfg->rms_norm_eps, json_at<float>(config_json, "rms_norm_eps"));
  ASSIGN_OR_RETURN(cfg->hidden_act, json_at<std::string>(config_json, "hidden_act"));
  ASSIGN_OR_RETURN(cfg->rope_theta, json_at<float>(config_json, "rope_theta"));
  ASSIGN_OR_RETURN(cfg->max_position_embeddings,
                   json_at<int>(config_json, "max_position_embeddings"));
  ASSIGN_OR_RETURN(cfg->num_logits_to_keep, json_at<int>(config_json, "num_logits_to_keep"));
  ASSIGN_OR_RETURN(cfg->projectors_bias, json_at<bool>(config_json, "projectors_bias"));

  // Misc
  ASSIGN_OR_RETURN(cfg->embedding_multiplier, json_at<float>(config_json, "embedding_multiplier"));
  ASSIGN_OR_RETURN(cfg->lm_head_multiplier, json_at<float>(config_json, "lm_head_multiplier"));
  ASSIGN_OR_RETURN(cfg->initializer_range, json_at<float>(config_json, "initializer_range"));
  ASSIGN_OR_RETURN(cfg->torch_dtype, json_at<std::string>(config_json, "torch_dtype"));

  if (cfg->ssm_multipliers.size() != 5) {
    return Status::InvalidArgument("ssm_multipliers must have length 5 [z,x,B,C,dt]");
  }
  if (cfg->mlp_multipliers.size() != 2) {
    return Status::InvalidArgument("mlp_multipliers must have length 2");
  }
  if (cfg->model_type != "falcon_h1") {
    return Status::InvalidArgument("expected model_type falcon_h1, got " + cfg->model_type);
  }

  return std::unique_ptr<ModelConfig>(std::move(cfg));
}
