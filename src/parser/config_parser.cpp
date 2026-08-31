#include "config/config_parser.h"

#include <fstream>

#include "config/utils.h"
#include "status.h"

StatusOr<Mamba2Config> Mamba2ConfigParser::parse(const std::string &model_dir) const {
    const std::string config_path = model_dir + "/config.json";

    auto config_json_status = json_load(config_path);

    if (!config_json_status.ok()) {
        return Status(config_json_status.status());
    }

    json config_json = config_json_status.value();
    Mamba2Config cfg{};

    ASSIGN_OR_RETURN(cfg.model_type, json_at<std::string>(config_json, "model_type"));
    ASSIGN_OR_RETURN(cfg.hidden_size, json_at<int>(config_json, "hidden_size"));
    ASSIGN_OR_RETURN(cfg.vocab_size, json_at<int>(config_json, "vocab_size"));
    ASSIGN_OR_RETURN(cfg.tie_word_embeddings, json_at<bool>(config_json, "tie_word_embeddings"));
    ASSIGN_OR_RETURN(cfg.num_hidden_layers, json_at<int>(config_json, "num_hidden_layers"));
    ASSIGN_OR_RETURN(cfg.expand, json_at<int>(config_json, "expand"));
    ASSIGN_OR_RETURN(cfg.conv_kernel, json_at<int>(config_json, "conv_kernel"));
    ASSIGN_OR_RETURN(cfg.state_size, json_at<int>(config_json, "state_size"));
    ASSIGN_OR_RETURN(cfg.head_dim, json_at<int>(config_json, "head_dim"));
    ASSIGN_OR_RETURN(cfg.num_heads, json_at<int>(config_json, "num_heads"));
    ASSIGN_OR_RETURN(cfg.n_groups, json_at<int>(config_json, "n_groups"));
    ASSIGN_OR_RETURN(cfg.chunk_size, json_at<int>(config_json, "chunk_size"));
    ASSIGN_OR_RETURN(cfg.time_step_rank, json_at<int>(config_json, "time_step_rank"));
    ASSIGN_OR_RETURN(cfg.time_step_max, json_at<float>(config_json, "time_step_max"));
    ASSIGN_OR_RETURN(cfg.time_step_min, json_at<float>(config_json, "time_step_min"));
    ASSIGN_OR_RETURN(cfg.time_step_floor, json_at<float>(config_json, "time_step_floor"));
    ASSIGN_OR_RETURN(cfg.rms_norm, json_at<bool>(config_json, "rms_norm"));
    ASSIGN_OR_RETURN(cfg.layer_norm_epsilon, json_at<uint32_t>(config_json, "layer_norm_epsilon"));
    ASSIGN_OR_RETURN(cfg.hiddent_act, json_at<std::string>(config_json, "hidden_act"));
    ASSIGN_OR_RETURN(cfg.residual_in_fp32, json_at<bool>(config_json, "residual_in_fp32"));
    ASSIGN_OR_RETURN(cfg.rescale_prenorm_residual,
                     json_at<bool>(config_json, "rescale_prenorm_residual"));
    ASSIGN_OR_RETURN(cfg.use_bias, json_at<bool>(config_json, "use_bias"));
    ASSIGN_OR_RETURN(cfg.use_conv_bias, json_at<bool>(config_json, "use_conv_bias"));
    ASSIGN_OR_RETURN(cfg.bos_token_id, json_at<int>(config_json, "bos_token_id"));
    ASSIGN_OR_RETURN(cfg.eos_token_id, json_at<int>(config_json, "eos_token_id"));
    ASSIGN_OR_RETURN(cfg.pad_token_id, json_at<int>(config_json, "pad_token_id"));
    ASSIGN_OR_RETURN(cfg.initializer_range, json_at<float>(config_json, "initializer_range"));
    ASSIGN_OR_RETURN(cfg.transformers_version,
                     json_at<std::string>(config_json, "transformers_version"));
    ASSIGN_OR_RETURN(cfg.use_cache, json_at<bool>(config_json, "use_cache"));

    return cfg;
}