#pragma once

#include <string>

#include "status.h"

template <typename ConfigT>
class ConfigParser {
   public:
    virtual ~ConfigParser() = default;
    virtual StatusOr<ConfigT> parse(const std::string& model_dir) const = 0;
};

struct Mamba2Config {
    // model identity
    std::string model_type;

    // embedding
    int hidden_size;
    int vocab_size;
    bool tie_word_embeddings = true;

    // Number of layers - backbone depth
    int num_hidden_layers;

    // Mamba2 SSM Mixer
    int expand;
    int conv_kernel;
    int state_size;
    int head_dim;
    int num_heads;
    int n_groups;
    int chunk_size;
    int time_step_rank;

    // timesteps
    float time_step_max;
    float time_step_min;
    float time_step_floor;
    // time_step_limit

    // Norm & activations
    bool rms_norm = true;
    float layer_norm_epsilon;
    std::string hiddent_act;
    bool residual_in_fp32 = true;
    bool rescale_prenorm_residual = false;

    // linear & conv bias flags
    bool use_bias = false;
    bool use_conv_bias = true;

    // others
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    float initializer_range;
    std::string transformers_version;
    bool use_cache = true;
};

class Mamba2ConfigParser : public ConfigParser<Mamba2Config> {
   public:
    StatusOr<Mamba2Config> parse(const std::string& model_dir) const override;
};