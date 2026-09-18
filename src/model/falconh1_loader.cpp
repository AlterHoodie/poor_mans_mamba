#include "model/falconh1_loader.h"

#include "safetensors.hh"

#include <string>

namespace {

std::string falcon_layer_key(int layer, const char* suffix) {
  return "model.layers." + std::to_string(layer) + suffix;
}

} // namespace

Status FalconH1ModelLoader::load(const ModelConfig& cfg, FalconH1Weights& weights,
                                 const std::string& model_dir, int device_id) {
  safetensors::safetensors_t st{};

  std::string err;
  if (!safetensors::mmap_from_file(model_dir + "/model.safetensors", &st, nullptr, &err)) {
    return Status::InvalidArgument(err);
  }

  weights = FalconH1Weights{};
  weights.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));

  COPY_OR_RETURN(weights.embeddings, "model.embed_tokens.weight");
  COPY_OR_RETURN(weights.norm_f, "model.final_layernorm.weight");

  if (cfg.tie_word_embeddings) {
    weights.lm_head = std::nullopt;
  } else {
    weights.lm_head = Tensor{};
    COPY_OR_RETURN(*weights.lm_head, "lm_head.weight");
  }

  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    FalconH1LayerWeights& layer = weights.layers[static_cast<size_t>(i)];

    COPY_OR_RETURN(layer.ff_up_proj, falcon_layer_key(i, ".feed_forward.up_proj.weight"));
    COPY_OR_RETURN(layer.ff_gate_proj, falcon_layer_key(i, ".feed_forward.gate_proj.weight"));
    COPY_OR_RETURN(layer.ff_down_proj, falcon_layer_key(i, ".feed_forward.down_proj.weight"));

    COPY_OR_RETURN(layer.A_log, falcon_layer_key(i, ".mamba.A_log"));
    COPY_OR_RETURN(layer.conv1d, falcon_layer_key(i, ".mamba.conv1d.weight"));
    COPY_OR_RETURN(layer.conv1d_bias, falcon_layer_key(i, ".mamba.conv1d.bias"));
    COPY_OR_RETURN(layer.D, falcon_layer_key(i, ".mamba.D"));
    COPY_OR_RETURN(layer.dt_bias, falcon_layer_key(i, ".mamba.dt_bias"));
    COPY_OR_RETURN(layer.in_proj, falcon_layer_key(i, ".mamba.in_proj.weight"));
    COPY_OR_RETURN(layer.out_proj, falcon_layer_key(i, ".mamba.out_proj.weight"));

    COPY_OR_RETURN(layer.input_layernorm, falcon_layer_key(i, ".input_layernorm.weight"));
    COPY_OR_RETURN(layer.pre_ff_layer_norm, falcon_layer_key(i, ".pre_ff_layernorm.weight"));

    COPY_OR_RETURN(layer.q_proj, falcon_layer_key(i, ".self_attn.q_proj.weight"));
    COPY_OR_RETURN(layer.k_proj, falcon_layer_key(i, ".self_attn.k_proj.weight"));
    COPY_OR_RETURN(layer.v_proj, falcon_layer_key(i, ".self_attn.v_proj.weight"));
    COPY_OR_RETURN(layer.o_proj, falcon_layer_key(i, ".self_attn.o_proj.weight"));
  }

  return Status::Ok();
}
