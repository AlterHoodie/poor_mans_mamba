#include "model/mamba2_loader.h"

#include "safetensors.hh"

#include <cstring>
#include <string>
#include <utility>

Status Mamba2ModelLoader::load(const ModelConfig& cfg, Mamba2Weights& weights,
                               const std::string& model_dir, int device_id) {
  safetensors::safetensors_t st{};
  std::string err;
  if (!safetensors::mmap_from_file(model_dir + "/model.safetensors", &st, nullptr, &err)) {
    return Status::InvalidArgument(err);
  }

  weights = Mamba2Weights{};
  // Resize layers to models number of layers
  weights.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));

  COPY_OR_RETURN(weights.embeddings, "backbone.embeddings.weight");
  COPY_OR_RETURN(weights.norm_f, "backbone.norm_f.weight");

  if (cfg.tie_word_embeddings) {
    weights.lm_head = std::nullopt;
  } else {
    weights.lm_head = Tensor{};
    COPY_OR_RETURN(*weights.lm_head, "lm_head.weight");
  }

  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    Mamba2LayerWeights& layer = weights.layers[static_cast<size_t>(i)];
    COPY_OR_RETURN(layer.in_proj, layer_key(i, ".mixer.in_proj.weight"));
    COPY_OR_RETURN(layer.conv1d, layer_key(i, ".mixer.conv1d.weight"));
    COPY_OR_RETURN(layer.conv1d_bias, layer_key(i, ".mixer.conv1d.bias"));
    COPY_OR_RETURN(layer.dt_bias, layer_key(i, ".mixer.dt_bias"));
    COPY_OR_RETURN(layer.out_proj, layer_key(i, ".mixer.out_proj.weight"));
    COPY_OR_RETURN(layer.A_log, layer_key(i, ".mixer.A_log"));
    COPY_OR_RETURN(layer.D, layer_key(i, ".mixer.D"));
    COPY_OR_RETURN(layer.mixer_norm, layer_key(i, ".mixer.norm.weight"));
    COPY_OR_RETURN(layer.layer_norm, layer_key(i, ".norm.weight"));
  }

  return Status::Ok();
}
