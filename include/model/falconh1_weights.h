#pragma once

#include "core/tensor.h"

#include <optional>
#include <vector>

struct FalconH1LayerWeights {
  // feed_forward
  Tensor ff_up_proj;
  Tensor ff_gate_proj;
  Tensor ff_down_proj;

  // Mamba
  Tensor in_proj;
  Tensor conv1d;
  Tensor conv1d_bias;
  Tensor dt_bias;
  Tensor A_log;
  Tensor out_proj;
  Tensor D;

  // norms
  Tensor input_layernorm;
  Tensor pre_ff_layer_norm;

  // Attention
  Tensor q_proj;
  Tensor k_proj;
  Tensor v_proj;
  Tensor o_proj;
};

struct FalconH1Weights {
  Tensor embeddings;
  std::vector<FalconH1LayerWeights> layers;
  Tensor norm_f;
  std::optional<Tensor> lm_head;
};