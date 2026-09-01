#pragma once

#include <optional>
#include <vector>

#include "core/tensor.h"

struct Mamba2LayerWeights {
    Tensor in_proj;
    Tensor conv1d;
    Tensor conv1d_bias;
    Tensor dt_bias;
    Tensor out_proj;
    Tensor A_log;
    Tensor D;
    Tensor mixer_norm;
    Tensor layer_norm;
};

struct Mamba2Weights {
    Tensor embeddings;
    std::vector<Mamba2LayerWeights> layers;
    Tensor norm_f;
    std::optional<Tensor> lm_head;
};
