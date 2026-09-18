#pragma once

#include "core/status.h"
#include "io/config.h"
#include "io/model_loader.h"
#include "model/mamba2_weights.h"

class Mamba2ModelLoader : public ModelLoader<ModelConfig, Mamba2Weights> {
public:
  Status load(const ModelConfig& cfg, Mamba2Weights& weights, const std::string& model_dir,
              int device_id = 0) override;
};
