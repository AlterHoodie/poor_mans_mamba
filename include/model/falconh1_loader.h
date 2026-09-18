#pragma once
#include "core/status.h"
#include "falconh1_weights.h"
#include "io/config.h"
#include "io/model_loader.h"

#include <cstring>

class FalconH1ModelLoader : public ModelLoader<ModelConfig, FalconH1Weights> {
public:
  Status load(const ModelConfig& cfg, FalconH1Weights& weights, const std::string& model_dir,
              int device_id = 0) override;
};
