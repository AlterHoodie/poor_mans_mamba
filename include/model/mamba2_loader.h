#pragma once

#include "core/status.h"
#include "io/model_loader.h"
#include "model/mamba2_weights.h"

class Mamba2ModelLoader : public ModelLoader<Mamba2Config, Mamba2Weights> {
   public:
    Status load(const Mamba2Config& cfg, Mamba2Weights& weights, const std::string& model_dir,
                int device_id = 0) override;
};
