#include <iostream>
#include <string>

#include "io/config_parser.h"
#include "io/model_loader.h"
#include "model/mamba2_loader.h"
#include "model/mamba2_weights.h"

int main(int argc, char** argv) {
    const std::string model_dir = (argc > 1) ? argv[1] : "models/mamba2-130m-hf";

    Mamba2ConfigParser parser;
    const auto cfg_result = parser.parse(model_dir);
    if (!cfg_result.ok()) {
        std::cerr << cfg_result.status().message() << '\n';
        return 1;
    }

    const Mamba2Config& cfg = cfg_result.value();
    Mamba2Weights weights{};
    Mamba2ModelLoader loader;
    const auto weights_result = loader.load(cfg, weights, model_dir);
    if (!weights_result.ok()) {
        std::cerr << weights_result.message() << '\n';
        return 1;
    }

    const Mamba2Weights& loaded = weights;
    std::cout << "model_type=" << cfg.model_type << '\n'
              << "layers=" << loaded.layers.size() << '\n'
              << "embeddings=" << loaded.embeddings.shape[0] << 'x' << loaded.embeddings.shape[1]
              << '\n'
              << "layer0.in_proj=" << loaded.layers[0].in_proj.shape[0] << 'x'
              << loaded.layers[0].in_proj.shape[1] << '\n';

    return 0;
}
