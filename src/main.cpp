#include <iostream>
#include <string>

#include "io/config_parser.h"
#include "io/model_loader.h"
#include "model/mamba2_loader.h"
#include "model/mamba2_weights.h"
#include "runtime/runner/mamba2runner.h"
#include "runtime/scheduler.h"
#include "runtime/tokenizer.h"

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

    Tokenizer tokenizer(cfg, model_dir);
    Mamba2Runner runner(cfg, std::move(weights), Device::CPU, 0);
    Scheduler sched(runner);

    std::string prompt = "Hi How are you?";
    auto ids = tokenizer.encode(prompt);
    GenerateParams params{.max_new_tokens = 8, .eos_id = cfg.eos_token_id};

    auto out = sched.generate(ids.value(), params);
    auto text = tokenizer.decode(out.value());

    std::cout << "Generated Text \n" << text.value() << '\n';
    return 0;
}
