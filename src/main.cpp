#include <iostream>
#include <string>

#include "runtime/model_registry.h"
#include "runtime/scheduler.h"
#include "runtime/tokenizer.h"

int main(int argc, char** argv) {
    const std::string model_dir = (argc > 1) ? argv[1] : "models/mamba2-130m-hf";

    auto entry_or = ModelRegistry::lookup(model_dir);
    if (!entry_or.ok()) {
        std::cerr << entry_or.status().message() << '\n';
        return 1;
    }
    const ModelEntry& entry = *entry_or.value();

    auto cfg_or = entry.parse(model_dir);
    if (!cfg_or.ok()) {
        std::cerr << cfg_or.status().message() << '\n';
        return 1;
    }
    const ModelConfig& cfg = *cfg_or.value();

    auto runner_or = entry.create_runner(cfg, model_dir, Device::CPU, /*device_id=*/0);
    if (!runner_or.ok()) {
        std::cerr << runner_or.status().message() << '\n';
        return 1;
    }
    std::unique_ptr<Runner> runner = std::move(runner_or.value());

    std::cout << "model_type=" << cfg.model_type << '\n'
              << "layers=" << cfg.num_hidden_layers << '\n'
              << "hidden_size=" << cfg.hidden_size << '\n';

    Tokenizer tokenizer(cfg, model_dir);
    Scheduler sched(*runner);

    std::string prompt = "Hi How are you?";
    auto ids = tokenizer.encode(prompt);
    if (!ids.ok()) {
        std::cerr << ids.status().message() << '\n';
        return 1;
    }

    GenerateParams params{.max_new_tokens = 8, .eos_id = cfg.eos_token_id};
    auto out = sched.generate(ids.value(), params);
    if (!out.ok()) {
        std::cerr << out.status().message() << '\n';
        return 1;
    }

    auto text = tokenizer.decode(out.value());
    if (!text.ok()) {
        std::cerr << text.status().message() << '\n';
        return 1;
    }

    std::cout << "Generated Text \n" << text.value() << '\n';
    return 0;
}
