#include <iostream>
#include <string>

#include "config/config_parser.h"

int main(int argc, char** argv) {
    const std::string model_dir = (argc > 1) ? argv[1] : "models/mamba2-130m-hf";

    Mamba2ConfigParser parser;
    const auto result = parser.parse(model_dir);
    if (!result.ok()) {
        std::cerr << result.status().message() << '\n';
        return 1;
    }

    const Mamba2Config& cfg = result.value();
    std::cout << "model_type=" << cfg.model_type << '\n'
              << "hidden_size=" << cfg.hidden_size << '\n'
              << "num_hidden_layers=" << cfg.num_hidden_layers << '\n'
              << "vocab_size=" << cfg.vocab_size << '\n'
              << "state_size=" << cfg.state_size << '\n';

    return 0;
}
