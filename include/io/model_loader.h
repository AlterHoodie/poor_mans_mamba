#pragma once

#include <cstring>
#include <string>

#include "core/status.h"
#include "core/tensor.h"
#include "io/config_parser.h"
#include "safetensors.hh"

// class loader will take in config and weights , will expose one function load which fills the
// weight object passed to it
template <typename ConfigT, typename WeightsT>
class ModelLoader {
   public:
    virtual ~ModelLoader() = default;
    virtual Status load(const ConfigT& cfg, WeightsT& weights, const std::string& model_dir,
                        int device_id = 0) = 0;
};

inline std::string layer_key(int layer, const char* suffix) {
    return "backbone.layers." + std::to_string(layer) + suffix;
}

Status copy_tensor_f32(const safetensors::safetensors_t& st, const std::string& key, Tensor& out,
                       int device_id);

#define COPY_OR_RETURN(field, key_expr)                              \
    do {                                                             \
        Status _s = copy_tensor_f32(st, key_expr, field, device_id); \
        if (!_s.ok()) {                                              \
            return Status(_s);                                       \
        }                                                            \
    } while (0)
