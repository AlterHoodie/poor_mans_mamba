#pragma once

#include "core/status.h"
#include "core/tensor.h"

#include <cstdint>

StatusOr<int32_t> argmax(Tensor& T);

// Softmax then greedy index (same token as argmax for a single vector).
StatusOr<int32_t> softmax(Tensor& T);
