#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "map.h"

StatusOr<Tensor> linear(const Tensor& a, const Tensor& b);
Status linear(const Tensor& a, const Tensor& b, Tensor& out);
