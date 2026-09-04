#pragma once

#include "core/status.h"
#include "core/tensor.h"

Status rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out);
StatusOr<Tensor> rms_norm(const Tensor& x, const Tensor& weight, float eps);

Status rms_norm_inplace(Tensor& x, const Tensor& weight, float eps);
