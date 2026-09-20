#pragma once

#include "ops/common/tensor_checks.h"

#include <cstdint>
#include <string>
#include <vector>

// a: [..., K], b: [N, K] -> [..., N]
inline StatusOr<std::vector<int64_t>> linear_output_shape(const Tensor& a, const Tensor& b) {
  if (a.size() < 1)
    return Status::InvalidArgument("a must have rank >= 1");
  if (b.size() != 2)
    return Status::InvalidArgument("b must be rank 2");

  int64_t k_a = 0;
  int64_t n_b = 0;
  int64_t k_b = 0;
  ASSIGN_OR_RETURN(k_a, a.last_dim());
  n_b = b.shape[0];
  k_b = b.shape[1];

  if (k_a != k_b) {
    return Status::InvalidArgument("inner dim mismatch: a.last=" + std::to_string(k_a) +
                                   " b.cols=" + std::to_string(k_b));
  }
  if (n_b <= 0 || k_b <= 0) {
    return Status::InvalidArgument("b dims must be positive");
  }

  std::vector<int64_t> out_shape = a.shape;
  out_shape.back() = n_b;
  return out_shape;
}

inline Status validate_rms_norm_shapes(const Tensor& x, const Tensor& weight, const Tensor& out) {
  if (x.size() < 1)
    return Status::InvalidArgument("x must have rank >= 1");

  int64_t dim = 0;
  ASSIGN_OR_RETURN(dim, x.last_dim());

  int64_t weight_numel = 0;
  ASSIGN_OR_RETURN(weight_numel, weight.numel());
  if (weight_numel != dim) {
    return Status::InvalidArgument("weight numel must match x last dim");
  }

  if (!same_shape(out.shape, x.shape)) {
    return Status::InvalidArgument("out shape must match x");
  }

  int64_t out_numel = 0;
  ASSIGN_OR_RETURN(out_numel, out.numel());
  if (out.buffer.bytes < static_cast<size_t>(out_numel) * sizeof(float)) {
    return Status::InvalidArgument("out buffer is too small");
  }
  return Status::Ok();
}
