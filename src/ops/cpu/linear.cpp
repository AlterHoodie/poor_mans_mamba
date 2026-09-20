#include "ops/cpu/linear.h"

#include "ops/common/shapes.h"

#include <utility>
#include <vector>

Status linear(const Tensor& a, const Tensor& b, Tensor& out) {
  if (Status s = require_f32(a, "a", Device::CPU); !s.ok())
    return s;
  if (Status s = require_f32(b, "b", Device::CPU); !s.ok())
    return s;
  if (Status s = require_f32(out, "out", Device::CPU); !s.ok())
    return s;

  StatusOr<std::vector<int64_t>> expected_shape = linear_output_shape(a, b);
  if (!expected_shape.ok())
    return Status(expected_shape.status());

  if (!same_shape(out.shape, expected_shape.value())) {
    return Status::InvalidArgument("out shape does not match linear result");
  }

  int64_t out_numel = 0;
  ASSIGN_OR_RETURN(out_numel, out.numel());
  if (out.buffer.bytes < static_cast<size_t>(out_numel) * sizeof(float)) {
    return Status::InvalidArgument("out buffer is too small");
  }

  if (a.size() == 1) {
    const auto b_map = as_mat_f32(b);
    const auto a_vec = as_vec_f32(a);
    auto out_vec = as_vec_f32(out);
    out_vec.noalias() = b_map * a_vec;
    return Status::Ok();
  }

  const auto a_map = as_mat_f32(a);
  const auto b_map = as_mat_f32(b);
  auto out_map = as_mat_f32(out);
  out_map.noalias() = a_map * b_map.transpose();
  return Status::Ok();
}

StatusOr<Tensor> linear(const Tensor& a, const Tensor& b) {
  StatusOr<std::vector<int64_t>> out_shape = linear_output_shape(a, b);
  if (!out_shape.ok())
    return Status(out_shape.status());

  StatusOr<Tensor> out = allocate_f32_tensor(std::move(out_shape).value());
  if (!out.ok())
    return Status(out.status());

  if (Status s = linear(a, b, out.value()); !s.ok()) {
    return s;
  }
  return std::move(out.value());
}
