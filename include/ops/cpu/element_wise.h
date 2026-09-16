#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "map.h"

template <typename Derived> struct BinarySameShapeOp {
  Status operator()(const Tensor& a, const Tensor& b, Tensor& out) const {
    if (Status s = validate_binary_same_shape(a, b, out); !s.ok())
      return s;
    return static_cast<const Derived*>(this)->compute(a, b, out);
  }

protected:
  static Status validate_binary_same_shape(const Tensor& a, const Tensor& b, Tensor& out) {
    if (Status s = require_f32_cpu(a, "a"); !s.ok())
      return s;
    if (Status s = require_f32_cpu(b, "b"); !s.ok())
      return s;
    if (Status s = require_f32_cpu(out, "out"); !s.ok())
      return s;
    if (!same_shape(a.shape, b.shape)) {
      return Status::InvalidArgument("shape mismatch between a and b");
    }
    if (!same_shape(out.shape, a.shape)) {
      return Status::InvalidArgument("out shape does not match element-wise result");
    }

    int64_t out_numel = 0;
    ASSIGN_OR_RETURN(out_numel, out.numel());
    if (out.buffer.bytes < static_cast<size_t>(out_numel) * sizeof(float)) {
      return Status::InvalidArgument("out buffer is too small");
    }
    return Status::Ok();
  }
};

template <typename Derived> struct UnarySameShapeOp {
  Status operator()(const Tensor& a, Tensor& out) const {
    if (Status s = validate_unary_same_shape(a, out); !s.ok())
      return s;
    return static_cast<const Derived*>(this)->compute(a, out);
  }

protected:
  static Status validate_unary_same_shape(const Tensor& a, Tensor& out) {
    if (Status s = require_f32_cpu(a, "a"); !s.ok())
      return s;
    if (Status s = require_f32_cpu(out, "out"); !s.ok())
      return s;
    if (!same_shape(out.shape, a.shape)) {
      return Status::InvalidArgument("out shape does not match unary element-wise result");
    }

    int64_t out_numel = 0;
    ASSIGN_OR_RETURN(out_numel, out.numel());
    if (out.buffer.bytes < static_cast<size_t>(out_numel) * sizeof(float)) {
      return Status::InvalidArgument("out buffer is too small");
    }
    return Status::Ok();
  }
};

struct MulOp : BinarySameShapeOp<MulOp> {
  Status compute(const Tensor& a, const Tensor& b, Tensor& out) const;
};

struct AddOp : BinarySameShapeOp<AddOp> {
  Status compute(const Tensor& a, const Tensor& b, Tensor& out) const;
};

struct SiluOp : UnarySameShapeOp<SiluOp> {
  Status compute(const Tensor& a, Tensor& out) const;
};

Status mul(const Tensor& a, const Tensor& b, Tensor& out);
StatusOr<Tensor> mul(const Tensor& a, const Tensor& b);

Status add(const Tensor& a, const Tensor& b, Tensor& out);
StatusOr<Tensor> add(const Tensor& a, const Tensor& b);

Status silu(const Tensor& a, Tensor& out);
StatusOr<Tensor> silu(const Tensor& a);
