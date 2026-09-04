#include "ops/cpu/element_wise.h"

#include <utility>

Status MulOp::compute(const Tensor& a, const Tensor& b, Tensor& out) const {
    as_vec_f32(out).array() = as_vec_f32(a).array() * as_vec_f32(b).array();
    return Status::Ok();
}

Status AddOp::compute(const Tensor& a, const Tensor& b, Tensor& out) const {
    if (&a == &out) {
        as_vec_f32(out).array() += as_vec_f32(b).array();
        return Status::Ok();
    }
    as_vec_f32(out).array() = as_vec_f32(a).array() + as_vec_f32(b).array();
    return Status::Ok();
}

Status SiluOp::compute(const Tensor& a, Tensor& out) const {
    const auto x = as_vec_f32(a).array();
    as_vec_f32(out).array() = x / (1.0f + (-x).exp());
    return Status::Ok();
}

namespace {

template <typename Op>
StatusOr<Tensor> allocate_and_apply_binary(const Tensor& a, const Tensor& b) {
    if (!same_shape(a.shape, b.shape)) {
        return Status::InvalidArgument("shape mismatch between a and b");
    }

    StatusOr<Tensor> out = allocate_f32_tensor(a.shape);
    if (!out.ok()) return Status(out.status());

    Op op;
    if (Status s = op(a, b, out.value()); !s.ok()) return s;
    return std::move(out.value());
}

template <typename Op>
StatusOr<Tensor> allocate_and_apply_unary(const Tensor& a) {
    StatusOr<Tensor> out = allocate_f32_tensor(a.shape);
    if (!out.ok()) return Status(out.status());

    Op op;
    if (Status s = op(a, out.value()); !s.ok()) return s;
    return std::move(out.value());
}

}  // namespace

Status mul(const Tensor& a, const Tensor& b, Tensor& out) { return MulOp{}(a, b, out); }

StatusOr<Tensor> mul(const Tensor& a, const Tensor& b) {
    return allocate_and_apply_binary<MulOp>(a, b);
}

Status add(const Tensor& a, const Tensor& b, Tensor& out) { return AddOp{}(a, b, out); }

StatusOr<Tensor> add(const Tensor& a, const Tensor& b) {
    return allocate_and_apply_binary<AddOp>(a, b);
}

Status silu(const Tensor& a, Tensor& out) { return SiluOp{}(a, out); }

StatusOr<Tensor> silu(const Tensor& a) { return allocate_and_apply_unary<SiluOp>(a); }
