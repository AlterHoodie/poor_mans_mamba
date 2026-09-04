#include "ops/cpu/rms_norm.h"

#include <cmath>
#include <utility>

#include "ops/cpu/map.h"

namespace {

Status validate_rms_norm_shapes(const Tensor& x, const Tensor& weight, const Tensor& out) {
    if (x.size() < 1) return Status::InvalidArgument("x must have rank >= 1");

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

void normalize_row(Eigen::Ref<const Eigen::RowVectorXf> x_row, Eigen::Ref<Eigen::RowVectorXf> y_row,
                   Eigen::Ref<const Eigen::VectorXf> weight, float eps) {
    const float mean_sq = x_row.array().square().mean();
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    y_row.array() = x_row.array() * inv_rms * weight.transpose().array();
}

void normalize_row_inplace(Eigen::Ref<Eigen::RowVectorXf> row,
                           Eigen::Ref<const Eigen::VectorXf> weight, float eps) {
    const float mean_sq = row.array().square().mean();
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    row.array() *= inv_rms * weight.transpose().array();
}

Status compute_rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
    const auto weight_vec = as_vec_f32(weight);

    if (x.size() == 1) {
        const auto x_vec = as_vec_f32(x);
        auto y_vec = as_vec_f32(out);
        normalize_row(x_vec.transpose(), y_vec.transpose(), weight_vec, eps);
        return Status::Ok();
    }

    const auto x_map = as_mat_f32(x);
    auto y_map = as_mat_f32(out);
    for (Eigen::Index r = 0; r < x_map.rows(); ++r) {
        normalize_row(x_map.row(r), y_map.row(r), weight_vec, eps);
    }
    return Status::Ok();
}

Status compute_rms_norm_inplace(Tensor& x, const Tensor& weight, float eps) {
    const auto weight_vec = as_vec_f32(weight);

    if (x.size() == 1) {
        auto x_vec = as_vec_f32(x);
        normalize_row_inplace(x_vec.transpose(), weight_vec, eps);
        return Status::Ok();
    }

    auto x_map = as_mat_f32(x);
    for (Eigen::Index r = 0; r < x_map.rows(); ++r) {
        normalize_row_inplace(x_map.row(r), weight_vec, eps);
    }
    return Status::Ok();
}

}  // namespace

Status rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
    if (Status s = require_f32_cpu(x, "x"); !s.ok()) return s;
    if (Status s = require_f32_cpu(weight, "weight"); !s.ok()) return s;
    if (Status s = require_f32_cpu(out, "out"); !s.ok()) return s;
    if (Status s = validate_rms_norm_shapes(x, weight, out); !s.ok()) return s;

    if (&x == &out) {
        return compute_rms_norm_inplace(out, weight, eps);
    }
    return compute_rms_norm(x, weight, eps, out);
}

StatusOr<Tensor> rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    StatusOr<Tensor> out = allocate_f32_tensor(x.shape);
    if (!out.ok()) return Status(out.status());

    if (Status s = rms_norm(x, weight, eps, out.value()); !s.ok()) {
        return s;
    }
    return std::move(out.value());
}

Status rms_norm_inplace(Tensor& x, const Tensor& weight, float eps) {
    if (Status s = require_f32_cpu(x, "x"); !s.ok()) return s;
    if (Status s = require_f32_cpu(weight, "weight"); !s.ok()) return s;

    int64_t dim = 0;
    ASSIGN_OR_RETURN(dim, x.last_dim());
    int64_t weight_numel = 0;
    ASSIGN_OR_RETURN(weight_numel, weight.numel());
    if (weight_numel != dim) {
        return Status::InvalidArgument("weight numel must match x last dim");
    }

    return compute_rms_norm_inplace(x, weight, eps);
}
