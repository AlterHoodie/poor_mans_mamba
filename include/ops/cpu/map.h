#pragma once

#include "ops/common/tensor_checks.h"

#include <Eigen/Core>
#include <cassert>
#include <cstdint>

using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline void require_cpu_f32(const Tensor& T) {
  assert(T.dtype == Dtype::F32);
  assert(T.buffer.device == Device::CPU);
  assert(!T.empty());
}

inline Eigen::Map<const Eigen::VectorXf> as_vec_f32(const Tensor& T) {
  require_cpu_f32(T);
  const int64_t numel = T.numel().value();
  return {static_cast<const float*>(T.buffer.ptr), static_cast<Eigen::Index>(numel)};
}

inline Eigen::Map<Eigen::VectorXf> as_vec_f32(Tensor& T) {
  require_cpu_f32(T);
  const int64_t numel = T.numel().value();
  return {static_cast<float*>(T.buffer.ptr), static_cast<Eigen::Index>(numel)};
}

inline Eigen::Map<const RowMajorMatrix> as_mat_f32(const Tensor& T) {
  require_cpu_f32(T);
  assert(T.size() >= 2);
  const int64_t rows = T.rows().value();
  const int64_t cols = T.cols().value();
  return {static_cast<const float*>(T.buffer.ptr), static_cast<Eigen::Index>(rows),
          static_cast<Eigen::Index>(cols)};
}

inline Eigen::Map<RowMajorMatrix> as_mat_f32(Tensor& T) {
  require_cpu_f32(T);
  assert(T.size() >= 2);
  const int64_t rows = T.rows().value();
  const int64_t cols = T.cols().value();
  return {static_cast<float*>(T.buffer.ptr), static_cast<Eigen::Index>(rows),
          static_cast<Eigen::Index>(cols)};
}
