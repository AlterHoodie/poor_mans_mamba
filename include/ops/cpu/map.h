#pragma once

#include "core/status.h"
#include "core/tensor.h"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

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

inline Status require_f32_cpu(const Tensor& t, const char* name) {
  if (t.empty())
    return Status::InvalidArgument(std::string(name) + " is empty");
  if (t.dtype != Dtype::F32) {
    return Status::InvalidArgument(std::string(name) + " must be F32");
  }
  if (t.buffer.device != Device::CPU) {
    return Status::InvalidArgument(std::string(name) + " must be on CPU");
  }
  return Status::Ok();
}

inline StatusOr<Tensor> allocate_f32_tensor(std::vector<int64_t> shape) {
  if (shape.empty())
    return Status::InvalidArgument("Shape cannot be empty");
  Tensor out;
  out.dtype = Dtype::F32;
  out.shape = std::move(shape);

  int64_t numel = 0;
  ASSIGN_OR_RETURN(numel, out.numel());
  size_t nbytes = static_cast<size_t>(numel) * sizeof(float);

  DeviceAllocator* a = current_allocator();
  if (!a)
    return Status::InvalidArgument("no current allocator");

  auto out_or = a->allocate(nbytes);
  if (!out_or.ok())
    return out_or.status();
  out.buffer = std::move(out_or.value());

  return out;
}

inline bool same_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}