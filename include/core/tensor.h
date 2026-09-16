#pragma once

#include "core/device.h"
#include "core/status.h"

#include <cstddef>
#include <cstdint>
#include <vector>

enum class Dtype { F32, F16, F8 };

struct Tensor {
  DeviceMemory buffer;
  std::vector<int64_t> shape;
  Dtype dtype = Dtype::F32;

  Tensor() = default;
  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&&) = default;

  // Copy Constructor and Assignment deleted
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  // Rank (number of dims). Not numel.
  [[nodiscard]] size_t size() const { return shape.size(); }

  [[nodiscard]] bool empty() const {
    if (buffer.ptr == nullptr || shape.empty())
      return true;
    for (int64_t d : shape) {
      if (d <= 0)
        return true;
    }
    return false;
  }

  StatusOr<int64_t> numel() const {
    if (shape.empty())
      return Status::InvalidArgument("Cannot compute numel of a rank-0 tensor");
    int64_t n = 1;
    for (int64_t d : shape) {
      if (d < 0)
        return Status::InvalidArgument("Shape dims must be non-negative");
      n *= d;
    }
    return n;
  }

  StatusOr<int64_t> last_dim() const {
    if (shape.empty())
      return Status::InvalidArgument("Cannot get last dim of a rank-0 tensor");
    return shape.back();
  }

  // Product of leading dims. Rank 1 -> 1.
  StatusOr<int64_t> batch_size() const {
    if (shape.empty())
      return Status::InvalidArgument("Cannot get batch size of a rank-0 tensor");
    if (shape.size() == 1)
      return 1;
    return rows();
  }

  // Leading dims as one row count: {2, 3, 4} -> 6. Requires rank >= 2.
  StatusOr<int64_t> rows() const {
    if (shape.size() < 2) {
      return Status::InvalidArgument("rows() requires rank >= 2");
    }
    int64_t r = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      if (shape[i] < 0)
        return Status::InvalidArgument("Shape dims must be non-negative");
      r *= shape[i];
    }
    return r;
  }

  // Last dim: {2, 3, 4} -> 4. Requires rank >= 2.
  StatusOr<int64_t> cols() const {
    if (shape.size() < 2) {
      return Status::InvalidArgument("cols() requires rank >= 2");
    }
    return shape.back();
  }
};
