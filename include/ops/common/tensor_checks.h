#pragma once

#include "core/status.h"
#include "core/tensor.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

inline Status require_f32(const Tensor& t, const char* name, Device want) {
  if (t.empty())
    return Status::InvalidArgument(std::string(name) + " is empty");
  if (t.dtype != Dtype::F32) {
    return Status::InvalidArgument(std::string(name) + " must be F32");
  }
  if (t.buffer.device != want) {
    const char* where = (want == Device::CPU) ? "CPU" : (want == Device::GPU) ? "GPU" : "expected device";
    return Status::InvalidArgument(std::string(name) + " must be on " + where);
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

  return std::move(out);
}

inline bool same_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
