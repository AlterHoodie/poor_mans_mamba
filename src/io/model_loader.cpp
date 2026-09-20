#include "io/model_loader.h"

#include <cstdint>
#include <vector>

#ifdef MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

inline float bf16_bits_to_f32(uint16_t bits) {
  const uint32_t u = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &u, sizeof(out));
  return out;
}

Status host_to_device_f32(DeviceMemory& buffer, const void* host, size_t nbytes) {
  if (buffer.device == Device::CPU) {
    std::memcpy(buffer.ptr, host, nbytes);
    return Status::Ok();
  }
#ifdef MAMBASERVE_WITH_CUDA
  if (buffer.device == Device::GPU) {
    cudaError_t err = cudaMemcpy(buffer.ptr, host, nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      return Status::RuntimeError(std::string("cudaMemcpy H2D failed: ") +
                                     cudaGetErrorString(err));
    }
    return Status::Ok();
  }
#endif
  return Status::InvalidArgument("unsupported device for weight copy");
}

} // namespace

Status copy_tensor_f32(const safetensors::safetensors_t& st, const std::string& key, Tensor& out,
                       int device_id) {
  (void)device_id;
  safetensors::tensor_t meta{};
  if (!st.tensors.at(key, &meta)) {
    return Status::NotFound("missing tensor: " + key);
  }

  const bool is_f32 = meta.dtype == safetensors::dtype::kFLOAT32;
  const bool is_bf16 = meta.dtype == safetensors::dtype::kBFLOAT16;
  if (!is_f32 && !is_bf16) {
    return Status::InvalidArgument("expected F32 or BF16 for tensor: " + key);
  }

  const uint8_t* base = st.mmaped ? st.databuffer_addr : st.storage.data();
  const size_t start = meta.data_offsets[0];
  const size_t end = meta.data_offsets[1];
  const size_t nbytes_src = end - start;

  size_t numel = 1;
  for (size_t dim : meta.shape) {
    if (dim == 0) {
      return Status::InvalidArgument("tensor has a zero dimension: " + key);
    }
    if (numel > SIZE_MAX / dim) {
      return Status::OOM("tensor numel overflow: " + key);
    }
    numel *= dim;
  }

  const size_t expect_src = numel * (is_f32 ? sizeof(float) : sizeof(uint16_t));
  if (nbytes_src != expect_src) {
    return Status::InvalidArgument("tensor byte size mismatch: " + key);
  }

  DeviceAllocator* a = current_allocator();
  if (!a)
    return Status::InvalidArgument("no current allocator");

  const size_t nbytes_dst = numel * sizeof(float);
  DeviceMemory buffer{};
  ASSIGN_OR_RETURN(buffer, a->allocate(nbytes_dst));

  if (is_f32) {
    if (Status s = host_to_device_f32(buffer, base + start, nbytes_src); !s.ok())
      return s;
  } else {
    std::vector<float> host(numel);
    const auto* src = reinterpret_cast<const uint16_t*>(base + start);
    for (size_t i = 0; i < numel; ++i)
      host[i] = bf16_bits_to_f32(src[i]);
    if (Status s = host_to_device_f32(buffer, host.data(), nbytes_dst); !s.ok())
      return s;
  }

  out.shape.clear();
  out.shape.reserve(meta.shape.size());
  for (size_t dim : meta.shape) {
    out.shape.push_back(static_cast<int64_t>(dim));
  }
  out.dtype = Dtype::F32;
  out.buffer = std::move(buffer);
  return Status::Ok();
}