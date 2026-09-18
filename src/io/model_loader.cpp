#include "io/model_loader.h"

#include <cstdint>

namespace {

inline float bf16_bits_to_f32(uint16_t bits) {
  const uint32_t u = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &u, sizeof(out));
  return out;
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
    std::memcpy(buffer.ptr, base + start, nbytes_src);
  } else {
    const auto* src = reinterpret_cast<const uint16_t*>(base + start);
    auto* dst = static_cast<float*>(buffer.ptr);
    for (size_t i = 0; i < numel; ++i)
      dst[i] = bf16_bits_to_f32(src[i]);
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