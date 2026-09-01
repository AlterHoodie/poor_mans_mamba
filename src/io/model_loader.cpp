#include "io/model_loader.h"

Status copy_tensor_f32(const safetensors::safetensors_t& st, const std::string& key, Tensor& out,
                       int device_id) {
    safetensors::tensor_t meta{};
    if (!st.tensors.at(key, &meta)) {
        return Status::NotFound("missing tensor: " + key);
    }
    if (meta.dtype != safetensors::dtype::kFLOAT32) {
        return Status::InvalidArgument("expected F32 for tensor: " + key);
    }

    const uint8_t* base = st.mmaped ? st.databuffer_addr : st.storage.data();
    const size_t start = meta.data_offsets[0];
    const size_t end = meta.data_offsets[1];
    const size_t nbytes = end - start;

    DeviceBuffer buffer{};
    buffer.device = Device::CPU;
    buffer.device_id = device_id;
    buffer.bytes = nbytes;
    buffer.data = std::malloc(nbytes);
    if (buffer.data == nullptr) {
        return Status::InvalidArgument("failed to allocate tensor: " + key);
    }
    std::memcpy(buffer.data, base + start, nbytes);

    out.shape.clear();
    out.shape.reserve(meta.shape.size());
    for (size_t dim : meta.shape) {
        out.shape.push_back(static_cast<int64_t>(dim));
    }
    out.dtype = Dtype::F32;
    out.buffer = std::move(buffer);
    return Status::Ok();
}