#include "core/device.h"

#include <cuda_runtime.h>

#include <string>

static Status CudaStatus(cudaError_t err, const char* what) {
  if (err == cudaSuccess)
    return Status::Ok();
  if (err == cudaErrorMemoryAllocation)
    return Status::OOM(what);
  return Status::RuntimeError(std::string(what) + ": " + cudaGetErrorString(err));
}

CUDAAllocator::~CUDAAllocator() {
  if (cudaSetDevice(device_id_) != cudaSuccess)
    return;
  for (auto& [bytes, ptrs] : free_lists_) {
    (void)bytes;
    for (void* p : ptrs)
      cudaFree(p);
  }
  free_lists_.clear();
}

StatusOr<DeviceMemory> CUDAAllocator::allocate(size_t bytes) {
  if (bytes == 0)
    return Status::InvalidArgument("bytes cannot be less than or equal to 0");

  auto st = CudaStatus(cudaSetDevice(device_id_), "cudaSetDevice");
  if (!st.ok())
    return st;

  void* ptr = nullptr;
  auto it = free_lists_.find(bytes);
  if (it != free_lists_.end() && !it->second.empty()) {
    ptr = it->second.back();
    it->second.pop_back();
  } else {
    st = CudaStatus(cudaMalloc(&ptr, bytes), "cudaMalloc");
    if (!st.ok())
      return st;
  }

  DeviceMemory mem{};
  mem.alloc = this;
  mem.ptr = ptr;
  mem.bytes = bytes;
  mem.device = kind_; // Device::GPU
  mem.device_id = device_id_;
  return std::move(mem);
}

Status CUDAAllocator::free(DeviceMemory& mem) {
  if (mem.device != kind_)
    return Status::InvalidArgument("device doesnt match");
  if (mem.device_id != device_id_)
    return Status::InvalidArgument("device_id doesnt match");
  if (!mem.ptr)
    return Status::InvalidArgument("nullptr");

  free_lists_[mem.bytes].push_back(mem.ptr);

  mem.alloc = nullptr;
  mem.ptr = nullptr;
  mem.bytes = 0;
  mem.device_id = -1;
  mem.device = Device::NA;
  return Status::Ok();
}

Status CUDAAllocator::memset_zero(DeviceMemory& mem, size_t offset, size_t bytes) {
  // same range checks as CPU
  auto st = CudaStatus(cudaSetDevice(device_id_), "cudaSetDevice");
  if (!st.ok())
    return st;

  return CudaStatus(cudaMemset(static_cast<char*>(mem.ptr) + offset, 0, bytes), "cudaMemset");
}
