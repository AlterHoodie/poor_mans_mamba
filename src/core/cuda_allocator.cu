#include "core/device.h"

#include <cuda_runtime.h>

static Status CudaStatus(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return Status::Ok();
  if (err == cudaErrorMemoryAllocation) return Status::OOM(what);
  return Status::RuntimeError(std::string(what) + ": " + cudaGetErrorString(err));
}

StatusOr<DeviceMemory> CUDAAllocator::allocate(size_t bytes) {
  if (bytes == 0)
    return Status::InvalidArgument("bytes cannot be less than or equal to 0");

  auto st = CudaStatus(cudaSetDevice(device_id_), "cudaSetDevice");
  if (!st.ok()) return st;

  void* ptr = nullptr;
  st = CudaStatus(cudaMalloc(&ptr, bytes), "cudaMalloc");
  if (!st.ok()) return st;

  DeviceMemory mem{};
  mem.alloc = this;
  mem.ptr = ptr;
  mem.bytes = bytes;
  mem.device = kind_;       // Device::GPU
  mem.device_id = device_id_;
  return std::move(mem);
}

Status CUDAAllocator::free(DeviceMemory& mem) {
  // same device / device_id / nullptr checks as CPU
  auto st = CudaStatus(cudaSetDevice(device_id_), "cudaSetDevice");
  if (!st.ok()) return st;

  st = CudaStatus(cudaFree(mem.ptr), "cudaFree");
  if (!st.ok()) return st;

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
  if (!st.ok()) return st;

  return CudaStatus(
      cudaMemset(static_cast<char*>(mem.ptr) + offset, 0, bytes),
      "cudaMemset");
}