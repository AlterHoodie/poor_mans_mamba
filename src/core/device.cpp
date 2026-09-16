#include "core/device.h"

#include <cstdlib>

StatusOr<DeviceMemory> CPUAllocator::allocate(size_t bytes) {
  if (bytes <= 0)
    return Status::InvalidArgument("bytes cannot be less than or equal to 0");

  DeviceMemory mem{};
  void* ptr = std::malloc(bytes);

  if (!ptr)
    return Status::OOM("Couldnt allocate memory");

  mem.alloc = this; // stamp memory with this allocator reference
  mem.bytes = bytes;
  mem.device = kind_;
  mem.device_id = device_id_;
  mem.ptr = ptr;

  return mem;
}

Status CPUAllocator::free(DeviceMemory& mem) {
  if (mem.device != kind_)
    return Status::InvalidArgument("device doesnt match");
  if (mem.device_id != device_id_)
    return Status::InvalidArgument("device_id doesnt match");
  if (!mem.ptr)
    return Status::InvalidArgument("nullptr");

  std::free(mem.ptr);

  mem.alloc = nullptr;
  mem.ptr = nullptr;
  mem.bytes = 0;
  mem.device_id = -1;
  mem.device = Device::NA;

  return Status::Ok();
}

Status CPUAllocator::memset_zero(DeviceMemory& mem, size_t offset, size_t bytes) {
  if (mem.device != kind_)
    return Status::InvalidArgument("device doesnt match");
  if (mem.device_id != device_id_)
    return Status::InvalidArgument("device_id doesnt match");
  if (!mem.ptr)
    return Status::InvalidArgument("nullptr");
  if (bytes <= 0)
    return Status::InvalidArgument("bytes cannot be less than or equal to 0");
  if (bytes > mem.bytes)
    return Status::InvalidArgument("bytes cannot be greated than size of DeviceMemory");
  if ((offset + bytes) > mem.bytes)
    return Status::InvalidArgument("range to memset zero should be less than DeviceMemory bytes");

  if (!std::memset(static_cast<char*>(mem.ptr) + offset, 0, bytes))
    return Status::OOM("Could not memset memory to zero");

  return Status::Ok();
}

StatusOr<std::unique_ptr<DeviceAllocator>> create_device_allocator(Device device, int device_id) {
  if (device == Device::NA)
    return Status::InvalidArgument("Invalid Device Argument");
  if (device == Device::GPU)
    return Status::InvalidArgument("Not yet implemented");
  if (device_id < 0)
    return Status::InvalidArgument("device_id cannot be less than 0");

  std::unique_ptr<DeviceAllocator> allocator = std::make_unique<CPUAllocator>(device_id);

  return allocator;
}

DeviceMemory::DeviceMemory(DeviceMemory&& mem) noexcept
    : alloc(mem.alloc), ptr(mem.ptr), bytes(mem.bytes), device(mem.device),
      device_id(mem.device_id) {
  mem.alloc = nullptr;
  mem.ptr = nullptr;
  mem.bytes = 0;

  mem.device = Device::NA;
  mem.device_id = -1;
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&& mem) noexcept {
  if (this != &mem) {
    alloc = mem.alloc;
    mem.alloc = nullptr;

    ptr = mem.ptr;
    mem.ptr = nullptr;

    bytes = mem.bytes;
    mem.bytes = 0;

    device = mem.device;
    mem.device = Device::NA;

    device_id = mem.device_id;
    mem.device_id = -1;
  }
  return *this;
}

DeviceMemory::~DeviceMemory() {
  if (!alloc)
    return;
  alloc->free(*this);
}

namespace {
thread_local DeviceAllocator* current = nullptr;
}

DeviceAllocator* current_allocator() { return current; }
void set_current_allocator(DeviceAllocator* a) { current = a; };

AllocatorScope::AllocatorScope(DeviceAllocator* a) : prev(current_allocator()) {
  set_current_allocator(a);
}
AllocatorScope::~AllocatorScope() { set_current_allocator(prev); }