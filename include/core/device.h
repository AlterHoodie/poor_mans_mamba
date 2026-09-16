#pragma once

#include "core/status.h"

#include <cstring>
#include <memory>

enum class Device { NA, GPU, CPU };

class DeviceAllocator; // forward declaration

struct DeviceMemory {
  DeviceAllocator* alloc = nullptr;

  void* ptr = nullptr;
  size_t bytes = 0;
  Device device = Device::NA;
  int device_id = -1;

  DeviceMemory() = default;
  // Each DeviceMemory is responsible to free itself
  ~DeviceMemory();

  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;

  DeviceMemory(DeviceMemory&&) noexcept;
  DeviceMemory& operator=(DeviceMemory&&) noexcept;
};

class DeviceAllocator {
protected:
  DeviceAllocator(Device kind, int device_id) : kind_(kind), device_id_(device_id) {}

  Device kind_ = Device::NA;
  int device_id_ = -1;

public:
  virtual ~DeviceAllocator() = default;

  virtual StatusOr<DeviceMemory> allocate(size_t bytes) = 0;
  virtual Status free(DeviceMemory&) = 0;
  virtual Status memset_zero(DeviceMemory&, size_t offset, size_t bytes) = 0;

  Device kind() const { return kind_; };
  int device_id() const { return device_id_; };
};

class CPUAllocator : public DeviceAllocator {
public:
  explicit CPUAllocator(int device_id) : DeviceAllocator(Device::CPU, device_id){};

  StatusOr<DeviceMemory> allocate(size_t bytes) final override;

  Status free(DeviceMemory& mem) final override;
  Status memset_zero(DeviceMemory& mem, size_t offset, size_t bytes) final override;
};

StatusOr<std::unique_ptr<DeviceAllocator>> create_device_allocator(Device device, int device_id);

DeviceAllocator* current_allocator();
void set_current_allocator(DeviceAllocator* a);

struct AllocatorScope {
  DeviceAllocator* prev;
  explicit AllocatorScope(DeviceAllocator* a);

  ~AllocatorScope();
  AllocatorScope(const AllocatorScope&) = delete;
  AllocatorScope& operator=(const AllocatorScope&) = delete;
};