#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

enum class Dtype { F32, F16, F8 };
enum class Device { GPU, CPU };

struct DeviceBuffer {
    // Explicitly managing data so need to do rule of five
    void* data = nullptr;
    size_t bytes = 0;
    int device_id = 0;
    Device device = Device::CPU;

    // Default Constructor
    DeviceBuffer() = default;

    // Copy Constructor deleted cause single ownership could use unique ptr but for now this simpler
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move Constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept;

    // Move Assignment
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    // Free owned memory (malloc/free). Does not zero caller-held views.
    void reset() noexcept;

    ~DeviceBuffer();
};

struct Tensor {
    DeviceBuffer buffer;
    std::vector<int64_t> shape;
    Dtype dtype = Dtype::F32;

    Tensor() = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    // Copy Constructor and Assignment deleted
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
};
