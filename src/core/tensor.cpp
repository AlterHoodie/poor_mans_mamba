#include "core/tensor.h"

#include <cstdlib>
#include <utility>

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data(other.data), bytes(other.bytes), device_id(other.device_id), device(other.device) {
    other.data = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        data = other.data;
        bytes = other.bytes;
        device_id = other.device_id;
        device = other.device;
        other.data = nullptr;
        other.bytes = 0;
    }
    return *this;
}

void DeviceBuffer::reset() noexcept {
    std::free(data);
    data = nullptr;
    bytes = 0;
    device_id = 0;
    device = Device::CPU;
}

DeviceBuffer::~DeviceBuffer() { reset(); }
