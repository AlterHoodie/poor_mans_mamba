#include "runtime/cache/cache_pool.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

CachePool::CachePool(const Mamba2Config& cfg, Device device, int device_id, int n_slots)
    : device_(device), device_id_(device_id) {
    if (n_slots <= 0) {
        throw std::runtime_error("n_slots must be positive");
    }

    StatusOr<CacheLayout> layout_status = CacheLayout::from_config(cfg);
    if (!layout_status.ok()) {
        throw std::runtime_error(layout_status.status().message());
    }
    layout_ = std::move(layout_status).value();

    slots_.resize(static_cast<size_t>(n_slots));
    free_list_.reserve(static_cast<size_t>(n_slots));

    for (int slot = 0; slot < n_slots; ++slot) {
        void* p = std::malloc(layout_.slot_bytes);
        if (!p) {
            throw std::runtime_error("failed to allocate cache slot");
        }
        std::memset(p, 0, layout_.slot_bytes);

        DeviceBuffer buf;
        buf.data = p;
        buf.bytes = layout_.slot_bytes;
        buf.device = device_;
        buf.device_id = device_id_;

        Slot& s = slots_[static_cast<size_t>(slot)];
        s.buffer = std::move(buf);
        s.views.resize(static_cast<size_t>(layout_.num_layers));

        auto* base = static_cast<std::byte*>(s.buffer.data);
        for (int layer = 0; layer < layout_.num_layers; ++layer) {
            const MambaLayerLayout& L = layout_.layers[static_cast<size_t>(layer)];
            s.views[static_cast<size_t>(layer)] = MambaLayerCacheView{
                .conv = base + L.conv_offset,
                .conv_bytes = L.conv_size,
                .ssm = base + L.ssm_offset,
                .ssm_bytes = L.ssm_size,
            };
        }

        free_list_.push_back(slot);
    }
}

StatusOr<CacheHandle> CachePool::acquire() {
    if (free_list_.empty()) {
        return Status::OOM("Slot Memory Pool Exhausted");
    }

    const int slot_id = free_list_.back();
    free_list_.pop_back();
    slots_[static_cast<size_t>(slot_id)].in_use = true;
    return CacheHandle(slot_id);
}

Status CachePool::zero_slot_(int slot_id) {
    if (slot_id < 0 || static_cast<size_t>(slot_id) >= slots_.size()) {
        return Status::InvalidArgument("Invalid slot id");
    }

    Slot& slot = slots_[static_cast<size_t>(slot_id)];
    if (slot.buffer.data == nullptr || slot.buffer.bytes == 0) {
        return Status::InvalidArgument("Slot buffer is not allocated");
    }

    // Keep allocation and views; clear recurrent state only.
    std::memset(slot.buffer.data, 0, slot.buffer.bytes);
    return Status::Ok();
}

Status CachePool::free_slot_(int slot_id) {
    if (slot_id < 0 || static_cast<size_t>(slot_id) >= slots_.size()) {
        return Status::InvalidArgument("Invalid slot id");
    }

    Slot& slot = slots_[static_cast<size_t>(slot_id)];
    if (!slot.in_use) {
        return Status::InvalidArgument("Slot is not in use");
    }

    Status z = zero_slot_(slot_id);
    if (!z.ok()) {
        return z;
    }

    slot.in_use = false;
    free_list_.push_back(slot_id);
    return Status::Ok();
}

Status CachePool::release(CacheHandle& handle) {
    if (!handle.valid()) {
        return Status::InvalidArgument("Invalid cache handle");
    }

    Status st = free_slot_(handle.id());
    handle.invalidate();
    return st;
}

Status CachePool::reset(CacheHandle& handle) {
    if (!handle.valid()) {
        return Status::InvalidArgument("Invalid cache handle");
    }

    const int slot_id = handle.id();
    if (static_cast<size_t>(slot_id) >= slots_.size() ||
        !slots_[static_cast<size_t>(slot_id)].in_use) {
        return Status::InvalidArgument("Slot is not in use");
    }

    return zero_slot_(slot_id);
}

StatusOr<MambaLayerCacheView> CachePool::layer_view(const CacheHandle& handle, int layer) const {
    if (!handle.valid()) {
        return Status::NotFound("Could not find slot or invalid slot_id");
    }
    if (layer < 0 || layer >= layout_.num_layers) {
        return Status::InvalidArgument("Invalid layer");
    }

    const int slot_id = handle.id();
    if (static_cast<size_t>(slot_id) >= slots_.size() ||
        !slots_[static_cast<size_t>(slot_id)].in_use) {
        return Status::InvalidArgument("Slot is not in use");
    }

    return slots_[static_cast<size_t>(slot_id)].views[static_cast<size_t>(layer)];
}
