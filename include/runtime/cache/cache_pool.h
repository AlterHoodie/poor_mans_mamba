#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "io/config_parser.h"
#include "runtime/cache/cache_layout.h"

// Opaque handle into CachePool. Pass this instead of raw pointers.
struct CacheHandle {
   private:
    int id_ = -1;

   public:
    CacheHandle() = default;
    explicit CacheHandle(int id) : id_(id) {}

    int id() const { return id_; }
    bool valid() const { return id_ >= 0; }
    void invalidate() { id_ = -1; }
};

// one slot - one request - consists of many layers (conv + ssm state)
// CachePool leases slots, resets state at slot level, and exposes layer views.
class CachePool {
   private:
    Device device_;
    int device_id_;

    // Built once from config; shared blueprint for all slots.
    CacheLayout layout_;

    struct Slot {
        DeviceBuffer buffer;
        // Precomputed non-owning views into buffer (stable for slot lifetime).
        std::vector<MambaLayerCacheView> views;
        bool in_use = false;
    };

    std::vector<Slot> slots_;
    std::vector<int> free_list_;  // pop_back to acquire, push_back to release

    Status free_slot_(int slot_id);
    Status zero_slot_(int slot_id);

   public:
    CachePool(const Mamba2Config& cfg, Device device, int device_id, int n_slots = 10);

    StatusOr<CacheHandle> acquire();
    Status release(CacheHandle& handle);  // return slot to pool; invalidates handle
    Status reset(CacheHandle& handle);    // zero state, keep lease

    StatusOr<MambaLayerCacheView> layer_view(const CacheHandle& handle, int layer) const;

    CachePool(const CachePool&) = delete;
    CachePool& operator=(const CachePool&) = delete;
};
