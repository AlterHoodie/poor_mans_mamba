#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "io/config.h"
#include "runtime/cache/cache_layout.h"

#include <memory>
#include <span>

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
  DeviceAllocator* allocator_ = nullptr; // non-owning; outlives the pool
  DeviceMemory slab_{};

  // Built once from config; shared blueprint for all slots.
  CacheLayout layout_;

  struct Slot {
    // Precomputed non-owning views into buffer (stable for slot lifetime).
    std::vector<LayerCacheView> views;
    bool in_use = false;
    // Tokens written into KV cache so far (Falcon-H1 / AttnKV). Unused for pure Mamba2.
    int64_t seq_len = 0;
  };

  std::vector<Slot> slots_;
  std::vector<int> free_list_; // pop_back to acquire, push_back to release

  Status free_slot_(int slot_id);
  Status zero_slot_(int slot_id);

  CachePool(CacheLayout layout, DeviceAllocator* allocator);

public:
  ~CachePool() = default;

  static StatusOr<std::unique_ptr<CachePool>> create(CacheLayout layout, DeviceAllocator* allocator,
                                                     int num_slots);

  StatusOr<CacheHandle> acquire();
  Status release(CacheHandle& handle); // return slot to pool; invalidates handle
  Status reset(CacheHandle& handle);   // zero state, keep lease

  StatusOr<LayerCacheView> layer_view(const CacheHandle& handle, int layer) const;
  StatusOr<std::span<LayerCacheView>> layer_views(const CacheHandle& handle);

  StatusOr<int64_t> seq_len(const CacheHandle& handle) const;
  Status set_seq_len(const CacheHandle& handle, int64_t len);

  size_t slot_bytes() const { return layout_.slot_bytes; }
  // Entire pool allocation (for CommAgent::register_slab / NIXL).
  void* slab_ptr() const { return slab_.ptr; }
  size_t slab_bytes() const { return slab_.bytes; }
  // returns a pointer into the cache slot
  StatusOr<void*> slot_ptr(const CacheHandle& handle) const;

  CachePool(const CachePool&) = delete;
  CachePool& operator=(const CachePool&) = delete;
};

template <typename Config, typename LayoutBuilder>
StatusOr<std::unique_ptr<CachePool>> create_cache_pool(const Config& cfg,
                                                       DeviceAllocator* allocator, int num_slots,
                                                       LayoutBuilder build_layout) {
  auto layout = build_layout(cfg);
  if (!layout.ok()) {
    return layout.status();
  }

  return CachePool::create(std::move(layout.value()), allocator, num_slots);
}