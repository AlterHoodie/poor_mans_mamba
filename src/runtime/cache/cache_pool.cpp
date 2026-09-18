#include "runtime/cache/cache_pool.h"

#include <cstring>
#include <limits>

CachePool::CachePool(CacheLayout layout, DeviceAllocator* allocator)
    : device_(allocator->kind()), device_id_(allocator->device_id()), allocator_(allocator),
      layout_(std::move(layout)) {}

// Static Method to create cachepool objects (Mini Factory thingy)
StatusOr<std::unique_ptr<CachePool>> CachePool::create(CacheLayout layout,
                                                       DeviceAllocator* allocator, int num_slots) {
  if (allocator == nullptr) {
    return Status::InvalidArgument("allocator cannot be null");
  }
  if (num_slots <= 0) {
    return Status::InvalidArgument("num_slots must be positive");
  }
  if (layout.slot_bytes == 0) {
    return Status::InvalidArgument("layout slot_bytes must be positive");
  }
  if (layout.num_layers <= 0 || layout.layers.size() != static_cast<size_t>(layout.num_layers)) {
    return Status::InvalidArgument("layout layer count is inconsistent");
  }

  const size_t slot_count = static_cast<size_t>(num_slots);
  if (layout.slot_bytes > std::numeric_limits<size_t>::max() / slot_count) {
    return Status::OOM("cache pool slab size overflow");
  }

  auto pool = std::unique_ptr<CachePool>(new CachePool(std::move(layout), allocator));

  auto slab = pool->allocator_->allocate(pool->layout_.slot_bytes * slot_count);
  if (!slab.ok()) {
    return slab.status();
  }
  pool->slab_ = std::move(slab.value());

  Status zero_status = pool->allocator_->memset_zero(pool->slab_, 0, pool->slab_.bytes);
  if (!zero_status.ok()) {
    pool->allocator_->free(pool->slab_);
    return zero_status;
  }

  pool->slots_.resize(slot_count);
  pool->free_list_.reserve(slot_count);

  auto* slab_base = static_cast<std::byte*>(pool->slab_.ptr);
  for (int slot_id = 0; slot_id < num_slots; ++slot_id) {
    Slot& slot = pool->slots_[static_cast<size_t>(slot_id)];
    slot.views.resize(static_cast<size_t>(pool->layout_.num_layers));

    auto* slot_base = slab_base + static_cast<size_t>(slot_id) * pool->layout_.slot_bytes;
    for (int layer = 0; layer < pool->layout_.num_layers; ++layer) {
      const LayerEntry& entry = pool->layout_.layers[static_cast<size_t>(layer)];

      // Direct views into the Device Memory Slab
      slot.views[static_cast<size_t>(layer)] = LayerCacheView{
          .kind = entry.kind,
          .conv = {.ptr = slot_base + entry.conv.offset, .bytes = entry.conv.bytes},
          .ssm = {.ptr = slot_base + entry.ssm.offset, .bytes = entry.ssm.bytes},
          .k = {.ptr = slot_base + entry.k.offset, .bytes = entry.k.bytes},
          .v = {.ptr = slot_base + entry.v.offset, .bytes = entry.v.bytes},
      };
    }

    slot.seq_len = 0;
    pool->free_list_.push_back(slot_id);
  }

  return std::move(pool);
}

StatusOr<CacheHandle> CachePool::acquire() {
  if (free_list_.empty()) {
    return Status::OOM("Slot Memory Pool Exhausted");
  }

  const int slot_id = free_list_.back();
  free_list_.pop_back();
  slots_[static_cast<size_t>(slot_id)].in_use = true;
  slots_[static_cast<size_t>(slot_id)].seq_len = 0;
  return CacheHandle(slot_id);
}

Status CachePool::zero_slot_(int slot_id) {
  if (slot_id < 0 || static_cast<size_t>(slot_id) >= slots_.size()) {
    return Status::InvalidArgument("Invalid slot id");
  }

  Slot& slot = slots_[static_cast<size_t>(slot_id)];
  slot.seq_len = 0;

  if (allocator_ != nullptr && slab_.ptr != nullptr) {
    const size_t offset = static_cast<size_t>(slot_id) * layout_.slot_bytes;
    return allocator_->memset_zero(slab_, offset, layout_.slot_bytes);
  }

  auto* slot_base = static_cast<std::byte*>(slab_.ptr) + slot_id * layout_.slot_bytes;
  size_t slot_bytes = layout_.slot_bytes; // same for every slot

  // Keep allocation and views; clear recurrent state only.
  std::memset(slot_base, 0, slot_bytes);
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

StatusOr<LayerCacheView> CachePool::layer_view(const CacheHandle& handle, int layer) const {
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

StatusOr<int64_t> CachePool::seq_len(const CacheHandle& handle) const {
  if (!handle.valid())
    return Status::InvalidArgument("invalid cache handle");
  const int slot_id = handle.id();
  if (static_cast<size_t>(slot_id) >= slots_.size() ||
      !slots_[static_cast<size_t>(slot_id)].in_use) {
    return Status::InvalidArgument("Slot is not in use");
  }
  return slots_[static_cast<size_t>(slot_id)].seq_len;
}

Status CachePool::set_seq_len(const CacheHandle& handle, int64_t len) {
  if (!handle.valid())
    return Status::InvalidArgument("invalid cache handle");
  if (len < 0)
    return Status::InvalidArgument("seq_len must be non-negative");
  const int slot_id = handle.id();
  if (static_cast<size_t>(slot_id) >= slots_.size() ||
      !slots_[static_cast<size_t>(slot_id)].in_use) {
    return Status::InvalidArgument("Slot is not in use");
  }
  slots_[static_cast<size_t>(slot_id)].seq_len = len;
  return Status::Ok();
}
