#include "runtime/policy/placement_policy.h"

#include "core/status.h"

#include <algorithm>

StatusOr<size_t> SimplePlacementPolicy::place(std::span<const int32_t> tokens,
                                              const GenerateParams& params,
                                              std::span<const WorkerStat> views) {
  (void)tokens;
  (void)params;

  if (views.empty())
    return Status::RuntimeError("no workers available");

  auto best = std::max_element(views.begin(), views.end(),
                               [](const WorkerStat& a, const WorkerStat& b) {
                                 const int a_free = a.capacity - a.inflight;
                                 const int b_free = b.capacity - b.inflight;
                                 if (a_free != b_free)
                                   return a_free < b_free; // prefer more free slots
                                 return a.inflight > b.inflight; // tie-break: fewer inflight
                               });

  const int free_slots = best->capacity - best->inflight;
  if (free_slots <= 0)
    return Status::OOM("no free cache slots on any worker");

  return best->index;
}
