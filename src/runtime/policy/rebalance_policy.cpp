#include "runtime/policy/rebalance_policy.h"

#include <algorithm>
#include <optional>

std::optional<RebalanceDecision> SimpleRebalancePolicy::check(
    std::span<const WorkerStat> views, std::span<const SessionView> sessions) {
  if (views.empty() || sessions.empty())
    return std::nullopt;

  for (const SessionView& s : sessions) {
    if (s.phase == SessionPhase::Migrating || s.migrate_pending)
      return std::nullopt;
  }

  auto free_slots = [](const WorkerStat& w) { return w.capacity - w.inflight; };

  auto hot = std::max_element(views.begin(), views.end(),
                              [&](const WorkerStat& a, const WorkerStat& b) {
                                const int a_free = free_slots(a);
                                const int b_free = free_slots(b);
                                if (a_free != b_free)
                                  return a_free > b_free; // least free = hotter
                                return a.inflight < b.inflight;
                              });

  auto cold = std::max_element(views.begin(), views.end(),
                               [&](const WorkerStat& a, const WorkerStat& b) {
                                 const int a_free = free_slots(a);
                                 const int b_free = free_slots(b);
                                 if (a_free != b_free)
                                   return a_free < b_free; // most free = colder
                                 return a.inflight > b.inflight;
                               });

  if (hot == views.end() || cold == views.end())
    return std::nullopt;
  if (hot->index == cold->index)
    return std::nullopt;
  if (free_slots(*cold) <= 0)
    return std::nullopt;

  const int gap = free_slots(*cold) - free_slots(*hot);
  if (gap < imbalance_threshold_)
    return std::nullopt;

  const SessionView* victim = nullptr;
  for (const SessionView& s : sessions) {
    if (s.worker_idx != hot->index)
      continue;
    if (s.phase != SessionPhase::Decoding || s.migrate_pending)
      continue;
    if (victim == nullptr || s.gen_len > victim->gen_len ||
        (s.gen_len == victim->gen_len && s.req_id < victim->req_id)) {
      victim = &s;
    }
  }
  if (victim == nullptr)
    return std::nullopt;

  return RebalanceDecision{.victim_req_id = victim->req_id, .dst_idx = cold->index};
}
