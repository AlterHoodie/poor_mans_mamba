#include <atomic>
#include <cstdint>
#include <string>

#include "comm/fake_comm_agent.h"
#include "comm/comm_agent.h"
#include "core/status.h"

StatusOr<XferHandle> FakeCommAgent::post(XferDesc& desc){
    uint64_t id = id_counter_.fetch_add(1,std::memory_order_relaxed);
    pending_transfers_[id] = Entry{
        .desc = desc,
        .state = XferState::Pending
    };

    return XferHandle{.id = id};
}

XferState FakeCommAgent::poll(const XferHandle& handle){
    auto it = pending_transfers_.find(handle.id);
    if(it == pending_transfers_.end()) return XferState::Error;
    if(it->second.polls == 0) {
        pending_transfers_.erase(it);
        return XferState::Done;
    }
    it->second.polls--;
    return XferState::Pending;
}