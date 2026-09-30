#pragma once

#include <unordered_map>

#include "comm/comm_agent.h"

struct Entry{
    XferDesc desc;
    XferState state;
    int polls = 3;
};

class FakeCommAgent : public CommAgent {
    private:
        std::unordered_map<uint64_t, Entry> pending_transfers_;
    public: 
        FakeCommAgent(int device_id) : CommAgent(device_id) {};

        Status register_slab(void* /*ptr*/, size_t /*bytes*/) override { return Status::Ok(); }
        
        StatusOr<XferHandle> post(XferDesc& desc) override;
        
        XferState poll(const XferHandle& handle) override;

        void shutdown() override {return;}; 
};