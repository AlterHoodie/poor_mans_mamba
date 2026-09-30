#pragma once

#include "core/device.h"
#include "core/status.h"

#include <cstddef>
#include <atomic>
#include <cstdint>


enum class XferRole {Send, Recv};
enum class XferState {Pending, Done, Error};

struct XferHandle {uint64_t id = 0;};

struct XferDesc{
    void* local_ptr = nullptr;
    void* remote_ptr = nullptr;
    size_t bytes = 0;

    int peer_device_id = -1;
    XferRole role = XferRole::Send;
    // Correlates Send+Recv sides of one migrate (typically req_id).
    uint64_t xfer_id = 0;
    // Send publishes cache metadata; Recv applies after copy completes.
    int64_t seq_len = 0;
};



// Responsible for tensor transfers between two gpu devices
class CommAgent{
    private: 
        int device_id_;
    protected:
        std::atomic<uint64_t> id_counter_;
    public:
        CommAgent(int device_id) : device_id_(device_id) {};
        virtual ~CommAgent() = default;

        int device_id() const { return device_id_; }

        // register the entire cachepool slab as potential transfer target
        // meaning any slot in the slab can be transferred
        // only use for NIXL not for NCCL(send/recv) or memcpypeer
        virtual Status register_slab(void* ptr, size_t bytes) = 0;

        // Post a XferDesc to start the transfer between two gpus
        virtual StatusOr<XferHandle> post(XferDesc& desc) = 0;
        
        // Poll the transfer status
        virtual XferState poll(const XferHandle& handle) = 0;

        // Seq_len published by the completed transfer (Recv side after copy).
        virtual int64_t xfer_seq_len(const XferHandle& /*handle*/) { return 0; }
        
        // shut the communication agent down
        virtual void shutdown() = 0;
};