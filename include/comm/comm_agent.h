#pragma once

#include "core/device.h"
#include "core/status.h"
#include "proto/worker.pb.h"

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <optional>


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
        virtual int64_t xfer_seq_len(const XferHandle& ) { return 0; }

        // Control-plane hooks (default: no control traffic).
        // After register_slab() succeeded and the worker reported Ready, returns a
        // control message the worker should send up to the parent once (e.g. NIXL
        // publishing its agent metadata), if any.
        virtual std::optional<mambaserve::TransportControl> make_register_announce() {
            return std::nullopt;
        }
        // After a successful post(), returns a control message the worker should
        // send up to the parent (e.g. NIXL Recv slot announce), if any.
        virtual std::optional<mambaserve::TransportControl> make_post_announce(const XferDesc& ) {
            return std::nullopt;
        }
        // Called once when a posted transfer leaves Pending (Done or Error).
        // Returns a control message the worker should send up to the parent
        // (e.g. MemcpyPeer Send publishing "copy done" to the Recv side), if any.
        virtual std::optional<mambaserve::TransportControl>
        make_completion_announce(const XferHandle& , XferState ) {
            return std::nullopt;
        }
        // Apply a control message the parent routed to this worker.
        virtual Status handle_transport(const mambaserve::TransportControl& ) {
            return Status::Ok();
        }
        // Reply produced by the last handle_transport(), if any (e.g. NCCL
        // bootstrap ack). The worker sends it up to the parent.
        virtual std::optional<mambaserve::TransportControl> take_transport_reply() {
            return std::nullopt;
        }
        
        // shut the communication agent down
        virtual void shutdown() = 0;
};