#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "core/status.h"
#include "proto/worker.pb.h"
#include "runtime/ipc/ipc_comm.h"

// IpcChannel over a connected Unix stream socket, for workers hosted in separate
// processes. Each Envelope is framed as a little-endian uint32 length followed by
// its serialized protobuf bytes.
//
// send() is thread-safe (frames are never interleaved). recv()/try_recv() are meant
// for a single reader thread. close() shuts the socket down in both directions, which
// unblocks a pending recv() here and gives the peer EOF; the fd itself is released
// by the destructor.
class ProcessChannel : public IpcChannel {
public:
  // Takes ownership of `fd`.
  explicit ProcessChannel(int fd);
  ~ProcessChannel() override;

  ProcessChannel(const ProcessChannel&) = delete;
  ProcessChannel& operator=(const ProcessChannel&) = delete;

  Status send(const mambaserve::Envelope& msg) override;
  StatusOr<mambaserve::Envelope> recv() override;
  StatusOr<mambaserve::Envelope> try_recv() override;
  void close() override;

  int fd() const { return fd_; }

private:
  // Returns Code::kEmpty only when !block and no complete frame is available yet.
  StatusOr<mambaserve::Envelope> read_frame_(bool block);

  int fd_ = -1;
  std::mutex send_mu_;
  std::mutex recv_mu_;
  std::string rx_; // bytes read from the socket but not yet consumed as a frame
};

// parent = send cmds / recv events; child = recv cmds / send events.
// Both fds are close-on-exec; a spawner must clear that flag on the child's fd.
struct ProcessChannelFds {
  int parent_fd = -1;
  int child_fd = -1;
};
StatusOr<ProcessChannelFds> make_process_socketpair();

StatusOr<std::pair<std::unique_ptr<IpcChannel>, std::unique_ptr<IpcChannel>>>
make_process_channel_pair();
