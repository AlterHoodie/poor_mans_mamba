#include "runtime/ipc/process_comm.h"

#include <cerrno>
#include <cstring>

#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

// Upper bound for one frame; guards against a corrupt length prefix.
// Protobuf Frame layout: [4-byte native uint32 length][protobuf body]
constexpr uint32_t kMaxFrameBytes = 256u * 1024u * 1024u; // max body size 256 MiB
constexpr size_t kHeaderBytes = sizeof(uint32_t);

} // namespace

ProcessChannel::ProcessChannel(int fd) : fd_(fd) {}

ProcessChannel::~ProcessChannel() {
  if (fd_ >= 0) {
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    fd_ = -1;
  }
}

Status ProcessChannel::send(const mambaserve::Envelope& msg) {
  if (fd_ < 0)
    return Status::RuntimeError("Channel Closed");

  const size_t body = msg.ByteSizeLong();
  if (body > kMaxFrameBytes)
    return Status::InvalidArgument("envelope too large for ProcessChannel");

  std::string frame(kHeaderBytes + body, '\0');
  const uint32_t len = static_cast<uint32_t>(body);
  std::memcpy(frame.data(), &len, sizeof(len));
  if (body > 0 && !msg.SerializeToArray(frame.data() + kHeaderBytes, static_cast<int>(body)))
    return Status::RuntimeError("failed to serialize envelope");

  std::lock_guard<std::mutex> lk(send_mu_);
  size_t off = 0;
  while (off < frame.size()) {
    // MSG_NOSIGNAL: a dead peer must surface as an error, not SIGPIPE-[kills the process].
    const ssize_t n = ::send(fd_, frame.data() + off, frame.size() - off, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return Status::RuntimeError(std::string("Channel Closed: send failed: ") +
                                  std::strerror(errno));
    }
    off += static_cast<size_t>(n);
  }
  return Status::Ok();
}

StatusOr<mambaserve::Envelope> ProcessChannel::read_frame_(bool block) {
  if (fd_ < 0)
    return Status::RuntimeError("Channel Closed");

  std::lock_guard<std::mutex> lk(recv_mu_);
  char tmp[64 * 1024];
  for (;;) {
    if (rx_.size() >= kHeaderBytes) {
      uint32_t len = 0;
      std::memcpy(&len, rx_.data(), sizeof(len));
      if (len > kMaxFrameBytes)
        return Status::RuntimeError("corrupt frame length on ProcessChannel");
      
      // if we have recieved complete envelope data bytes
      if (rx_.size() >= kHeaderBytes + len) {
        mambaserve::Envelope env;
        const bool parsed =
            env.ParseFromArray(rx_.data() + kHeaderBytes, static_cast<int>(len));
        rx_.erase(0, kHeaderBytes + len);
        if (!parsed)
          return Status::RuntimeError("failed to parse envelope from ProcessChannel");
        return env;
      }
    }

    const ssize_t n = ::recv(fd_, tmp, sizeof(tmp), block ? 0 : MSG_DONTWAIT);
    
    // if we have read some bytes
    if (n > 0) {
      rx_.append(tmp, static_cast<size_t>(n));
      continue;
    }
    if (n == 0)
      return Status::RuntimeError("Channel Closed");

    // if recv was interrupted and kernel aborted, try again
    if (errno == EINTR)
      continue;
    // if dont block, and nothing was read, return empty
    if (!block && (errno == EAGAIN || errno == EWOULDBLOCK))
      return Status::Empty();

    // else some error close the channel
    return Status::RuntimeError(std::string("Channel Closed: recv failed: ") +
                                std::strerror(errno));
  }
}

StatusOr<mambaserve::Envelope> ProcessChannel::recv() { return read_frame_(true); }

StatusOr<mambaserve::Envelope> ProcessChannel::try_recv() { return read_frame_(false); }

void ProcessChannel::close() {
  if (fd_ >= 0)
    ::shutdown(fd_, SHUT_RDWR);
}

StatusOr<ProcessChannelFds> make_process_socketpair() {
  int fds[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) != 0)
    return Status::RuntimeError(std::string("socketpair failed: ") + std::strerror(errno));
  return ProcessChannelFds{.parent_fd = fds[0], .child_fd = fds[1]};
}

StatusOr<std::pair<std::unique_ptr<IpcChannel>, std::unique_ptr<IpcChannel>>>
make_process_channel_pair() {
  StatusOr<ProcessChannelFds> fds_or = make_process_socketpair();
  if (!fds_or.ok())
    return fds_or.status();
  const ProcessChannelFds fds = fds_or.value();
  std::unique_ptr<IpcChannel> parent = std::make_unique<ProcessChannel>(fds.parent_fd);
  std::unique_ptr<IpcChannel> child = std::make_unique<ProcessChannel>(fds.child_fd);
  return std::make_pair(std::move(parent), std::move(child));
}
