#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

#include "core/status.h"
#include "proto/worker.pb.h"
#include "runtime/ipc/ipc_comm.h"

struct MailBox {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<mambaserve::Envelope> q;
  bool closed = false;
};

class ThreadChannel : public IpcChannel {
public:
  ThreadChannel(std::shared_ptr<MailBox> in, std::shared_ptr<MailBox> out)
      : in_(std::move(in)), out_(std::move(out)) {}

  Status send(const mambaserve::Envelope& msg) override;
  StatusOr<mambaserve::Envelope> recv() override;
  StatusOr<mambaserve::Envelope> try_recv() override;
  void close() override;

private:
  std::shared_ptr<MailBox> in_;
  std::shared_ptr<MailBox> out_;
};

// parent = send cmds / recv events; child = recv cmds / send events
inline std::pair<std::unique_ptr<IpcChannel>, std::unique_ptr<IpcChannel>> make_thread_channel_pair() {
  auto a = std::make_shared<MailBox>(); // parent → child
  auto b = std::make_shared<MailBox>(); // child → parent
  auto parent = std::make_unique<ThreadChannel>(/*in=*/b, /*out=*/a);
  auto child = std::make_unique<ThreadChannel>(/*in=*/a, /*out=*/b);
  return {std::move(parent), std::move(child)};
}
