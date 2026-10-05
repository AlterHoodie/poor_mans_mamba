#include "runtime/ipc/thread_comm.h"

#include <utility>

Status ThreadChannel::send(const mambaserve::Envelope& msg) {
  if (!out_)
    return Status::RuntimeError("OUT Mailbox not initialized");
  {
    std::lock_guard<std::mutex> lk(out_->mu);
    if (out_->closed)
      return Status::RuntimeError("Channel Closed");
    out_->q.push_back(msg);
  }
  out_->cv.notify_one();
  return Status::Ok();
}

StatusOr<mambaserve::Envelope> ThreadChannel::recv() {
  if (!in_)
    return Status::RuntimeError("IN Mailbox not initialized");
  mambaserve::Envelope msg;
  {
    std::unique_lock<std::mutex> lk(in_->mu);
    in_->cv.wait(lk, [&]() { return !in_->q.empty() || in_->closed; });
    if (in_->q.empty() && in_->closed)
      return Status::RuntimeError("Channel Closed");
    msg = std::move(in_->q.front());
    in_->q.pop_front();
  }
  return msg;
}

StatusOr<mambaserve::Envelope> ThreadChannel::try_recv() {
  if (!in_)
    return Status::RuntimeError("IN Mailbox not initialized");
  mambaserve::Envelope msg;
  {
    std::lock_guard<std::mutex> lk(in_->mu);
    if (!in_->q.empty()) {
      msg = std::move(in_->q.front());
      in_->q.pop_front();
    } else if (in_->closed) {
      return Status::RuntimeError("Channel Closed");
    } else {
      return Status::Empty();
    }
  }
  return msg;
}

void ThreadChannel::close() {
  auto close_box = [](const std::shared_ptr<MailBox>& box) {
    if (!box)
      return;
    {
      std::lock_guard<std::mutex> lk(box->mu);
      box->closed = true;
    }
    box->cv.notify_all();
  };
  close_box(in_);
  close_box(out_);
}
