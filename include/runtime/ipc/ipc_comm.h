#pragma once

#include "core/status.h"
#include "proto/worker.pb.h"

class IpcChannel {
public:
  virtual ~IpcChannel() = default;

  virtual Status send(const mambaserve::Envelope& msg) = 0;
  virtual StatusOr<mambaserve::Envelope> recv() = 0;
  // Returns Code::kEmpty when nothing is ready (non-blocking).
  virtual StatusOr<mambaserve::Envelope> try_recv() = 0;
  virtual void close() {}
};
