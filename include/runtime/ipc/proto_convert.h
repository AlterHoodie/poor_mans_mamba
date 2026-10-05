#pragma once

#include "comm/comm_agent.h"
#include "core/status.h"
#include "proto/worker.pb.h"

inline mambaserve::Code to_proto(Code c) {
  switch (c) {
  case Code::kOk:
    return mambaserve::CODE_OK;
  case Code::kNotFound:
    return mambaserve::CODE_NOT_FOUND;
  case Code::kInvalidArgument:
    return mambaserve::CODE_INVALID_ARGUMENT;
  case Code::kOOM:
    return mambaserve::CODE_OOM;
  case Code::kKvCacheOverflow:
    return mambaserve::CODE_KV_CACHE_OVERFLOW;
  case Code::kRuntimeError:
    return mambaserve::CODE_RUNTIME_ERROR;
  case Code::kNotImplemented:
    return mambaserve::CODE_NOTIMPLEMENTED;
  case Code::kEmpty:
    return mambaserve::CODE_RUNTIME_ERROR;
  }
  return mambaserve::CODE_RUNTIME_ERROR;
}

inline Code from_proto(mambaserve::Code c) {
  switch (c) {
  case mambaserve::CODE_OK:
    return Code::kOk;
  case mambaserve::CODE_NOT_FOUND:
    return Code::kNotFound;
  case mambaserve::CODE_INVALID_ARGUMENT:
    return Code::kInvalidArgument;
  case mambaserve::CODE_OOM:
    return Code::kOOM;
  case mambaserve::CODE_KV_CACHE_OVERFLOW:
    return Code::kKvCacheOverflow;
  case mambaserve::CODE_RUNTIME_ERROR:
    return Code::kRuntimeError;
  case mambaserve::CODE_NOTIMPLEMENTED:
    return Code::kNotImplemented;
  default:
    return Code::kRuntimeError;
  }
}

inline mambaserve::StatusMsg to_proto(const Status& s) {
  mambaserve::StatusMsg m;
  m.set_code(to_proto(s.code()));
  m.set_message(s.message());
  return m;
}

inline Status from_proto(const mambaserve::StatusMsg& m) {
  switch (from_proto(m.code())) {
  case Code::kOk:
    return Status::Ok();
  case Code::kNotFound:
    return Status::NotFound(m.message());
  case Code::kInvalidArgument:
    return Status::InvalidArgument(m.message());
  case Code::kOOM:
    return Status::OOM(m.message());
  case Code::kKvCacheOverflow:
    return Status::KvCacheOverflow(m.message());
  case Code::kNotImplemented:
    return Status::NotImplemented(m.message());
  case Code::kRuntimeError:
  case Code::kEmpty:
  default:
    return Status::RuntimeError(m.message());
  }
}

inline mambaserve::XferRole to_proto(XferRole r) {
  return r == XferRole::Recv ? mambaserve::XFER_RECV : mambaserve::XFER_SEND;
}

inline XferRole from_proto(mambaserve::XferRole r) {
  return r == mambaserve::XFER_RECV ? XferRole::Recv : XferRole::Send;
}

inline bool status_ok(const mambaserve::StatusMsg& m) {
  return m.code() == mambaserve::CODE_OK;
}
