#pragma once

#include <cassert>
#include <string>

enum class Code { kOk, kNotFound, kInvalidArgument, kOOM };

class Status {
private:
  Code code_;
  std::string message_;

  Status(Code code, std::string message) : code_(code), message_(message) {}

public:
  static Status Ok() { return Status(Code::kOk, ""); }
  static Status NotFound(std::string msg) { return Status(Code::kNotFound, std::move(msg)); }
  static Status InvalidArgument(std::string msg) {
    return Status(Code::kInvalidArgument, std::move(msg));
  }
  static Status OOM(std::string msg) { return Status(Code::kOOM, std::move(msg)); }

  bool ok() const { return code_ == Code::kOk; }

  const std::string& message() const { return message_; }

  Code code() const { return code_; }
};

template <typename T> class StatusOr {
private:
  bool ok_;
  T obj_;
  Status s_;

public:
  StatusOr(T obj) : obj_(std::move(obj)), ok_(true), s_(Status::Ok()){};
  StatusOr(const Status& stat) : ok_(false), s_(stat) { assert(!s_.ok()); }
  StatusOr(Status&& stat) : s_(stat), ok_(false) { assert(!s_.ok()); };

  bool ok() const { return ok_; }

  T& value() {
    assert(ok_);
    return obj_;
  }

  const T& value() const {
    assert(ok_);
    return obj_;
  }

  const Status& status() const {
    assert(!ok_);
    return s_;
  }
};

#define ASSIGN_OR_RETURN(lhs, expr)                                                                \
  do {                                                                                             \
    auto _statusor = (expr);                                                                       \
    if (!_statusor.ok()) {                                                                         \
      return Status(_statusor.status());                                                           \
    }                                                                                              \
    lhs = std::move(_statusor.value());                                                            \
  } while (0)
