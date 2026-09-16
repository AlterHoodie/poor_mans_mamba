#include "core/status.h"
#include <gtest/gtest.h>

#include <string>

TEST(Status, OkIsTruthyAndEmptyMessage) {
  const Status s = Status::Ok();
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.code(), Code::kOk);
  EXPECT_TRUE(s.message().empty());
}

TEST(Status, ErrorFactoriesSetCodeAndMessage) {
  const Status not_found = Status::NotFound("missing");
  EXPECT_FALSE(not_found.ok());
  EXPECT_EQ(not_found.code(), Code::kNotFound);
  EXPECT_EQ(not_found.message(), "missing");

  const Status invalid = Status::InvalidArgument("bad arg");
  EXPECT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.code(), Code::kInvalidArgument);
  EXPECT_EQ(invalid.message(), "bad arg");

  const Status oom = Status::OOM("out of memory");
  EXPECT_FALSE(oom.ok());
  EXPECT_EQ(oom.code(), Code::kOOM);
  EXPECT_EQ(oom.message(), "out of memory");
}

TEST(StatusOr, HoldsValue) {
  StatusOr<int> result(42);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), 42);
}

TEST(StatusOr, HoldsError) {
  StatusOr<int> result(Status::NotFound("gone"));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), Code::kNotFound);
  EXPECT_EQ(result.status().message(), "gone");
}

namespace {

StatusOr<int> maybe_parse(bool succeed) {
  if (!succeed) {
    return Status::InvalidArgument("parse failed");
  }
  return 7;
}

Status assign_helper(bool succeed, int& out) {
  ASSIGN_OR_RETURN(out, maybe_parse(succeed));
  return Status::Ok();
}

} // namespace

TEST(StatusOr, AssignOrReturnPropagatesError) {
  int value = 0;
  const Status bad = assign_helper(false, value);
  EXPECT_FALSE(bad.ok());
  EXPECT_EQ(bad.code(), Code::kInvalidArgument);
  EXPECT_EQ(value, 0);

  const Status good = assign_helper(true, value);
  EXPECT_TRUE(good.ok());
  EXPECT_EQ(value, 7);
}
