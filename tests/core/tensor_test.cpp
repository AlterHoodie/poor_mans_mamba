#include "core/tensor.h"

#include <gtest/gtest.h>

#include <cstdlib>

TEST(Tensor, NumelIsProduct) {
    Tensor t;
    t.shape = {2, 3, 4};
    ASSERT_TRUE(t.numel().ok());
    EXPECT_EQ(t.numel().value(), 24);
}

TEST(Tensor, RowsColsFlattenTo2D) {
    Tensor t;
    t.shape = {2, 3, 4};
    ASSERT_TRUE(t.rows().ok());
    ASSERT_TRUE(t.cols().ok());
    EXPECT_EQ(t.rows().value(), 6);
    EXPECT_EQ(t.cols().value(), 4);

    t.shape = {768, 4096};
    EXPECT_EQ(t.rows().value(), 768);
    EXPECT_EQ(t.cols().value(), 4096);
}

TEST(Tensor, RowsColsRejectRank1) {
    Tensor t;
    t.shape = {4096};
    EXPECT_FALSE(t.rows().ok());
    EXPECT_FALSE(t.cols().ok());
    ASSERT_TRUE(t.numel().ok());
    EXPECT_EQ(t.numel().value(), 4096);
    ASSERT_TRUE(t.last_dim().ok());
    EXPECT_EQ(t.last_dim().value(), 4096);
    ASSERT_TRUE(t.batch_size().ok());
    EXPECT_EQ(t.batch_size().value(), 1);
}

TEST(Tensor, BatchSizeFlattensLeadingDims) {
    Tensor t;
    t.shape = {2, 3, 4};
    ASSERT_TRUE(t.batch_size().ok());
    EXPECT_EQ(t.batch_size().value(), 6);
    ASSERT_TRUE(t.last_dim().ok());
    EXPECT_EQ(t.last_dim().value(), 4);
}

TEST(Tensor, EmptyWhenNoDataOrZeroDim) {
    Tensor t;
    t.shape = {2, 3};
    EXPECT_TRUE(t.empty());

    t.buffer.data = std::malloc(6 * sizeof(float));
    t.buffer.bytes = 6 * sizeof(float);
    EXPECT_FALSE(t.empty());

    t.shape = {2, 0};
    EXPECT_TRUE(t.empty());
}
