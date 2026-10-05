#include "runtime/ipc/process_comm.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

namespace {

mambaserve::Envelope prefill_env(uint64_t req_id, int n_tokens) {
  mambaserve::Envelope env;
  auto* p = env.mutable_cmd()->mutable_prefill();
  p->set_req_id(req_id);
  p->set_max_new_tokens(3);
  p->set_eos_id(7);
  for (int i = 0; i < n_tokens; ++i)
    p->add_tokens(i);
  return env;
}

} // namespace

TEST(ProcessChannel, RoundTripsEnvelopesInBothDirections) {
  auto pair_or = make_process_channel_pair();
  ASSERT_TRUE(pair_or.ok()) << pair_or.status().message();
  auto& [parent, child] = pair_or.value();

  ASSERT_TRUE(parent->send(prefill_env(11, 4)).ok());
  auto got = child->recv();
  ASSERT_TRUE(got.ok()) << got.status().message();
  ASSERT_EQ(got.value().body_case(), mambaserve::Envelope::kCmd);
  EXPECT_EQ(got.value().cmd().prefill().req_id(), 11u);
  EXPECT_EQ(got.value().cmd().prefill().tokens_size(), 4);

  mambaserve::Envelope ev;
  ev.mutable_event()->mutable_ready()->set_worker_idx(3);
  ASSERT_TRUE(child->send(ev).ok());
  auto back = parent->recv();
  ASSERT_TRUE(back.ok());
  ASSERT_EQ(back.value().body_case(), mambaserve::Envelope::kEvent);
  EXPECT_EQ(back.value().event().ready().worker_idx(), 3u);
}

TEST(ProcessChannel, PreservesMessageOrderAndEmptyEnvelope) {
  auto pair_or = make_process_channel_pair();
  ASSERT_TRUE(pair_or.ok());
  auto& [parent, child] = pair_or.value();

  ASSERT_TRUE(parent->send(prefill_env(1, 1)).ok());
  ASSERT_TRUE(parent->send(mambaserve::Envelope{}).ok()); // zero-length frame
  ASSERT_TRUE(parent->send(prefill_env(2, 2)).ok());

  auto a = child->recv();
  auto b = child->recv();
  auto c = child->recv();
  ASSERT_TRUE(a.ok() && b.ok() && c.ok());
  EXPECT_EQ(a.value().cmd().prefill().req_id(), 1u);
  EXPECT_EQ(b.value().body_case(), mambaserve::Envelope::BODY_NOT_SET);
  EXPECT_EQ(c.value().cmd().prefill().req_id(), 2u);
}

TEST(ProcessChannel, TryRecvIsEmptyUntilAMessageArrives) {
  auto pair_or = make_process_channel_pair();
  ASSERT_TRUE(pair_or.ok());
  auto& [parent, child] = pair_or.value();

  auto none = child->try_recv();
  ASSERT_FALSE(none.ok());
  EXPECT_EQ(none.status().code(), Code::kEmpty);

  ASSERT_TRUE(parent->send(prefill_env(5, 2)).ok());
  auto got = child->try_recv();
  ASSERT_TRUE(got.ok()) << got.status().message();
  EXPECT_EQ(got.value().cmd().prefill().req_id(), 5u);

  EXPECT_EQ(child->try_recv().status().code(), Code::kEmpty);
}

TEST(ProcessChannel, HandlesFramesLargerThanTheSocketBuffer) {
  auto pair_or = make_process_channel_pair();
  ASSERT_TRUE(pair_or.ok());
  auto& [parent, child] = pair_or.value();

  constexpr int kTokens = 1 << 20; // ~4 MiB on the wire
  Status send_status = Status::Ok();
  std::thread writer([&] { send_status = parent->send(prefill_env(9, kTokens)); });

  // Poll with try_recv so partial frames are buffered across calls.
  StatusOr<mambaserve::Envelope> got = Status::Empty();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  do {
    got = child->try_recv();
    if (!got.ok() && got.status().code() == Code::kEmpty)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (!got.ok() && got.status().code() == Code::kEmpty &&
           std::chrono::steady_clock::now() < deadline);
  writer.join();

  ASSERT_TRUE(send_status.ok()) << send_status.message();
  ASSERT_TRUE(got.ok());
  EXPECT_EQ(got.value().cmd().prefill().tokens_size(), kTokens);
  EXPECT_EQ(got.value().cmd().prefill().tokens(kTokens - 1), kTokens - 1);
}

TEST(ProcessChannel, ConcurrentSendersNeverInterleaveFrames) {
  auto pair_or = make_process_channel_pair();
  ASSERT_TRUE(pair_or.ok());
  auto& [parent, child] = pair_or.value();

  constexpr int kThreads = 4;
  constexpr int kPerThread = 200;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kPerThread; ++i)
        ASSERT_TRUE(parent->send(prefill_env(static_cast<uint64_t>(t), 64)).ok());
    });
  }
  int received = 0;
  for (int i = 0; i < kThreads * kPerThread; ++i) {
    auto got = child->recv();
    ASSERT_TRUE(got.ok()) << got.status().message();
    EXPECT_EQ(got.value().cmd().prefill().tokens_size(), 64);
    ++received;
  }
  for (auto& th : threads)
    th.join();
  EXPECT_EQ(received, kThreads * kPerThread);
}

TEST(ProcessChannel, CloseUnblocksPendingRecvAndFailsSend) {
  auto pair_or = make_process_channel_pair();
  ASSERT_TRUE(pair_or.ok());
  auto& [parent, child] = pair_or.value();

  StatusOr<mambaserve::Envelope> result = Status::Empty();
  std::thread reader([&] { result = child->recv(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  parent->close(); // peer sees EOF
  reader.join();
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.status().code(), Code::kEmpty);

  // Sending into a closed channel is an error, not a SIGPIPE.
  EXPECT_FALSE(child->send(prefill_env(1, 1)).ok());
  EXPECT_FALSE(parent->send(prefill_env(1, 1)).ok());
}

TEST(ProcessChannel, QueuedMessagesSurviveCloseBeforeEof) {
  auto pair_or = make_process_channel_pair();
  ASSERT_TRUE(pair_or.ok());
  auto& [parent, child] = pair_or.value();

  ASSERT_TRUE(parent->send(prefill_env(21, 2)).ok());
  parent->close();

  auto got = child->recv();
  ASSERT_TRUE(got.ok()) << got.status().message();
  EXPECT_EQ(got.value().cmd().prefill().req_id(), 21u);
  EXPECT_FALSE(child->recv().ok());
}
