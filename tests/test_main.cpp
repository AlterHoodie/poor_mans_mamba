#include "worker_process.h"

#include <gtest/gtest.h>

// Process-mode tests spawn this very binary as a worker (`--mambaserve-worker ...`),
// so main() has to serve as a worker before gtest sees the arguments.
int main(int argc, char** argv) {
  if (auto rc = maybe_run_worker_process(argc, argv))
    return *rc;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
