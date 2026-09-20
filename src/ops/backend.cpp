#include "ops/backend.h"
#include "ops/cpu_backend.h"

#include <cstdlib>

const OpsBackend& ops_for(Device d) {
  switch (d) {
  case Device::CPU:
    return cpu_ops();
#ifdef MAMBASERVE_WITH_CUDA
  case Device::GPU:
    return cuda_ops();
#endif
  case Device::NA:
  default:
    std::abort();
  }
}

const OpsBackend& cpu_ops() {
  static const CPUBackend k;
  return k;
}
