#pragma once

// NVTX ranges for Nsight Systems. Compiled out unless MAMBASERVE_WITH_NVTX is
// defined (CMake -DMAMBASERVE_WITH_NVTX=ON).

#ifdef MAMBASERVE_WITH_NVTX
#include <nvtx3/nvToolsExt.h>
namespace telemetry {
struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};
} // namespace telemetry
#define MS_NVTX_CAT2(a, b) a##b
#define MS_NVTX_CAT(a, b) MS_NVTX_CAT2(a, b)
#define MS_NVTX_RANGE(name) ::telemetry::NvtxRange MS_NVTX_CAT(_ms_nvtx_, __LINE__)(name)
#else
#define MS_NVTX_RANGE(name) ((void)0)
#endif
