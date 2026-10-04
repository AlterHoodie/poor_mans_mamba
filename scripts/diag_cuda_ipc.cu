// Standalone diagnostic: can CUDA IPC (what UCX cuda_ipc uses) work between GPU 0 and GPU 1
// on this machine, and does it work inside a single process (as MambaServe's thread-per-worker
// design requires)?
//
//   nvcc -O1 -o /tmp/diag_cuda_ipc scripts/diag_cuda_ipc.cu && /tmp/diag_cuda_ipc
//
// Tests:
//   1. cudaDeviceCanAccessPeer(0,1) / (1,0)        -> hardware/VM P2P support
//   2. IPC handle open in the SAME process          -> if this fails, UCX cuda_ipc cannot connect
//                                                      two in-process workers
//   3. IPC handle open from a separate CHILD process -> if this works, process-per-worker
//                                                      would make cuda_ipc usable
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CK(x)                                                   \
  do {                                                          \
    cudaError_t e_ = (x);                                       \
    if (e_ != cudaSuccess) {                                    \
      std::printf("  %s -> %s\n", #x, cudaGetErrorString(e_));  \
      return false;                                             \
    }                                                           \
  } while (0)

static bool open_and_copy(cudaIpcMemHandle_t h, size_t bytes, const char* tag) {
  CK(cudaSetDevice(1));
  void* remote = nullptr;
  CK(cudaIpcOpenMemHandle(&remote, h, cudaIpcMemLazyEnablePeerAccess));
  void* local = nullptr;
  CK(cudaMalloc(&local, bytes));
  CK(cudaMemcpy(local, remote, bytes, cudaMemcpyDeviceToDevice));
  CK(cudaDeviceSynchronize());
  CK(cudaIpcCloseMemHandle(remote));
  CK(cudaFree(local));
  std::printf("  [%s] open + D2D copy from IPC mapping: OK\n", tag);
  return true;
}

static std::string to_hex(const void* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    unsigned char c = static_cast<const unsigned char*>(p)[i];
    s += d[c >> 4];
    s += d[c & 15];
  }
  return s;
}

static void from_hex(const char* s, void* out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    unsigned v;
    std::sscanf(s + 2 * i, "%2x", &v);
    static_cast<unsigned char*>(out)[i] = static_cast<unsigned char>(v);
  }
}

int main(int argc, char** argv) {
  const size_t bytes = 1 << 20;

  if (argc == 3 && std::strcmp(argv[1], "child") == 0) {
    cudaIpcMemHandle_t h;
    from_hex(argv[2], &h, sizeof(h));
    return open_and_copy(h, bytes, "child-process") ? 0 : 1;
  }

  int n = 0;
  cudaGetDeviceCount(&n);
  std::printf("devices: %d\n", n);
  if (n < 2) {
    std::printf("need 2 GPUs\n");
    return 1;
  }

  int a = 0, b = 0;
  cudaDeviceCanAccessPeer(&a, 0, 1);
  cudaDeviceCanAccessPeer(&b, 1, 0);
  std::printf("1) canAccessPeer 0->1: %d, 1->0: %d\n", a, b);
  if (!a || !b)
    std::printf("   => no peer access: UCX cuda_ipc will not be used on this VM/topology\n");

  cudaSetDevice(0);
  void* src = nullptr;
  if (cudaMalloc(&src, bytes) != cudaSuccess)
    return 1;
  cudaMemset(src, 0x5a, bytes);
  cudaIpcMemHandle_t h;
  if (cudaIpcGetMemHandle(&h, src) != cudaSuccess) {
    std::printf("cudaIpcGetMemHandle failed\n");
    return 1;
  }

  std::printf("2) same-process IPC open:\n");
  open_and_copy(h, bytes, "same-process");
  std::fflush(stdout);

  std::printf("3) separate-process IPC open (fork+exec):\n");
  std::fflush(stdout);
  pid_t pid = fork();
  if (pid == 0) {
    std::string hex = to_hex(&h, sizeof(h));
    execl("/proc/self/exe", "diag_cuda_ipc", "child", hex.c_str(), (char*)nullptr);
    _exit(127);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  std::printf("   child exit status: %d\n", WIFEXITED(st) ? WEXITSTATUS(st) : -1);
  return 0;
}
