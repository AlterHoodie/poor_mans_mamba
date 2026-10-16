#pragma once

// Execution resources for a single in-flight forward (prefill/decode).
// GPU: stream + cublas handle (usually one pair per cache slot).
// CPU: both null; backends ignore them.
struct ExecContext {
  void* stream = nullptr; // cudaStream_t when Device::GPU
  void* cublas = nullptr; // cublasHandle_t when Device::GPU
};
