#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// out[t, h] = scale * table[tokens[t], h]
// table: [vocab, H], tokens: [T] on device, out: [T, H]
cudaError_t embedding_lookup_f32(const float* table, const int32_t* tokens, float* out, int T,
                                 int H, int vocab, float scale, int block_size = 256);
