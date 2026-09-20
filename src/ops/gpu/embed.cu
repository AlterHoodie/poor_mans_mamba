#include "ops/gpu/embed.cuh"

namespace {

__global__ void embedding_lookup_kernel(const float* __restrict__ table,
                                        const int32_t* __restrict__ tokens, float* __restrict__ out,
                                        int T, int H, float scale) {
  const int t = blockIdx.x;
  if (t >= T)
    return;
  const int32_t id = tokens[t];
  const float* row = table + static_cast<size_t>(id) * static_cast<size_t>(H);
  float* dest = out + static_cast<size_t>(t) * static_cast<size_t>(H);
  for (int h = threadIdx.x; h < H; h += blockDim.x)
    dest[h] = row[h] * scale;
}

} // namespace

cudaError_t embedding_lookup_f32(const float* table, const int32_t* tokens, float* out, int T,
                                 int H, int /*vocab*/, float scale, int block_size) {
  if (T <= 0 || H <= 0)
    return cudaErrorInvalidValue;
  embedding_lookup_kernel<<<T, block_size>>>(table, tokens, out, T, H, scale);
  return cudaGetLastError();
}
