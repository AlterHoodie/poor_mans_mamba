#include "ops/cpu/reductions.h"

#include "ops/cpu/map.h"

StatusOr<int32_t> argmax(Tensor& T) {
  Eigen::Map<Eigen::VectorXf> vec = as_vec_f32(T);
  Eigen::Index idx;
  vec.maxCoeff(&idx);
  return static_cast<int32_t>(idx);
}

StatusOr<int32_t> softmax(Tensor& T) {
  Eigen::Map<Eigen::VectorXf> vec = as_vec_f32(T);
  Eigen::Index idx;

  auto shifted = vec.array() - vec.maxCoeff();
  auto e = shifted.exp();
  auto probs = e / e.sum();

  probs.maxCoeff(&idx);
  return static_cast<int32_t>(idx);
}