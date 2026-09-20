#pragma once

#include "ops/backend.h"
#include <cublas_v2.h>

class CUDABackend final : public OpsBackend {
    private:
        int block_size_ = 256;
        mutable cublasHandle_t handle_ = nullptr;
    public:
        CUDABackend() : OpsBackend(Device::GPU){};

        ~CUDABackend(){
            if(!handle_) return;
            cublasDestroy(handle_);
        }
        
        Status attention_f32(const Tensor& normed_hidden, const AttnWeights& w, const AttnConfig& attn,
            const ScaleConfig* scales, int max_seq_length, int hidden_size,
            LayerCacheView& cache, int64_t past_len, Tensor& out) const override;

        Status mamba2_mixer_f32(const Tensor& normed_hidden, const Mamba2MixerWeights& w,
                       const SsmConfig& ssm, const ScaleConfig* scales, float rms_norm_eps,
                       LayerCacheView& cache, Tensor& out) const override;
        
        Status mlp_f32(const Tensor& normed_hidden, const MlpWeights& w, const MlpConfig& mlp,
              const ScaleConfig* scales, Tensor& out) const override;
        
        Status rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) const override;
        
        Status add(const Tensor& a, const Tensor& b, Tensor& out) const override;
        
        Status scale(Tensor& x, float s) const override;

        Status linear(const Tensor& x, const Tensor& W, Tensor& out) const override;

        Status embedding_lookup(const Tensor& table, const int32_t* tokens, int64_t n_tokens,
                                float scale, Tensor& out) const override;

        Status take_last_row(const Tensor& hidden, Tensor& out) const override;
};