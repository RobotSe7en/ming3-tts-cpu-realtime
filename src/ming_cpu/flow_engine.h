#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speech_decoders/ming_flow_runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ming_cpu {

enum class FlowWeightFormat {
    Q8_0,
    Q4_1_G64,
    Q4_1_G128,
};

struct FastEngineSettings {
    FlowWeightFormat flow_format = FlowWeightFormat::Q4_1_G64;
    FlowWeightFormat aggregator_format = FlowWeightFormat::Q4_1_G64;
    int flow_steps = 10;
    int cfg_interval = 1;
};

const FastEngineSettings & fast_settings();
std::vector<float> flow_time_grid(int steps);

// Low-level GEMM/quantization building blocks exposed for unit & parity tests.
struct PackedQ8Matrix {
    int64_t out_features = 0;
    int64_t in_features = 0;
    // Interleaved pairs of output channels: for pair p, block b (32 elems):
    // q stores [ch=2p, 32B][ch=2p+1, 32B], scale stores [ch=2p][ch=2p+1].
    std::vector<int8_t> q;
    std::vector<float> scale;
    std::vector<float> bias;
};

// Asymmetric 4-bit group quantization: w ~= min[ch, g] + scale[ch, g] * u4, u4 in [0, 15].
// Group size G is 64 (2x32) or 128 (4x32), enabling AVX2 vpmaddubsw + vpaddw without vpsignb.
// Each output channel is stored as one contiguous 32-byte aligned slab: [q4 (ng*G/2 B) | scale (ng*4 B) | min (ng*4 B)]
// so each channel streams through a single hardware prefetch stream.
struct PackedQ4GMatrix {
    int64_t out_features = 0;
    int64_t in_features = 0;
    int group_size = 64;  // 64 or 128
    size_t channel_stride = 0;
    size_t scale_offset = 0;
    size_t min_offset = 0;
    std::vector<uint8_t> data;
    std::vector<float> bias;
};

struct Q8InputBuffer {
    int64_t rows = 0;
    int64_t in_features = 0;
    std::vector<int8_t> q;      // rows * in_features
    std::vector<float> scale;   // rows * (in_features / 32)
};

struct Q8GroupInputBuffer {
    int64_t rows = 0;
    int64_t in_features = 0;
    int group_size = 64;
    // Group-major layout to eliminate 4KB L1d set aliasing across rows (e.g. in_features=4096):
    // q[(g * rows + r) * group_size], scale[g * rows + r], while sum[r * ng + g] is row-major for SIMD reduction.
    std::vector<int8_t> q;
    std::vector<float> scale;
    std::vector<float> sum;
};

float round_bf16_scalar(float value);
bool cpu_supports_avx2_fma();

PackedQ8Matrix pack_q8_from_f32(const float * weights, const float * bias,
    int64_t out_features, int64_t in_features);
PackedQ4GMatrix pack_q4g_from_f32(const float * weights, const float * bias,
    int64_t out_features, int64_t in_features, int group_size);

void quantize_rows_q8(const float * x, int64_t rows, int64_t in_features, Q8InputBuffer & out);
void quantize_rows_q8_group(const float * x, int64_t rows, int64_t in_features, int group_size,
    Q8GroupInputBuffer & out);

// Single-threaded slice [ch_begin, ch_end) kernels (for use inside a persistent OpenMP region).
// Epilogue modes:
//   0 = round_bf16(acc + bias)
//   1 = round_bf16(residual + round_bf16(acc + bias))
//   2 = gelu_table[bf16(acc + bias)]
void gemm_q8_slice(const PackedQ8Matrix & w, const Q8InputBuffer & x,
    float * y, const float * residual, const float * gelu_table,
    int epilogue, int64_t ch_begin, int64_t ch_end, bool allow_avx2 = true);

void gemm_q4g_slice(const PackedQ4GMatrix & w, const Q8GroupInputBuffer & x,
    float * y, const float * residual, const float * gelu_table,
    int epilogue, int64_t ch_begin, int64_t ch_end, bool allow_avx2 = true);

// Persistent C++ MingPatchNetwork (Flow or Aggregator) running with minimal barriers.
class DirectPatchNetwork {
public:
    DirectPatchNetwork(const engine::assets::TensorSource & source,
        const engine::modules::MingPatchNetworkConfig & config,
        const std::string & prefix, int64_t latent_dim, int64_t llm_dim,
        bool conditioned, FlowWeightFormat format, engine::core::ExecutionContext & execution);
    ~DirectPatchNetwork();
    DirectPatchNetwork(const DirectPatchNetwork &) = delete;
    DirectPatchNetwork & operator=(const DirectPatchNetwork &) = delete;

    // General predict_velocity / single-step entry (also used by aggregator for batch >= 1).
    std::vector<float> run(const std::vector<float> & latent, const std::vector<float> & condition,
        const std::vector<float> & history, float time, bool dual = true);

    // Full Flow sampling loop with patch-constant caching (history_emb, cond_emb, block0 history QKV,
    // and precomputed time_mlp embeddings).
    std::vector<float> sample(const std::vector<float> & conditioning,
        const std::vector<float> & history, const std::vector<float> & noise,
        const std::vector<float> & sde_noise, const std::vector<float> & time_grid,
        float cfg, float sigma, float temperature, int cfg_interval);

    size_t weight_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ming_cpu
