#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/packed_linear_weights.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ming_cpu {

enum class LayerWeightQuant {
    Native,
    F16,
    Q4_0,
    IQ4_NL,
};

LayerWeightQuant parse_layer_quant_env(const char * name, LayerWeightQuant fallback);
bool fused_fp32_act_enabled(LayerWeightQuant quant);

// Returns ggml_backend_cpu_repack_buffer_type() when AVX2 is available on CPU, else nullptr.
ggml_backend_buffer_type_t cpu_repack_buffer_type_if_supported(engine::core::BackendType backend_type);

// Loads a single 2D linear weight (and optional 1D F32 bias).
// When quant != Native and repack_store != nullptr and dimensions satisfy AVX2 repack
// requirements (out % 8 == 0, in % 32 == 0), converts at load time into repack_store.
engine::modules::LinearWeights load_linear_maybe_repack(
    engine::core::BackendWeightStore & regular_store,
    engine::core::BackendWeightStore * repack_store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool has_bias,
    LayerWeightQuant quant);

// Loads a single 2D weight matrix without bias (e.g. o_proj.weight).
engine::core::TensorValue load_matrix_maybe_repack(
    engine::core::BackendWeightStore & regular_store,
    engine::core::BackendWeightStore * repack_store,
    const engine::assets::TensorSource & source,
    const std::string & weight_name,
    int64_t out_features,
    int64_t in_features,
    LayerWeightQuant quant);

// Loads and concatenates multiple linear projections (e.g. QKV or Gate+Up).
// Concatenates quantized bytes before calling make_tensor so CPU_REPACK sees one
// contiguous 2D tensor, while any 1D bias goes to regular_store.
engine::modules::LinearWeights load_packed_linear_maybe_repack(
    engine::core::BackendWeightStore & regular_store,
    engine::core::BackendWeightStore * repack_store,
    const engine::assets::TensorSource & source,
    const engine::modules::PackedLinearWeightsConfig & config,
    LayerWeightQuant quant);

}  // namespace ming_cpu
