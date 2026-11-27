#include "repack_weights.h"
#include "flow_engine.h"

#include "engine/framework/modules/weight_binding.h"
#include "ggml-cpu/repack.h"

#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ming_cpu {
namespace {

engine::assets::TensorStorageType to_storage_type(LayerWeightQuant quant) {
    switch (quant) {
        case LayerWeightQuant::Native: return engine::assets::TensorStorageType::Native;
        case LayerWeightQuant::F16: return engine::assets::TensorStorageType::F16;
        case LayerWeightQuant::Q4_0: return engine::assets::TensorStorageType::Q4_0;
        case LayerWeightQuant::IQ4_NL: return engine::assets::TensorStorageType::IQ4_NL;
    }
    return engine::assets::TensorStorageType::Native;
}

bool can_repack(engine::core::BackendWeightStore * repack_store, LayerWeightQuant quant,
    int64_t out_features, int64_t in_features) {
    return repack_store != nullptr &&
        (quant == LayerWeightQuant::Q4_0 || quant == LayerWeightQuant::IQ4_NL) &&
        cpu_supports_avx2_fma() && out_features > 0 && in_features > 0 &&
        (out_features % 8 == 0) && (in_features % 32 == 0);
}

std::vector<std::byte> convert_tensor_f16_parallel(const engine::assets::TensorSource & source,
    const std::string & weight_name, int64_t out_features, int64_t in_features) {
    const auto values = source.require_f32(weight_name, std::vector<int64_t>{out_features, in_features});
    std::vector<std::byte> bytes(static_cast<size_t>(out_features * in_features) * sizeof(ggml_fp16_t));
    auto * dst = reinterpret_cast<ggml_fp16_t *>(bytes.data());
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < out_features; ++r) {
        ggml_fp32_to_fp16_row(values.data() + r * in_features, dst + r * in_features, in_features);
    }
    return bytes;
}

std::vector<std::byte> quantize_tensor_parallel(const engine::assets::TensorSource & source,
    const std::string & weight_name, int64_t out_features, int64_t in_features, ggml_type type) {
    if (type == GGML_TYPE_F16) {
        return convert_tensor_f16_parallel(source, weight_name, out_features, in_features);
    }
    const auto values = source.require_f32(weight_name, std::vector<int64_t>{out_features, in_features});
    const size_t row_bytes = ggml_row_size(type, in_features);
    std::vector<std::byte> bytes(static_cast<size_t>(out_features) * row_bytes);
    ggml_quantize_init(type);
    const int64_t chunk_rows = 32;
    const int64_t chunks = (out_features + chunk_rows - 1) / chunk_rows;
    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < chunks; ++c) {
        const int64_t r0 = c * chunk_rows;
        const int64_t rn = std::min(chunk_rows, out_features - r0);
        ggml_quantize_chunk(type, values.data(), bytes.data(), r0 * in_features, rn, in_features, nullptr);
    }
    return bytes;
}

}  // namespace

LayerWeightQuant parse_layer_quant_env(const char * name, LayerWeightQuant fallback) {
    const char * value = std::getenv(name);
    if (!value || *value == '\0') return fallback;
    const std::string s(value);
    if (s == "q8" || s == "q8_0" || s == "native") return LayerWeightQuant::Native;
    if (s == "f16" || s == "fp16") return LayerWeightQuant::F16;
    if (s == "q4_0" || s == "q4") return LayerWeightQuant::Q4_0;
    if (s == "iq4_nl" || s == "iq4") return LayerWeightQuant::IQ4_NL;
    throw std::runtime_error(std::string("Invalid layer quant for ") + name + ": " + s);
}

bool fused_fp32_act_enabled(LayerWeightQuant quant) {
    const char * value = std::getenv("MING_FUSED_FP32_ACT");
    if (value && *value != '\0') {
        const std::string s(value);
        if (s == "1" || s == "true" || s == "on") return true;
        if (s == "0" || s == "false" || s == "off") return false;
        throw std::runtime_error("Invalid MING_FUSED_FP32_ACT");
    }
    return quant != LayerWeightQuant::Native;
}

ggml_backend_buffer_type_t cpu_repack_buffer_type_if_supported(engine::core::BackendType backend_type) {
    if (backend_type != engine::core::BackendType::Cpu || !cpu_supports_avx2_fma()) {
        return nullptr;
    }
    return ggml_backend_cpu_repack_buffer_type();
}

engine::core::TensorValue load_matrix_maybe_repack(
    engine::core::BackendWeightStore & regular_store,
    engine::core::BackendWeightStore * repack_store,
    const engine::assets::TensorSource & source,
    const std::string & weight_name,
    int64_t out_features,
    int64_t in_features,
    LayerWeightQuant quant) {
    if (quant == LayerWeightQuant::F16) {
        const auto bytes = convert_tensor_f16_parallel(source, weight_name, out_features, in_features);
        return regular_store.make_tensor(
            engine::core::TensorShape::from_dims({out_features, in_features}),
            GGML_TYPE_F16, bytes.data(), bytes.size());
    }
    if (!can_repack(repack_store, quant, out_features, in_features)) {
        return regular_store.load_tensor(source, weight_name,
            engine::assets::TensorStorageType::Native, {out_features, in_features});
    }
    const ggml_type type = engine::assets::ggml_type_for_tensor_storage(to_storage_type(quant));
    const auto bytes = quantize_tensor_parallel(source, weight_name, out_features, in_features, type);
    return repack_store->make_tensor(
        engine::core::TensorShape::from_dims({out_features, in_features}),
        type, bytes.data(), bytes.size());
}

engine::modules::LinearWeights load_linear_maybe_repack(
    engine::core::BackendWeightStore & regular_store,
    engine::core::BackendWeightStore * repack_store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool has_bias,
    LayerWeightQuant quant) {
    if (quant != LayerWeightQuant::F16 && !can_repack(repack_store, quant, out_features, in_features)) {
        return engine::modules::binding::linear_from_source(regular_store, source, prefix,
            engine::assets::TensorStorageType::Native, out_features, in_features, has_bias);
    }
    engine::modules::LinearWeights weights;
    weights.weight = load_matrix_maybe_repack(regular_store, repack_store, source,
        prefix + ".weight", out_features, in_features, quant);
    if (has_bias) {
        weights.bias = regular_store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

engine::modules::LinearWeights load_packed_linear_maybe_repack(
    engine::core::BackendWeightStore & regular_store,
    engine::core::BackendWeightStore * repack_store,
    const engine::assets::TensorSource & source,
    const engine::modules::PackedLinearWeightsConfig & config,
    LayerWeightQuant quant) {
    int64_t total_out = 0;
    for (const auto & s : config.sources) total_out += s.out_features;
    const bool use_f16 = (quant == LayerWeightQuant::F16);
    const bool use_repack = !use_f16 && can_repack(repack_store, quant, total_out, config.in_features);
    if (!use_f16 && !use_repack) {
        return engine::modules::PackedLinearWeightsBuilder(config).build(
            regular_store, source, engine::assets::TensorStorageType::Native);
    }
    const ggml_type expected_type = use_f16
        ? GGML_TYPE_F16
        : engine::assets::ggml_type_for_tensor_storage(to_storage_type(quant));
    const size_t row_bytes = ggml_row_size(expected_type, config.in_features);
    std::vector<std::byte> packed_weight;
    packed_weight.reserve(static_cast<size_t>(total_out) * row_bytes);
    std::vector<std::byte> packed_bias;
    if (config.use_bias) {
        packed_bias.reserve(static_cast<size_t>(total_out) * sizeof(float));
    }
    for (const auto & item : config.sources) {
        const auto bytes = quantize_tensor_parallel(source, item.weight,
            item.out_features, config.in_features, expected_type);
        packed_weight.insert(packed_weight.end(), bytes.begin(), bytes.end());
        if (config.use_bias) {
            const auto bias = source.require_tensor(*item.bias, engine::assets::TensorStorageType::F32,
                {item.out_features});
            if (bias.type != GGML_TYPE_F32 || bias.bytes.size() != static_cast<size_t>(item.out_features) * sizeof(float)) {
                throw std::runtime_error("Repacked linear bias must be F32: " + *item.bias);
            }
            packed_bias.insert(packed_bias.end(), bias.bytes.begin(), bias.bytes.end());
        }
    }
    engine::modules::LinearWeights weights;
    auto & target_store = use_f16 ? regular_store : *repack_store;
    weights.weight = target_store.make_tensor(
        engine::core::TensorShape::from_dims({total_out, config.in_features}),
        expected_type, packed_weight.data(), packed_weight.size());
    if (config.use_bias) {
        weights.bias = regular_store.make_tensor(
            engine::core::TensorShape::from_dims({total_out}),
            GGML_TYPE_F32, packed_bias.data(), packed_bias.size());
    }
    return weights;
}

}  // namespace ming_cpu
