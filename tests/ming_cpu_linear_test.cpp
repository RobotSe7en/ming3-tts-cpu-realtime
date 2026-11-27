#include "kernels.h"
#include "engine/framework/core/execution_context.h"
#include <ggml-alloc.h>
#include <ggml-cpu.h>
#include <omp.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() try {
    using namespace engine;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    core::BackendConfig config;
    config.type = core::BackendType::Cpu;
    config.threads = 4;
    core::ExecutionContext execution(config);
    const auto gelu_table = ming_cpu::make_gelu_bf16_table(execution);
    int cases = 0;
    for (int threads : {1, 2, 4}) for (int64_t width : {32, 256, 1024, 4096}) {
        omp_set_num_threads(threads);
        ggml_backend_cpu_set_n_threads(execution.backend(), threads);
        for (int64_t rows : {1, 2, 3, 4, 6, 7}) {
            constexpr int64_t channels = 64;
            auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
            core::ModuleBuildContext build{ctx, "linear.test", core::BackendType::Cpu};
            auto input = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({rows, width}));
            auto weight = core::make_tensor(build, GGML_TYPE_Q8_0, core::TensorShape::from_dims({channels, width}));
            auto bias = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({channels}));
            auto norm_weight = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({width}));
            for (auto * tensor : {input.tensor, weight.tensor, bias.tensor, norm_weight.tensor}) {
                ggml_set_input(tensor);
                ggml_set_output(tensor);
            }
            modules::LinearWeights weights{weight, bias};
            auto original = modules::LinearModule({width, channels, true}).build(build, input, weights);
            auto * expected = ggml_round_bf16(ctx, original.tensor);
            auto actual = ming_cpu::linear_bf16(build, input, weights);
            auto fallback = ming_cpu::linear_bf16(build, input, weights, false);
            modules::NormWeights norm_weights{norm_weight, std::nullopt};
            modules::NormConfig norm_config{width, 1.0e-6f, true, false};
            norm_config.normalized_round_type = GGML_TYPE_BF16;
            auto original_norm = modules::RMSNormModule(norm_config).build(build, input, norm_weights);
            auto * expected_norm = ggml_round_bf16(ctx, original_norm.tensor);
            auto actual_norm = ming_cpu::rms_norm_bf16(build, input, norm_weights);
            auto * expected_add = ggml_round_bf16(ctx, ggml_add(ctx, input.tensor, expected_norm));
            auto actual_add = ming_cpu::add_bf16(build, input, actual_norm);
            auto * expected_gelu = ggml_round_bf16(ctx, ggml_gelu(ctx, expected_norm));
            auto actual_gelu = ming_cpu::gelu_bf16(build, actual_norm, gelu_table);
            auto * graph = ggml_new_graph_custom(ctx, 256, false);
            for (auto * tensor : {expected, actual.tensor, fallback.tensor, expected_norm,
                    actual_norm.tensor, expected_add, actual_add.tensor, expected_gelu, actual_gelu.tensor}) {
                ggml_set_output(tensor);
                ggml_build_forward_expand(graph, tensor);
            }
            auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("Allocation failed");
            std::vector<float> w(channels * width), b(channels), values(rows * width);
            std::vector<std::byte> quantized(ggml_nbytes(weight.tensor));
            for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(i * 0.173f) * 0.1f;
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(w.data(), quantized.data(), w.size());
            ggml_backend_tensor_set(weight.tensor, quantized.data(), 0, quantized.size());
            for (size_t i = 0; i < b.size(); ++i) b[i] = std::cos(i * 0.31f) * 0.3f;
            core::write_tensor_f32(bias, b);
            std::vector<float> nw(width);
            for (size_t i = 0; i < nw.size(); ++i) nw[i] = 0.5f + std::cos(i * 0.71f) * 0.3f;
            core::write_tensor_f32(norm_weight, nw);
            for (int repeat = 0; repeat < 4; ++repeat) {
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = repeat == 0 ? 0.0f : std::sin(i * (0.113f + repeat * 0.037f)) * 3.25f;
                }
                core::write_tensor_f32(input, values);
                if (ggml_backend_graph_compute(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("Graph computation failed");
                }
                const std::array<std::pair<ggml_tensor *, ggml_tensor *>, 5> pairs{{
                    {expected, actual.tensor}, {expected, fallback.tensor},
                    {expected_norm, actual_norm.tensor}, {expected_add, actual_add.tensor},
                    {expected_gelu, actual_gelu.tensor}}};
                for (const auto & pair : pairs) {
                    const auto lhs = core::read_tensor_f32(pair.first);
                    const auto rhs = core::read_tensor_f32(pair.second);
                    if (std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) != 0) {
                        for (size_t i = 0; i < lhs.size(); ++i) if (lhs[i] != rhs[i]) {
                            std::cerr << "width=" << width << " rows=" << rows << " repeat=" << repeat
                                      << " threads=" << threads << " index=" << i << " expected=" << lhs[i]
                                      << " actual=" << rhs[i] << '\n';
                            break;
                        }
                        throw std::runtime_error("Fused BF16 output differs from original GGML");
                    }
                }
            }
            ggml_gallocr_free(allocator);
            ggml_free(ctx);
            ++cases;
        }
    }
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
    core::ModuleBuildContext build{ctx, "gelu.exhaustive", core::BackendType::Cpu};
    auto input = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({256, 256}));
    auto * values = static_cast<float *>(input.tensor->data);
    for (uint32_t i = 0; i < 65536; ++i) {
        const uint32_t bits = (i ^ 0x9876U) << 16;
        std::memcpy(values + i, &bits, sizeof(bits));
    }
    auto * expected = ggml_round_bf16(ctx, ggml_gelu(ctx, input.tensor));
    auto actual = ming_cpu::gelu_bf16(build, input, gelu_table);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, expected);
    ggml_build_forward_expand(graph, actual.tensor);
    if (ggml_backend_graph_compute(execution.backend(), graph) != GGML_STATUS_SUCCESS ||
        std::memcmp(expected->data, actual.tensor->data, 65536 * sizeof(float)) != 0) {
        throw std::runtime_error("Exhaustive BF16 GELU comparison failed");
    }
    ggml_free(ctx);
    std::cout << cases << " shapes/thread configurations passed, linear AVX2/fallback, norm, residual, GELU, "
              << "four executions each; all 65536 GELU encodings passed\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
