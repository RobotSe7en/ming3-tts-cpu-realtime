#include "packed_q8.h"
#include "engine/framework/core/execution_context.h"
#include <ggml-cpu.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() try {
    using namespace engine;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    core::BackendConfig backend;
    backend.type = core::BackendType::Cpu;
    backend.threads = 4;
    core::ExecutionContext execution(backend);
    int cases = 0;
    for (int threads : {1, 2, 4}) for (int64_t width : {32, 256, 1024, 4096}) {
        omp_set_num_threads(threads);
        ggml_backend_cpu_set_n_threads(execution.backend(), threads);
        for (int64_t rows : {1, 2, 3, 4, 6, 7}) {
            constexpr int64_t channels = 64;
            auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, false});
            core::ModuleBuildContext build{ctx, "packed.test", core::BackendType::Cpu};
            auto input = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({rows, width}));
            auto weight = core::make_tensor(build, GGML_TYPE_Q8_0, core::TensorShape::from_dims({channels, width}));
            auto bias = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({channels}));
            std::vector<float> w(channels * width);
            for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(i * 0.173f) * 0.1f;
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(w.data(), weight.tensor->data, w.size());
            auto * b = static_cast<float *>(bias.tensor->data);
            for (int64_t i = 0; i < channels; ++i) b[i] = std::cos(i * 0.31f) * 0.3f;
            modules::LinearWeights weights{weight, bias};
            auto expected = ming_cpu::linear_bf16(build, input, weights);
            ming_cpu::trial::PackedWeights cache;
            auto actual = cache.linear_bf16(build, input, weights);
            const auto bytes = cache.bytes();
            auto reused = cache.linear_bf16(build, input, weights);
            if (cache.bytes() != bytes) throw std::runtime_error("Same weights were packed twice");
            auto * graph = ggml_new_graph_custom(ctx, 128, false);
            for (auto * tensor : {expected.tensor, actual.tensor, reused.tensor}) ggml_build_forward_expand(graph, tensor);
            auto * values = static_cast<float *>(input.tensor->data);
            for (int repeat = 0; repeat < 4; ++repeat) {
                for (int64_t i = 0; i < rows * width; ++i) {
                    values[i] = repeat == 0 ? 0.0f : std::sin(i * (0.113f + repeat * 0.037f)) * 3.25f;
                }
                if (ggml_backend_graph_compute(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("Packed graph computation failed");
                }
                for (auto * tensor : {actual.tensor, reused.tensor}) {
                    if (std::memcmp(expected.tensor->data, tensor->data, ggml_nbytes(tensor)) != 0) {
                        std::cerr << "threads=" << threads << " width=" << width << " rows=" << rows
                                  << " repeat=" << repeat << '\n';
                        throw std::runtime_error("Packed Q8 BF16 result differs from the native CPU path");
                    }
                }
            }
            ggml_free(ctx);
            ++cases;
        }
    }
    std::cout << cases << " packed Q8 configurations passed bitwise comparison, repeat/input changes, cache reuse and fallback\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
