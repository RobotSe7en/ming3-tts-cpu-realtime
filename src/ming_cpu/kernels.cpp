#include "kernels.h"
#include "engine/framework/modules/activation_modules.h"

#include <ggml-cpu.h>
#ifdef MING_USE_GGML_GEMM
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/llamafile/sgemm.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define MING_AVX2 __attribute__((target("avx2,fma,f16c")))
#endif

namespace ming_cpu {
namespace {

struct Q8Block {
    ggml_fp16_t scale;
    int8_t values[32];
};
static_assert(sizeof(Q8Block) == 34);

float round_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        bits = (bits & 0xffff0000U) | 0x00400000U;
    } else {
        bits = (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void norm(ggml_tensor * dst, const ggml_tensor * input, const ggml_tensor * weight,
    int ith, int nth, void *) {
    const int64_t width = input->ne[0];
    const auto * x = static_cast<const float *>(input->data);
    const auto * w = static_cast<const float *>(weight->data);
    auto * y = static_cast<float *>(dst->data);
    for (int64_t row = ith; row < ggml_nrows(input); row += nth) {
        double sum = 0.0;
        for (int64_t column = 0; column < width; ++column) {
            const float value = x[row * width + column];
            sum += static_cast<double>(value * value);
        }
        const float mean = sum / width;
        const float scale = 1.0f / std::sqrt(mean + 1.0e-6f);
        for (int64_t column = 0; column < width; ++column) {
            y[row * width + column] = round_bf16(round_bf16(x[row * width + column] * scale) * w[column]);
        }
    }
}

void add(ggml_tensor * dst, const ggml_tensor * lhs, const ggml_tensor * rhs,
    int ith, int nth, void *) {
    const auto * x = static_cast<const float *>(lhs->data);
    const auto * y = static_cast<const float *>(rhs->data);
    auto * result = static_cast<float *>(dst->data);
    const int64_t count = ggml_nelements(dst);
    for (int64_t i = count * ith / nth; i < count * (ith + 1) / nth; ++i) {
        result[i] = round_bf16(x[i] + y[i]);
    }
}

void gelu_lookup(ggml_tensor * dst, const ggml_tensor * input, int ith, int nth, void * userdata) {
    const auto * table = static_cast<const float *>(userdata);
    const auto * x = static_cast<const float *>(input->data);
    auto * y = static_cast<float *>(dst->data);
    const int64_t count = ggml_nelements(dst);
    for (int64_t i = count * ith / nth; i < count * (ith + 1) / nth; ++i) {
        uint32_t bits;
        std::memcpy(&bits, x + i, sizeof(bits));
        y[i] = table[bits >> 16];
    }
}

void quantize_rows(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * input = dst->src[0];
    const auto convert = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for (int64_t row = ith; row < dst->ne[1]; row += nth) {
        convert(reinterpret_cast<const float *>(static_cast<const char *>(input->data) + row * input->nb[1]),
            static_cast<char *>(dst->data) + row * dst->nb[1], dst->ne[0]);
    }
}

#if defined(MING_AVX2)
MING_AVX2 float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_extractf128_ps(value, 1), _mm256_castps256_ps128(value));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    return _mm_cvtss_f32(_mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1)));
}

// Two output channels and three input rows fit without spilling the eight-lane
// accumulators on AVX2's sixteen vector registers. Keep GGML's block order,
// FMA, and horizontal reduction so BF16 rounding remains reproducible.
template<int Columns>
MING_AVX2 void tile(const Q8Block * weights, const Q8Block * input, const float * bias,
    float * output, int64_t blocks, int64_t channels, int64_t first_channel, int64_t first_row) {
    __m256 accumulators[Columns][2] = {};
    for (int64_t block = 0; block < blocks; ++block) {
        const auto & w0 = weights[first_channel * blocks + block];
        const auto & w1 = weights[(first_channel + 1) * blocks + block];
        const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0.values));
        const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w1.values));
        const __m256i a0 = _mm256_sign_epi8(x0, x0);
        const __m256i a1 = _mm256_sign_epi8(x1, x1);
        const float scale0 = _cvtsh_ss(w0.scale), scale1 = _cvtsh_ss(w1.scale);
        for (int column = 0; column < Columns; ++column) {
            const auto & y = input[(first_row + column) * blocks + block];
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y.values));
            const float scale = _cvtsh_ss(y.scale);
            const __m256i p0 = _mm256_maddubs_epi16(a0, _mm256_sign_epi8(v, x0));
            const __m256i p1 = _mm256_maddubs_epi16(a1, _mm256_sign_epi8(v, x1));
            const __m256 q0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), p0));
            const __m256 q1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), p1));
            accumulators[column][0] = _mm256_fmadd_ps(_mm256_set1_ps(scale0 * scale), q0, accumulators[column][0]);
            accumulators[column][1] = _mm256_fmadd_ps(_mm256_set1_ps(scale1 * scale), q1, accumulators[column][1]);
        }
    }
    for (int column = 0; column < Columns; ++column) {
        for (int channel = 0; channel < 2; ++channel) {
            output[(first_row + column) * channels + first_channel + channel] =
                round_bf16(sum8(accumulators[column][channel]) + bias[first_channel + channel]);
        }
    }
}
#endif

void matmul(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * weights = dst->src[0];
    const auto * input = dst->src[1];
    const auto * bias = static_cast<const float *>(dst->src[2]->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t channels = dst->ne[0], rows = dst->ne[1], width = weights->ne[0];
#ifdef MING_USE_GGML_GEMM
    if (userdata) {
        const int64_t begin = (channels / 2) * ith / nth * 2;
        const int64_t end = (channels / 2) * (ith + 1) / nth * 2;
        if (begin == end) return;
        // Give each worker independent output channels. The existing tiled
        // GEMM then runs without a shared barrier, and this same worker owns
        // the bias/round epilogue for every output it has just produced.
        ggml_compute_params params{};
        params.nth = 1;
        if (llamafile_sgemm(&params, end - begin, rows, width / 32,
                static_cast<const char *>(weights->data) + begin * weights->nb[1], width / 32,
                input->data, width / 32, output + begin, channels,
                GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32)) {
            for (int64_t row = 0; row < rows; ++row) {
                for (int64_t channel = begin; channel < end; ++channel) {
                    output[row * channels + channel] = round_bf16(output[row * channels + channel] + bias[channel]);
                }
            }
            return;
        }
    }
#endif
#if defined(MING_AVX2)
    if (userdata) {
        const auto * w = static_cast<const Q8Block *>(weights->data);
        const auto * x = static_cast<const Q8Block *>(input->data);
        const int64_t tiles = channels / 2;
        const int64_t begin = tiles * ith / nth, end = tiles * (ith + 1) / nth;
        for (int64_t work = begin; work < end; ++work) {
            int64_t row = 0;
            for (; row + 3 <= rows; row += 3) {
                tile<3>(w, x, bias, output, width / 32, channels, work * 2, row);
            }
            if (rows - row == 2) tile<2>(w, x, bias, output, width / 32, channels, work * 2, row);
            if (rows - row == 1) tile<1>(w, x, bias, output, width / 32, channels, work * 2, row);
        }
        return;
    }
#else
    (void) userdata;
#endif
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot;
    for (int64_t channel = channels * ith / nth; channel < channels * (ith + 1) / nth; ++channel) {
        for (int64_t row = 0; row < rows; ++row) {
            float value;
            dot(width, &value, 0, static_cast<const char *>(weights->data) + channel * weights->nb[1], 0,
                static_cast<const char *>(input->data) + row * input->nb[1], 0, 1);
            output[row * channels + channel] = round_bf16(value + bias[channel]);
        }
    }
}

}  // namespace

std::vector<float> make_gelu_bf16_table(engine::core::ExecutionContext & execution) {
    // Evaluate with this build's original CPU GELU, including its FMA behavior.
    // This is exhaustive over BF16 encodings, rather than a new approximation.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
    if (!ctx) throw std::runtime_error("GELU table allocation failed");
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 65536);
    auto * values = static_cast<float *>(input->data);
    for (uint32_t i = 0; i < 65536; ++i) {
        const uint32_t bits = i << 16;
        std::memcpy(values + i, &bits, sizeof(bits));
    }
    auto * output = ggml_round_bf16(ctx.get(), ggml_gelu(ctx.get(), input));
    auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GELU table computation failed");
    }
    const auto * result = static_cast<const float *>(output->data);
    return std::vector<float>(result, result + 65536);
}

engine::core::TensorValue gelu_bf16(engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input, const std::vector<float> & table) {
    using namespace engine;
    if (ctx.backend_type != core::BackendType::Cpu || input.type != GGML_TYPE_F32) {
        auto output = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, input);
        return core::wrap_tensor(ggml_round_bf16(ctx.ggml, output.tensor), output.shape, GGML_TYPE_F32);
    }
    if (table.size() != 65536) throw std::runtime_error("Invalid BF16 GELU table");
    auto * contiguous = ggml_is_contiguous(input.tensor) ? input.tensor : ggml_cont(ctx.ggml, input.tensor);
    return core::wrap_tensor(ggml_map_custom1(ctx.ggml, contiguous,
        gelu_lookup, GGML_N_TASKS_MAX, const_cast<float *>(table.data())), input.shape, GGML_TYPE_F32);
}

engine::core::TensorValue rms_norm_bf16(engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input, const engine::modules::NormWeights & weights) {
    using namespace engine;
    if (ctx.backend_type != core::BackendType::Cpu || input.type != GGML_TYPE_F32 || weights.bias ||
        !weights.weight || weights.weight->type != GGML_TYPE_F32) {
        modules::NormConfig config{input.shape.last_dim(), 1.0e-6f, true, false};
        config.normalized_round_type = GGML_TYPE_BF16;
        auto output = modules::RMSNormModule(config).build(ctx, input, weights);
        return core::wrap_tensor(ggml_round_bf16(ctx.ggml, output.tensor), output.shape, GGML_TYPE_F32);
    }
    auto * contiguous = ggml_is_contiguous(input.tensor) ? input.tensor : ggml_cont(ctx.ggml, input.tensor);
    return core::wrap_tensor(ggml_map_custom2(ctx.ggml, contiguous, weights.weight->tensor,
        norm, GGML_N_TASKS_MAX, nullptr), input.shape, GGML_TYPE_F32);
}

engine::core::TensorValue add_bf16(engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs, const engine::core::TensorValue & rhs) {
    using namespace engine;
    core::validate_shape(rhs, lhs.shape, "Ming residual");
    if (ctx.backend_type != core::BackendType::Cpu || lhs.type != GGML_TYPE_F32 || rhs.type != GGML_TYPE_F32) {
        return core::wrap_tensor(ggml_round_bf16(ctx.ggml, ggml_add(ctx.ggml, lhs.tensor, rhs.tensor)),
            lhs.shape, GGML_TYPE_F32);
    }
    auto * x = ggml_is_contiguous(lhs.tensor) ? lhs.tensor : ggml_cont(ctx.ggml, lhs.tensor);
    auto * y = ggml_is_contiguous(rhs.tensor) ? rhs.tensor : ggml_cont(ctx.ggml, rhs.tensor);
    return core::wrap_tensor(ggml_map_custom2(ctx.ggml, x, y, add, GGML_N_TASKS_MAX, nullptr),
        lhs.shape, GGML_TYPE_F32);
}

engine::core::TensorValue linear_bf16(engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::modules::LinearWeights & weights, bool allow_avx2) {
    using namespace engine;
    const int64_t width = weights.weight.tensor->ne[0], channels = weights.weight.tensor->ne[1];
    const int64_t rows = input.shape.num_elements() / width;
    if (ctx.backend_type != core::BackendType::Cpu || weights.weight.type != GGML_TYPE_Q8_0 ||
        input.type != GGML_TYPE_F32 || !weights.bias || channels % 2 || rows > 6 || width % 32 ||
        !ggml_is_contiguous(weights.weight.tensor)) {
        auto output = modules::LinearModule({width, channels, weights.bias.has_value()}).build(ctx, input, weights);
        return core::wrap_tensor(ggml_round_bf16(ctx.ggml, output.tensor), output.shape, GGML_TYPE_F32);
    }
    GGML_ASSERT(sizeof(Q8Block) == ggml_type_size(GGML_TYPE_Q8_0));
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto * matrix = ggml_reshape_2d(ctx.ggml, contiguous.tensor, width, rows);
    ggml_tensor * quant_args[] = {matrix};
    auto * quantized = ggml_custom_4d(ctx.ggml, GGML_TYPE_Q8_0, width, rows, 1, 1,
        quant_args, 1, quantize_rows, GGML_N_TASKS_MAX, nullptr);
    bool use_avx2 = false;
#if defined(MING_AVX2)
    // GGML's ggml_cpu_has_* helpers describe compile flags in this revision.
    // Query the actual CPU/OS capability before entering our target function.
    static const bool avx2_available = __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    use_avx2 = allow_avx2 && avx2_available;
#else
    (void) allow_avx2;
#endif
    ggml_tensor * args[] = {weights.weight.tensor, quantized, weights.bias->tensor};
    // The callback only uses this address as a dispatch tag; it is never read.
    static const bool avx2_tag = true;
    auto * output = ggml_custom_4d(ctx.ggml, GGML_TYPE_F32, channels, rows, 1, 1,
        args, 3, matmul, GGML_N_TASKS_MAX, use_avx2 ? const_cast<bool *>(&avx2_tag) : nullptr);
    return core::reshape_tensor(ctx, core::wrap_tensor(output,
        core::TensorShape::from_dims({rows, channels}), GGML_TYPE_F32), input.shape.with_last_dim(channels));
}

}  // namespace ming_cpu
