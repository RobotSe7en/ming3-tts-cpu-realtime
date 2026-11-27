#include "packed_q8.h"

#include <ggml-cpu.h>
#include <cstring>
#include <unordered_map>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define MING_PACK_AVX2 __attribute__((target("avx2,fma,f16c")))
#endif

namespace ming_cpu::trial {
namespace {

struct Q8 {
    ggml_fp16_t scale;
    int8_t q[32];
};
struct ExpandedQ8 {
    float scale;
    int8_t q[32];
};
static_assert(sizeof(Q8) == 34 && sizeof(ExpandedQ8) == 36);

struct Weight {
    int64_t width, channels;
    std::vector<int8_t> q;
    std::vector<float> scale;
};

float bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    bits = (bits & 0x7fffffffU) > 0x7f800000U
        ? ((bits & 0xffff0000U) | 0x00400000U)
        : ((bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U);
    std::memcpy(&value, &bits, 4);
    return value;
}

void expand_quantized(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * input = dst->src[0];
    const int64_t width = input->ne[0], blocks = width / 32;
    const auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for (int64_t row = ith; row < dst->ne[1]; row += nth) {
        auto * bytes = static_cast<char *>(dst->data) + row * dst->nb[1];
        const auto * x = reinterpret_cast<const float *>(static_cast<const char *>(input->data) + row * input->nb[1]);
        quantize(x, bytes, width);
        // Expand backwards within the same allocation, without per-call scratch
        // allocations. Quantization and FP16 scale rounding remain unchanged.
        for (int64_t b = blocks; b-- > 0;) {
            const auto * source = bytes + b * sizeof(Q8);
            auto * target = bytes + b * sizeof(ExpandedQ8);
            ggml_fp16_t scale;
            std::memcpy(&scale, source, sizeof(scale));
            const float value = ggml_fp16_to_fp32(scale);
            std::memmove(target + sizeof(float), source + sizeof(scale), 32);
            std::memcpy(target, &value, sizeof(value));
        }
    }
}

#ifdef MING_PACK_AVX2
MING_PACK_AVX2 float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_extractf128_ps(value, 1), _mm256_castps256_ps128(value));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    return _mm_cvtss_f32(_mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1)));
}

template<int Columns>
MING_PACK_AVX2 void tile(const Weight & weight, const ExpandedQ8 * input,
    const float * bias, float * output, int64_t pair, int64_t first_row) {
    const int64_t blocks = weight.width / 32;
    __m256 acc[Columns][2] = {};
    for (int64_t b = 0; b < blocks; ++b) {
        const int64_t offset = (pair * blocks + b) * 2;
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weight.q.data() + offset * 32));
        const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weight.q.data() + (offset + 1) * 32));
        const __m256i a0 = _mm256_sign_epi8(w0, w0), a1 = _mm256_sign_epi8(w1, w1);
        const float scale0 = weight.scale[offset], scale1 = weight.scale[offset + 1];
        for (int c = 0; c < Columns; ++c) {
            const auto & x = input[(first_row + c) * blocks + b];
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x.q));
            const __m256i p0 = _mm256_maddubs_epi16(a0, _mm256_sign_epi8(v, w0));
            const __m256i p1 = _mm256_maddubs_epi16(a1, _mm256_sign_epi8(v, w1));
            const __m256 q0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(p0, _mm256_set1_epi16(1)));
            const __m256 q1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(p1, _mm256_set1_epi16(1)));
            acc[c][0] = _mm256_fmadd_ps(_mm256_set1_ps(scale0 * x.scale), q0, acc[c][0]);
            acc[c][1] = _mm256_fmadd_ps(_mm256_set1_ps(scale1 * x.scale), q1, acc[c][1]);
        }
    }
    for (int c = 0; c < Columns; ++c) for (int r = 0; r < 2; ++r) {
        const int64_t channel = pair * 2 + r;
        output[(first_row + c) * weight.channels + channel] = bf16(sum8(acc[c][r]) + bias[channel]);
    }
}
#endif

void packed_matmul(ggml_tensor * dst, int ith, int nth, void * userdata) {
#ifdef MING_PACK_AVX2
    const auto & weight = *static_cast<const Weight *>(userdata);
    const auto * input = static_cast<const ExpandedQ8 *>(dst->src[0]->data);
    const auto * bias = static_cast<const float *>(dst->src[1]->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t pairs = weight.channels / 2;
    for (int64_t pair = pairs * ith / nth; pair < pairs * (ith + 1) / nth; ++pair) {
        int64_t row = 0;
        for (; row + 3 <= dst->ne[1]; row += 3) tile<3>(weight, input, bias, output, pair, row);
        if (dst->ne[1] - row == 2) tile<2>(weight, input, bias, output, pair, row);
        if (dst->ne[1] - row == 1) tile<1>(weight, input, bias, output, pair, row);
    }
#else
    (void) dst; (void) ith; (void) nth; (void) userdata;
    GGML_ABORT("Packed AVX2 kernel was selected on an unsupported build");
#endif
}

bool supported() {
#ifdef MING_PACK_AVX2
    static const bool result = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("f16c");
    return result;
#else
    return false;
#endif
}

}  // namespace

struct PackedWeights::Impl {
    std::unordered_map<const ggml_tensor *, std::unique_ptr<Weight>> weights;
};

PackedWeights::PackedWeights() : impl_(std::make_unique<Impl>()) {}
PackedWeights::~PackedWeights() = default;

size_t PackedWeights::bytes() const {
    size_t size = 0;
    for (const auto & entry : impl_->weights) size += entry.second->q.size() + entry.second->scale.size() * sizeof(float);
    return size;
}

engine::core::TensorValue PackedWeights::linear_bf16(engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input, const engine::modules::LinearWeights & weights) {
    using namespace engine;
    const auto * tensor = weights.weight.tensor;
    const int64_t width = tensor->ne[0], channels = tensor->ne[1], rows = input.shape.num_elements() / width;
    if (!supported() || ctx.backend_type != core::BackendType::Cpu || tensor->type != GGML_TYPE_Q8_0 ||
        input.type != GGML_TYPE_F32 || !weights.bias || channels % 2 || rows > 6 || width % 32 ||
        !ggml_is_contiguous(tensor)) return ming_cpu::linear_bf16(ctx, input, weights);
    auto & packed = impl_->weights[tensor];
    if (!packed) {
        packed = std::make_unique<Weight>();
        packed->width = width; packed->channels = channels;
        const int64_t blocks = width / 32;
        packed->q.resize(channels * width);
        packed->scale.resize(channels * blocks);
        const auto * source = static_cast<const Q8 *>(tensor->data);
        for (int64_t pair = 0; pair < channels / 2; ++pair) for (int64_t b = 0; b < blocks; ++b) {
            for (int r = 0; r < 2; ++r) {
                const auto & original = source[(pair * 2 + r) * blocks + b];
                const int64_t index = (pair * blocks + b) * 2 + r;
                std::memcpy(packed->q.data() + index * 32, original.q, 32);
                packed->scale[index] = ggml_fp16_to_fp32(original.scale);
            }
        }
    }
    auto * contiguous = ggml_is_contiguous(input.tensor) ? input.tensor : ggml_cont(ctx.ggml, input.tensor);
    auto * matrix = ggml_reshape_2d(ctx.ggml, contiguous, width, rows);
    ggml_tensor * args[] = {matrix};
    auto * expanded = ggml_custom_4d(ctx.ggml, GGML_TYPE_I8, width / 32 * sizeof(ExpandedQ8), rows, 1, 1,
        args, 1, expand_quantized, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * projected_args[] = {expanded, weights.bias->tensor};
    auto * projected = ggml_custom_4d(ctx.ggml, GGML_TYPE_F32, channels, rows, 1, 1,
        projected_args, 2, packed_matmul, GGML_N_TASKS_MAX, packed.get());
    return core::reshape_tensor(ctx, core::wrap_tensor(projected,
        core::TensorShape::from_dims({rows, channels}), GGML_TYPE_F32), input.shape.with_last_dim(channels));
}

}  // namespace ming_cpu::trial
