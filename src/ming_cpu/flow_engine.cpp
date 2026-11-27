#include "flow_engine.h"
#include "kernels.h"

#include "engine/framework/core/backend.h"

#include <ggml.h>
#include <ggml-cpu.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define MING_FAST_AVX2 __attribute__((target("avx2,fma,f16c")))
#endif

namespace ming_cpu {
namespace {

FlowWeightFormat parse_format_env(const char * name, FlowWeightFormat fallback) {
    const char * value = std::getenv(name);
    if (!value || *value == '\0') return fallback;
    const std::string s(value);
    if (s == "q8" || s == "q8_0") return FlowWeightFormat::Q8_0;
    if (s == "q4g64" || s == "q4_1_g64") return FlowWeightFormat::Q4_1_G64;
    if (s == "q4g128" || s == "q4_1_g128") return FlowWeightFormat::Q4_1_G128;
    throw std::runtime_error(std::string("Invalid flow weight format for ") + name + ": " + s);
}

int parse_int_env(const char * name, int fallback, int lo, int hi) {
    const char * value = std::getenv(name);
    if (!value || *value == '\0') return fallback;
    char * end = nullptr;
    const long v = std::strtol(value, &end, 10);
    if (*end != '\0' || v < lo || v > hi) {
        throw std::runtime_error(std::string("Invalid integer for ") + name);
    }
    return static_cast<int>(v);
}

struct Q8BlockRaw {
    ggml_fp16_t d;
    int8_t qs[32];
};
static_assert(sizeof(Q8BlockRaw) == 34);

inline float apply_epilogue(float value, float bias, const float * residual,
    const float * gelu_table, int epilogue, int64_t idx) {
    const float rounded = round_bf16_scalar(value + bias);
    if (epilogue == 0) return rounded;
    if (epilogue == 1) return round_bf16_scalar(residual[idx] + rounded);
    uint32_t bits;
    std::memcpy(&bits, &rounded, sizeof(bits));
    return gelu_table[bits >> 16];
}

template <int Epilogue>
inline float apply_epilogue_t(float value, float bias, const float * residual,
    const float * gelu_table, int64_t idx) {
    const float rounded = round_bf16_scalar(value + bias);
    if constexpr (Epilogue == 0) {
        return rounded;
    } else if constexpr (Epilogue == 1) {
        return round_bf16_scalar(residual[idx] + rounded);
    } else {
        uint32_t bits;
        std::memcpy(&bits, &rounded, sizeof(bits));
        return gelu_table[bits >> 16];
    }
}

#if defined(MING_FAST_AVX2)
MING_FAST_AVX2 inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
}

MING_FAST_AVX2 inline __m256 round_bf16_avx2(__m256 v) {
    const __m256i bits = _mm256_castps_si256(v);
    const __m256i abs_bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
    const __m256i is_nan = _mm256_cmpgt_epi32(abs_bits, _mm256_set1_epi32(0x7f800000));
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    const __m256i rounded = _mm256_and_si256(
        _mm256_add_epi32(_mm256_add_epi32(bits, _mm256_set1_epi32(0x7fff)), lsb),
        _mm256_set1_epi32(static_cast<int>(0xffff0000U)));
    const __m256i qnan = _mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(static_cast<int>(0xffff0000U))),
        _mm256_set1_epi32(0x00400000));
    return _mm256_castsi256_ps(_mm256_blendv_epi8(rounded, qnan, is_nan));
}

MING_FAST_AVX2 void linear_f32_row_avx2(const float * weights, const float * bias,
    const float * input, float * output, int64_t out_features, int64_t in_features) {
    for (int64_t ch = 0; ch < out_features; ++ch) {
        const float * row = weights + ch * in_features;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int64_t k = 0;
        for (; k + 16 <= in_features; k += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k), _mm256_loadu_ps(input + k), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k + 8), _mm256_loadu_ps(input + k + 8), acc1);
        }
        float sum = hsum8(_mm256_add_ps(acc0, acc1));
        for (; k < in_features; ++k) sum += row[k] * input[k];
        output[ch] = round_bf16_scalar(sum + (bias ? bias[ch] : 0.0f));
    }
}

MING_FAST_AVX2 inline void quantize_block32_avx2(const float * p, float id, int8_t * dst) {
    const __m256 vid = _mm256_set1_ps(id);
    const __m256i vmin = _mm256_set1_epi32(-127);
    const __m256i vmax = _mm256_set1_epi32(127);
    __m256i i0 = _mm256_min_epi32(vmax, _mm256_max_epi32(vmin, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 0), vid))));
    __m256i i1 = _mm256_min_epi32(vmax, _mm256_max_epi32(vmin, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 8), vid))));
    __m256i i2 = _mm256_min_epi32(vmax, _mm256_max_epi32(vmin, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 16), vid))));
    __m256i i3 = _mm256_min_epi32(vmax, _mm256_max_epi32(vmin, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 24), vid))));
    __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
    __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
    __m256i packed = _mm256_permute4x64_epi64(_mm256_packs_epi16(p01, p23), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), packed);
}

MING_FAST_AVX2 void quantize_row_q8_avx2(const float * row, int64_t width, int8_t * dst_q, float * dst_scale) {
    const int64_t nb = width / 32;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (int64_t b = 0; b < nb; ++b) {
        const float * p = row + b * 32;
        __m256 a0 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(p + 0));
        __m256 a1 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(p + 8));
        __m256 a2 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(p + 16));
        __m256 a3 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(p + 24));
        __m256 am = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
        __m128 m128 = _mm_max_ps(_mm256_extractf128_ps(am, 1), _mm256_castps256_ps128(am));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        const float amax = _mm_cvtss_f32(_mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1)));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        dst_scale[b] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
        quantize_block32_avx2(p, id, dst_q + b * 32);
    }
}

MING_FAST_AVX2 void quantize_row_q8_group_major_avx2(const float * row, int64_t r, int64_t rows,
    int64_t width, int gs, int8_t * q_base, float * scale_base, float * sum_row) {
    const int64_t ng = width / gs;
    const int sub_blocks = gs / 32;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (int64_t g = 0; g < ng; ++g) {
        const float * gp = row + g * gs;
        __m256 vmax = _mm256_setzero_ps();
        __m256 vsum = _mm256_setzero_ps();
        for (int s = 0; s < sub_blocks; ++s) {
            const float * p = gp + s * 32;
            const __m256 x0 = _mm256_loadu_ps(p + 0);
            const __m256 x1 = _mm256_loadu_ps(p + 8);
            const __m256 x2 = _mm256_loadu_ps(p + 16);
            const __m256 x3 = _mm256_loadu_ps(p + 24);
            vsum = _mm256_add_ps(vsum, _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)));
            const __m256 a01 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, x0), _mm256_andnot_ps(sign_mask, x1));
            const __m256 a23 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, x2), _mm256_andnot_ps(sign_mask, x3));
            vmax = _mm256_max_ps(vmax, _mm256_max_ps(a01, a23));
        }
        __m128 m128 = _mm_max_ps(_mm256_extractf128_ps(vmax, 1), _mm256_castps256_ps128(vmax));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        const float amax = _mm_cvtss_f32(_mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1)));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        scale_base[g * rows + r] = d;
        sum_row[g] = hsum8(vsum);
        int8_t * dst = q_base + (g * rows + r) * gs;
        for (int s = 0; s < sub_blocks; ++s) {
            quantize_block32_avx2(gp + s * 32, id, dst + s * 32);
        }
    }
}

MING_FAST_AVX2 void rms_norm_bf16_row_avx2(const float * x, const float * w, float * y, int64_t width) {
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    int64_t k = 0;
    for (; k + 8 <= width; k += 8) {
        const __m256 v = _mm256_loadu_ps(x + k);
        const __m256 sq = _mm256_mul_ps(v, v);
        sum0 = _mm256_add_pd(sum0, _mm256_cvtps_pd(_mm256_castps256_ps128(sq)));
        sum1 = _mm256_add_pd(sum1, _mm256_cvtps_pd(_mm256_extractf128_ps(sq, 1)));
    }
    const __m256d s256 = _mm256_add_pd(sum0, sum1);
    const __m128d s128 = _mm_add_pd(_mm256_extractf128_pd(s256, 1), _mm256_castpd256_pd128(s256));
    double sum = _mm_cvtsd_f64(_mm_add_sd(s128, _mm_unpackhi_pd(s128, s128)));
    for (; k < width; ++k) sum += static_cast<double>(x[k] * x[k]);
    const float scale = 1.0f / std::sqrt(static_cast<float>(sum / width) + 1.0e-6f);
    const __m256 vscale = _mm256_set1_ps(scale);
    k = 0;
    for (; k + 8 <= width; k += 8) {
        const __m256 xv = _mm256_loadu_ps(x + k);
        const __m256 wv = _mm256_loadu_ps(w + k);
        const __m256 n = round_bf16_avx2(_mm256_mul_ps(xv, vscale));
        _mm256_storeu_ps(y + k, round_bf16_avx2(_mm256_mul_ps(n, wv)));
    }
    for (; k < width; ++k) {
        y[k] = round_bf16_scalar(round_bf16_scalar(x[k] * scale) * w[k]);
    }
}

MING_FAST_AVX2 float dot64_avx2(const float * a, const float * b) {
    __m256 s0 = _mm256_mul_ps(_mm256_loadu_ps(a + 0), _mm256_loadu_ps(b + 0));
    __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), s1);
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), s1);
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), s1);
    return hsum8(_mm256_add_ps(s0, s1));
}

MING_FAST_AVX2 void weighted_sum_bf16_64_avx2(int64_t steps, const float * scores,
    const float * const * v_ptr, float * dst) {
    for (int64_t d = 0; d < 64; d += 8) {
        __m256 acc = _mm256_mul_ps(_mm256_set1_ps(scores[0]), _mm256_loadu_ps(v_ptr[0] + d));
        for (int64_t sk = 1; sk < steps; ++sk) {
            acc = _mm256_fmadd_ps(_mm256_set1_ps(scores[sk]), _mm256_loadu_ps(v_ptr[sk] + d), acc);
        }
        _mm256_storeu_ps(dst + d, round_bf16_avx2(acc));
    }
}

template <int M>
MING_FAST_AVX2 void gemm_q8_avx2_2xM(const PackedQ8Matrix & w, const Q8InputBuffer & x,
    float * y, const float * residual, const float * gelu_table, int epilogue,
    int64_t ch_begin, int64_t ch_end) {
    const int64_t nb = w.in_features / 32;
    const int64_t out_ch = w.out_features;
    const __m256i ones = _mm256_set1_epi16(1);
    const int64_t pair_begin = ch_begin / 2;
    const int64_t pair_end = ch_end / 2;
    for (int64_t p = pair_begin; p < pair_end; ++p) {
        __m256 acc[M][2];
        for (int m = 0; m < M; ++m) {
            acc[m][0] = _mm256_setzero_ps();
            acc[m][1] = _mm256_setzero_ps();
        }
        const int8_t * wp = w.q.data() + p * nb * 64;
        const float * sp = w.scale.data() + p * nb * 2;
        for (int64_t b = 0; b < nb; ++b) {
            const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wp + b * 64));
            const __m256i a0 = _mm256_sign_epi8(w0, w0);
            const float sw0 = sp[b * 2];
            for (int m = 0; m < M; ++m) {
                const __m256i xv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(x.q.data() + (m * nb + b) * 32));
                const __m256i prod = _mm256_maddubs_epi16(a0, _mm256_sign_epi8(xv, w0));
                const __m256 q = _mm256_cvtepi32_ps(_mm256_madd_epi16(prod, ones));
                acc[m][0] = _mm256_fmadd_ps(_mm256_set1_ps(sw0 * x.scale[m * nb + b]), q, acc[m][0]);
            }
            const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wp + b * 64 + 32));
            const __m256i a1 = _mm256_sign_epi8(w1, w1);
            const float sw1 = sp[b * 2 + 1];
            for (int m = 0; m < M; ++m) {
                const __m256i xv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(x.q.data() + (m * nb + b) * 32));
                const __m256i prod = _mm256_maddubs_epi16(a1, _mm256_sign_epi8(xv, w1));
                const __m256 q = _mm256_cvtepi32_ps(_mm256_madd_epi16(prod, ones));
                acc[m][1] = _mm256_fmadd_ps(_mm256_set1_ps(sw1 * x.scale[m * nb + b]), q, acc[m][1]);
            }
        }
        const int64_t c0 = p * 2;
        const int64_t c1 = c0 + 1;
        for (int m = 0; m < M; ++m) {
            const int64_t idx0 = m * out_ch + c0;
            const int64_t idx1 = m * out_ch + c1;
            y[idx0] = apply_epilogue(hsum8(acc[m][0]), w.bias[c0], residual, gelu_table, epilogue, idx0);
            y[idx1] = apply_epilogue(hsum8(acc[m][1]), w.bias[c1], residual, gelu_table, epilogue, idx1);
        }
    }
}

template <int M, int SubBlocks, int Epilogue>
MING_FAST_AVX2 void gemm_q4g_avx2_1xM(const PackedQ4GMatrix & w, const Q8GroupInputBuffer & x,
    float * y, const float * residual, const float * gelu_table,
    int64_t ch_begin, int64_t ch_end, int64_t y_stride, int64_t y_ch_offset) {
    constexpr int G = SubBlocks * 32;
    const int64_t ng = w.in_features / G;
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    for (int64_t ch = ch_begin; ch < ch_end; ++ch) {
        const uint8_t * slab = w.data.data() + static_cast<size_t>(ch) * w.channel_stride;
        const uint8_t * wp = slab;
        const float * sp = reinterpret_cast<const float *>(slab + w.scale_offset);
        const float * mp = reinterpret_cast<const float *>(slab + w.min_offset);
        __m256 acc[M];
        for (int m = 0; m < M; ++m) acc[m] = _mm256_setzero_ps();
        for (int64_t g = 0; g < ng; ++g) {
            __m256i wv[SubBlocks];
            const uint8_t * gp = wp + g * (G / 2);
            for (int s = 0; s < SubBlocks; ++s) {
                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(gp + s * 16));
                wv[s] = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), low_mask);
            }
            const float sw = sp[g];
            const int8_t * x_group = x.q.data() + (g * M) * G;
            const float * x_scales = x.scale.data() + g * M;
            for (int m = 0; m < M; ++m) {
                const int8_t * xp = x_group + m * G;
                __m256i sum16;
                if constexpr (SubBlocks == 2) {
                    const __m256i s0 = _mm256_maddubs_epi16(
                        wv[0], _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xp + 0)));
                    const __m256i s1 = _mm256_maddubs_epi16(
                        wv[1], _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xp + 32)));
                    sum16 = _mm256_add_epi16(s0, s1);
                } else {
                    const __m256i s0 = _mm256_maddubs_epi16(
                        wv[0], _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xp + 0)));
                    const __m256i s1 = _mm256_maddubs_epi16(
                        wv[1], _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xp + 32)));
                    const __m256i s2 = _mm256_maddubs_epi16(
                        wv[2], _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xp + 64)));
                    const __m256i s3 = _mm256_maddubs_epi16(
                        wv[3], _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xp + 96)));
                    sum16 = _mm256_add_epi16(_mm256_add_epi16(s0, s1), _mm256_add_epi16(s2, s3));
                }
                const __m256 q = _mm256_cvtepi32_ps(_mm256_madd_epi16(sum16, ones));
                acc[m] = _mm256_fmadd_ps(_mm256_set1_ps(sw * x_scales[m]), q, acc[m]);
            }
        }
        const float bias_ch = w.bias[ch];
        const int64_t out_col = ch - y_ch_offset;
        for (int m = 0; m < M; ++m) {
            const float * xsum = x.sum.data() + m * ng;
            __m256 cvec = _mm256_setzero_ps();
            int64_t g = 0;
            for (; g + 8 <= ng; g += 8) {
                cvec = _mm256_fmadd_ps(_mm256_loadu_ps(mp + g), _mm256_loadu_ps(xsum + g), cvec);
            }
            float total = hsum8(_mm256_add_ps(acc[m], cvec));
            for (; g < ng; ++g) {
                total += mp[g] * xsum[g];
            }
            const int64_t idx = m * y_stride + out_col;
            y[idx] = apply_epilogue_t<Epilogue>(total, bias_ch, residual, gelu_table, idx);
        }
    }
}

template <int M, int SubBlocks, int Epilogue>
MING_FAST_AVX2 void gemm_q4g_avx2_4xM(const PackedQ4GMatrix & w, const Q8GroupInputBuffer & x,
    float * y, const float * residual, const float * gelu_table,
    int64_t ch_begin, int64_t ch_end, int64_t y_stride, int64_t y_ch_offset) {
    static_assert(M == 1 || M == 2);
    constexpr int G = SubBlocks * 32;
    const int64_t ng = w.in_features / G;
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    for (int64_t ch = ch_begin; ch < ch_end; ch += 4) {
        const uint8_t * slab[4];
        const float * sp[4];
        const float * mp[4];
        __m256 acc[4][M];
        for (int c = 0; c < 4; ++c) {
            slab[c] = w.data.data() + static_cast<size_t>(ch + c) * w.channel_stride;
            sp[c] = reinterpret_cast<const float *>(slab[c] + w.scale_offset);
            mp[c] = reinterpret_cast<const float *>(slab[c] + w.min_offset);
            for (int m = 0; m < M; ++m) acc[c][m] = _mm256_setzero_ps();
        }
        for (int64_t g = 0; g < ng; ++g) {
            const int8_t * x_group = x.q.data() + (g * M) * G;
            const float * x_scales = x.scale.data() + g * M;
            const int64_t g_off = g * (G / 2);
            for (int c = 0; c < 4; ++c) {
                const uint8_t * gp = slab[c] + g_off;
                const float sw = sp[c][g];
                __m256i sum0 = _mm256_setzero_si256();
                __m256i sum1 = _mm256_setzero_si256();
                for (int s = 0; s < SubBlocks; ++s) {
                    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(gp + s * 16));
                    const __m256i wv = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), low_mask);
                    const __m256i p0 = _mm256_maddubs_epi16(
                        wv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x_group + s * 32)));
                    sum0 = (s == 0) ? p0 : _mm256_add_epi16(sum0, p0);
                    if constexpr (M == 2) {
                        const __m256i p1 = _mm256_maddubs_epi16(
                            wv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x_group + G + s * 32)));
                        sum1 = (s == 0) ? p1 : _mm256_add_epi16(sum1, p1);
                    }
                }
                const __m256 q0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(sum0, ones));
                acc[c][0] = _mm256_fmadd_ps(_mm256_set1_ps(sw * x_scales[0]), q0, acc[c][0]);
                if constexpr (M == 2) {
                    const __m256 q1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(sum1, ones));
                    acc[c][1] = _mm256_fmadd_ps(_mm256_set1_ps(sw * x_scales[1]), q1, acc[c][1]);
                }
            }
        }
        for (int c = 0; c < 4; ++c) {
            const float bias_ch = w.bias[ch + c];
            const int64_t out_col = (ch + c) - y_ch_offset;
            for (int m = 0; m < M; ++m) {
                const float * xsum = x.sum.data() + m * ng;
                __m256 cvec = _mm256_setzero_ps();
                int64_t g = 0;
                for (; g + 8 <= ng; g += 8) {
                    cvec = _mm256_fmadd_ps(_mm256_loadu_ps(mp[c] + g), _mm256_loadu_ps(xsum + g), cvec);
                }
                float total = hsum8(_mm256_add_ps(acc[c][m], cvec));
                for (; g < ng; ++g) {
                    total += mp[c][g] * xsum[g];
                }
                const int64_t idx = m * y_stride + out_col;
                y[idx] = apply_epilogue_t<Epilogue>(total, bias_ch, residual, gelu_table, idx);
            }
        }
    }
}
#endif

void linear_f32_row(const float * weights, const float * bias, const float * input,
    float * output, int64_t out_features, int64_t in_features) {
#if defined(MING_FAST_AVX2)
    if (cpu_supports_avx2_fma()) {
        linear_f32_row_avx2(weights, bias, input, output, out_features, in_features);
        return;
    }
#endif
    for (int64_t ch = 0; ch < out_features; ++ch) {
        const float * row = weights + ch * in_features;
        float sum = 0.0f;
        for (int64_t k = 0; k < in_features; ++k) {
            sum += row[k] * input[k];
        }
        output[ch] = round_bf16_scalar(sum + (bias ? bias[ch] : 0.0f));
    }
}

void rms_norm_bf16_row(const float * x, const float * w, float * y, int64_t width) {
#if defined(MING_FAST_AVX2)
    if (cpu_supports_avx2_fma()) {
        rms_norm_bf16_row_avx2(x, w, y, width);
        return;
    }
#endif
    double sum = 0.0;
    for (int64_t k = 0; k < width; ++k) {
        const float v = x[k];
        sum += static_cast<double>(v * v);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(sum / width) + 1.0e-6f);
    for (int64_t k = 0; k < width; ++k) {
        y[k] = round_bf16_scalar(round_bf16_scalar(x[k] * scale) * w[k]);
    }
}

void quantize_row_q8_scalar(const float * row, int64_t width, int8_t * dst_q, float * dst_scale) {
    const int64_t nb = width / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float * p = row + b * 32;
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(p[k]));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        dst_scale[b] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
        int8_t * dst = dst_q + b * 32;
        for (int k = 0; k < 32; ++k) {
            const int v = static_cast<int>(std::lrintf(p[k] * id));
            dst[k] = static_cast<int8_t>(std::clamp(v, -127, 127));
        }
    }
}

void quantize_row_q8_group_major_scalar(const float * row, int64_t r, int64_t rows,
    int64_t width, int gs, int8_t * q_base, float * scale_base, float * sum_row) {
    const int64_t ng = width / gs;
    for (int64_t g = 0; g < ng; ++g) {
        const float * p = row + g * gs;
        float amax = 0.0f;
        float sum = 0.0f;
        for (int k = 0; k < gs; ++k) {
            const float v = p[k];
            amax = std::max(amax, std::fabs(v));
            sum += v;
        }
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        scale_base[g * rows + r] = d;
        sum_row[g] = sum;
        int8_t * dst = q_base + (g * rows + r) * gs;
        for (int k = 0; k < gs; ++k) {
            const int v = static_cast<int>(std::lrintf(p[k] * id));
            dst[k] = static_cast<int8_t>(std::clamp(v, -127, 127));
        }
    }
}

void gemm_q8_scalar(const PackedQ8Matrix & w, const Q8InputBuffer & x,
    float * y, const float * residual, const float * gelu_table, int epilogue,
    int64_t ch_begin, int64_t ch_end) {
    const int64_t nb = w.in_features / 32;
    const int64_t out_ch = w.out_features;
    for (int64_t ch = ch_begin; ch < ch_end; ++ch) {
        const int64_t pair = ch / 2;
        const int64_t r = ch & 1;
        for (int64_t m = 0; m < x.rows; ++m) {
            float acc = 0.0f;
            for (int64_t b = 0; b < nb; ++b) {
                const int8_t * wp = w.q.data() + (pair * nb + b) * 64 + r * 32;
                const int8_t * xp = x.q.data() + (m * nb + b) * 32;
                int32_t dot = 0;
                for (int k = 0; k < 32; ++k) dot += static_cast<int32_t>(wp[k]) * xp[k];
                acc += w.scale[(pair * nb + b) * 2 + r] * x.scale[m * nb + b] * static_cast<float>(dot);
            }
            const int64_t idx = m * out_ch + ch;
            y[idx] = apply_epilogue(acc, w.bias[ch], residual, gelu_table, epilogue, idx);
        }
    }
}

void gemm_q4g_scalar(const PackedQ4GMatrix & w, const Q8GroupInputBuffer & x,
    float * y, const float * residual, const float * gelu_table, int epilogue,
    int64_t ch_begin, int64_t ch_end, int64_t y_stride, int64_t y_ch_offset) {
    const int G = w.group_size;
    const int sub_blocks = G / 32;
    const int64_t ng = w.in_features / G;
    const int64_t rows = x.rows;
    for (int64_t ch = ch_begin; ch < ch_end; ++ch) {
        const uint8_t * slab = w.data.data() + static_cast<size_t>(ch) * w.channel_stride;
        const uint8_t * wp = slab;
        const float * sp = reinterpret_cast<const float *>(slab + w.scale_offset);
        const float * mp = reinterpret_cast<const float *>(slab + w.min_offset);
        const int64_t out_col = ch - y_ch_offset;
        for (int64_t m = 0; m < rows; ++m) {
            float acc = 0.0f;
            for (int64_t g = 0; g < ng; ++g) {
                const uint8_t * gp = wp + g * (G / 2);
                const int8_t * xp = x.q.data() + (g * rows + m) * G;
                int32_t dot = 0;
                for (int s = 0; s < sub_blocks; ++s) {
                    const uint8_t * bp = gp + s * 16;
                    const int8_t * xs = xp + s * 32;
                    for (int j = 0; j < 16; ++j) {
                        const int32_t lo = bp[j] & 0x0F;
                        const int32_t hi = (bp[j] >> 4) & 0x0F;
                        dot += lo * xs[j] + hi * xs[j + 16];
                    }
                }
                acc += sp[g] * x.scale[g * rows + m] * static_cast<float>(dot) +
                    mp[g] * x.sum[m * ng + g];
            }
            const int64_t idx = m * y_stride + out_col;
            y[idx] = apply_epilogue(acc, w.bias[ch], residual, gelu_table, epilogue, idx);
        }
    }
}

struct FastLinear {
    FlowWeightFormat format = FlowWeightFormat::Q8_0;
    PackedQ8Matrix q8;
    PackedQ4GMatrix q4g;
    int64_t out_features = 0;
    int64_t in_features = 0;

    size_t bytes() const {
        if (format == FlowWeightFormat::Q8_0) {
            return q8.q.size() + q8.scale.size() * sizeof(float) + q8.bias.size() * sizeof(float);
        }
        return q4g.data.size() + q4g.bias.size() * sizeof(float);
    }
};

FastLinear load_fast_linear(const engine::assets::TensorSource & source,
    const std::string & weight_name, const std::string & bias_name,
    int64_t out_features, int64_t in_features, FlowWeightFormat format) {
    FastLinear linear;
    linear.format = format;
    linear.out_features = out_features;
    linear.in_features = in_features;
    const auto bias = source.require_f32(bias_name, std::vector<int64_t>{out_features});
    if (format == FlowWeightFormat::Q8_0) {
        const auto native = source.require_tensor(weight_name,
            engine::assets::TensorStorageType::Native, {out_features, in_features});
        if (native.type == GGML_TYPE_Q8_0 && out_features % 2 == 0 && in_features % 32 == 0) {
            const int64_t nb = in_features / 32;
            linear.q8.out_features = out_features;
            linear.q8.in_features = in_features;
            linear.q8.q.resize(static_cast<size_t>(out_features * in_features));
            linear.q8.scale.resize(static_cast<size_t>(out_features * nb));
            linear.q8.bias = bias;
            const auto * blocks = reinterpret_cast<const Q8BlockRaw *>(native.bytes.data());
            for (int64_t p = 0; p < out_features / 2; ++p) {
                for (int64_t b = 0; b < nb; ++b) {
                    for (int r = 0; r < 2; ++r) {
                        const auto & src = blocks[(p * 2 + r) * nb + b];
                        const int64_t dst_idx = (p * nb + b) * 2 + r;
                        std::memcpy(linear.q8.q.data() + dst_idx * 32, src.qs, 32);
                        linear.q8.scale[dst_idx] = ggml_fp16_to_fp32(src.d);
                    }
                }
            }
            return linear;
        }
        const auto weights = source.require_f32(weight_name, std::vector<int64_t>{out_features, in_features});
        linear.q8 = pack_q8_from_f32(weights.data(), bias.data(), out_features, in_features);
        return linear;
    }
    const auto weights = source.require_f32(weight_name, std::vector<int64_t>{out_features, in_features});
    const int group_size = (format == FlowWeightFormat::Q4_1_G64) ? 64 : 128;
    linear.q4g = pack_q4g_from_f32(weights.data(), bias.data(), out_features, in_features, group_size);
    return linear;
}

FastLinear load_fast_qkv(const engine::assets::TensorSource & source,
    const std::string & prefix, int64_t hidden, FlowWeightFormat format) {
    const std::array<std::string, 3> names = {
        prefix + ".attn.to_q", prefix + ".attn.to_k", prefix + ".attn.to_v"
    };
    std::vector<float> bias;
    bias.reserve(static_cast<size_t>(3 * hidden));
    for (const auto & n : names) {
        const auto b = source.require_f32(n + ".bias", std::vector<int64_t>{hidden});
        bias.insert(bias.end(), b.begin(), b.end());
    }
    FastLinear linear;
    linear.format = format;
    linear.out_features = 3 * hidden;
    linear.in_features = hidden;
    if (format == FlowWeightFormat::Q8_0) {
        bool all_q8 = true;
        std::vector<engine::assets::TensorData> parts;
        parts.reserve(3);
        for (const auto & n : names) {
            parts.push_back(source.require_tensor(n + ".weight",
                engine::assets::TensorStorageType::Native, {hidden, hidden}));
            if (parts.back().type != GGML_TYPE_Q8_0) all_q8 = false;
        }
        if (all_q8 && hidden % 2 == 0 && hidden % 32 == 0) {
            const int64_t nb = hidden / 32;
            linear.q8.out_features = 3 * hidden;
            linear.q8.in_features = hidden;
            linear.q8.q.resize(static_cast<size_t>(3 * hidden * hidden));
            linear.q8.scale.resize(static_cast<size_t>(3 * hidden * nb));
            linear.q8.bias = std::move(bias);
            for (int part = 0; part < 3; ++part) {
                const auto * blocks = reinterpret_cast<const Q8BlockRaw *>(parts[part].bytes.data());
                const int64_t pair_offset = (part * hidden) / 2;
                for (int64_t p = 0; p < hidden / 2; ++p) {
                    for (int64_t b = 0; b < nb; ++b) {
                        for (int r = 0; r < 2; ++r) {
                            const auto & src = blocks[(p * 2 + r) * nb + b];
                            const int64_t dst_idx = ((pair_offset + p) * nb + b) * 2 + r;
                            std::memcpy(linear.q8.q.data() + dst_idx * 32, src.qs, 32);
                            linear.q8.scale[dst_idx] = ggml_fp16_to_fp32(src.d);
                        }
                    }
                }
            }
            return linear;
        }
    }
    std::vector<float> weights;
    weights.reserve(static_cast<size_t>(3 * hidden * hidden));
    for (const auto & n : names) {
        const auto w = source.require_f32(n + ".weight", std::vector<int64_t>{hidden, hidden});
        weights.insert(weights.end(), w.begin(), w.end());
    }
    if (format == FlowWeightFormat::Q8_0) {
        linear.q8 = pack_q8_from_f32(weights.data(), bias.data(), 3 * hidden, hidden);
    } else {
        const int group_size = (format == FlowWeightFormat::Q4_1_G64) ? 64 : 128;
        linear.q4g = pack_q4g_from_f32(weights.data(), bias.data(), 3 * hidden, hidden, group_size);
    }
    return linear;
}

std::vector<float> build_rope_table_exact(const std::vector<float> & inv_freq_raw,
    int64_t steps, int64_t head_dim) {
    std::vector<float> freq = inv_freq_raw;
    engine::core::round_f32_to_bf16_in_place(freq);
    std::vector<float> table(static_cast<size_t>(steps * head_dim));
    const int64_t half = head_dim / 2;
    for (int64_t pos = 0; pos < steps; ++pos) {
        for (int64_t j = 0; j < half; ++j) {
            const float angle = static_cast<float>(pos) * freq[j];
            table[pos * head_dim + j] = round_bf16_scalar(std::cos(angle));
            table[pos * head_dim + half + j] = round_bf16_scalar(std::sin(angle));
        }
    }
    return table;
}

inline void apply_rope_vec(const float * src, const float * factors, float * dst, int64_t dim) {
    const int64_t half = dim / 2;
    for (int64_t j = 0; j < half; ++j) {
        const float low = src[2 * j];
        const float high = src[2 * j + 1];
        const float c = factors[j];
        const float s = factors[j + half];
        const float lc = round_bf16_scalar(low * c);
        const float hs = round_bf16_scalar(high * s);
        const float hc = round_bf16_scalar(high * c);
        const float ls = round_bf16_scalar(low * s);
        dst[2 * j] = round_bf16_scalar(lc - hs);
        dst[2 * j + 1] = round_bf16_scalar(hc + ls);
    }
}

}  // namespace

const FastEngineSettings & fast_settings() {
    static const FastEngineSettings s = [] {
        FastEngineSettings v;
        v.flow_format = parse_format_env("MING_FLOW_QUANT", FlowWeightFormat::Q4_1_G64);
        v.aggregator_format = parse_format_env("MING_AGG_QUANT", v.flow_format);
        v.flow_steps = parse_int_env("MING_FLOW_STEPS", 10, 5, 10);
        v.cfg_interval = parse_int_env("MING_CFG_INTERVAL", 1, 1, 10);
        return v;
    }();
    return s;
}

std::vector<float> flow_time_grid(int steps) {
    if (steps < 5 || steps > 10) {
        throw std::runtime_error("Unsupported Ming flow steps (expected 5..10)");
    }
    std::vector<float> time;
    if (steps == 10) {
        time = {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
        for (auto & t : time) t /= 32.0f;
    } else {
        time.resize(static_cast<size_t>(steps + 1));
        for (int i = 0; i <= steps; ++i) {
            time[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(steps);
        }
    }
    auto phase = time;
    for (auto & t : phase) t *= 1.5707963267948966f;
    engine::core::round_f32_to_bf16_in_place(phase);
    for (auto & t : phase) t = std::cos(t);
    engine::core::round_f32_to_bf16_in_place(phase);
    for (auto & t : phase) t -= 1.0f;
    engine::core::round_f32_to_bf16_in_place(phase);
    for (size_t i = 0; i < time.size(); ++i) phase[i] += time[i];
    engine::core::round_f32_to_bf16_in_place(phase);
    for (size_t i = 0; i < time.size(); ++i) time[i] -= phase[i];
    engine::core::round_f32_to_bf16_in_place(time);
    return time;
}

float round_bf16_scalar(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        bits = (bits & 0xffff0000U) | 0x00400000U;
    } else {
        bits = (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
    }
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

bool cpu_supports_avx2_fma() {
#if defined(MING_FAST_AVX2)
    static const bool supported = __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return supported;
#else
    return false;
#endif
}

PackedQ8Matrix pack_q8_from_f32(const float * weights, const float * bias,
    int64_t out_features, int64_t in_features) {
    if (out_features <= 0 || in_features <= 0 || out_features % 2 != 0 || in_features % 32 != 0) {
        throw std::runtime_error("PackedQ8Matrix requires even out_features and in_features % 32 == 0");
    }
    const int64_t nb = in_features / 32;
    PackedQ8Matrix m;
    m.out_features = out_features;
    m.in_features = in_features;
    m.q.resize(static_cast<size_t>(out_features * in_features));
    m.scale.resize(static_cast<size_t>(out_features * nb));
    m.bias.assign(bias, bias + out_features);
    for (int64_t p = 0; p < out_features / 2; ++p) {
        for (int64_t b = 0; b < nb; ++b) {
            for (int r = 0; r < 2; ++r) {
                const float * src = weights + (p * 2 + r) * in_features + b * 32;
                float amax = 0.0f;
                for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(src[k]));
                const float d = amax / 127.0f;
                const float id = d > 0.0f ? 1.0f / d : 0.0f;
                const int64_t dst_idx = (p * nb + b) * 2 + r;
                m.scale[dst_idx] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
                int8_t * dst = m.q.data() + dst_idx * 32;
                for (int k = 0; k < 32; ++k) {
                    const int v = static_cast<int>(std::lrintf(src[k] * id));
                    dst[k] = static_cast<int8_t>(std::clamp(v, -127, 127));
                }
            }
        }
    }
    return m;
}

PackedQ4GMatrix pack_q4g_from_f32(const float * weights, const float * bias,
    int64_t out_features, int64_t in_features, int group_size) {
    if (group_size != 64 && group_size != 128) {
        throw std::runtime_error("PackedQ4GMatrix supports group_size 64 or 128");
    }
    if (out_features <= 0 || in_features <= 0 || in_features % group_size != 0) {
        throw std::runtime_error("PackedQ4GMatrix requires in_features divisible by group_size");
    }
    const int64_t ng = in_features / group_size;
    const int sub_blocks = group_size / 32;
    const size_t q4_bytes = static_cast<size_t>(in_features / 2);
    const size_t float_bytes = static_cast<size_t>(ng) * sizeof(float);
    const size_t scale_off = (q4_bytes + 31U) & ~size_t{31};
    const size_t min_off = (scale_off + float_bytes + 31U) & ~size_t{31};
    const size_t stride = (min_off + float_bytes + 31U) & ~size_t{31};

    PackedQ4GMatrix m;
    m.out_features = out_features;
    m.in_features = in_features;
    m.group_size = group_size;
    m.scale_offset = scale_off;
    m.min_offset = min_off;
    m.channel_stride = stride;
    m.data.assign(static_cast<size_t>(out_features) * stride, 0);
    m.bias.assign(bias, bias + out_features);

    #pragma omp parallel for schedule(static)
    for (int64_t ch = 0; ch < out_features; ++ch) {
        const float * row = weights + ch * in_features;
        uint8_t * slab = m.data.data() + static_cast<size_t>(ch) * stride;
        uint8_t * dst_q4 = slab;
        float * dst_scale = reinterpret_cast<float *>(slab + scale_off);
        float * dst_min = reinterpret_cast<float *>(slab + min_off);
        for (int64_t g = 0; g < ng; ++g) {
            const float * src = row + g * group_size;
            float wmin = src[0];
            float wmax = src[0];
            for (int k = 1; k < group_size; ++k) {
                wmin = std::min(wmin, src[k]);
                wmax = std::max(wmax, src[k]);
            }
            const float range = wmax - wmin;
            const float scale = range / 15.0f;
            const float inv_scale = range > 0.0f ? (15.0f / range) : 0.0f;
            dst_scale[g] = scale;
            dst_min[g] = wmin;
            uint8_t * gp = dst_q4 + g * (group_size / 2);
            for (int s = 0; s < sub_blocks; ++s) {
                const float * sp = src + s * 32;
                uint8_t * bp = gp + s * 16;
                for (int j = 0; j < 16; ++j) {
                    const int lo = std::clamp(static_cast<int>(std::lrintf((sp[j] - wmin) * inv_scale)), 0, 15);
                    const int hi = std::clamp(static_cast<int>(std::lrintf((sp[j + 16] - wmin) * inv_scale)), 0, 15);
                    bp[j] = static_cast<uint8_t>(lo | (hi << 4));
                }
            }
        }
    }
    return m;
}

void quantize_rows_q8(const float * x, int64_t rows, int64_t in_features, Q8InputBuffer & out) {
    const int64_t nb = in_features / 32;
    out.rows = rows;
    out.in_features = in_features;
    out.q.resize(static_cast<size_t>(rows * in_features));
    out.scale.resize(static_cast<size_t>(rows * nb));
    const bool use_avx2 = cpu_supports_avx2_fma();
    for (int64_t r = 0; r < rows; ++r) {
#if defined(MING_FAST_AVX2)
        if (use_avx2) {
            quantize_row_q8_avx2(x + r * in_features, in_features,
                out.q.data() + r * in_features, out.scale.data() + r * nb);
            continue;
        }
#endif
        quantize_row_q8_scalar(x + r * in_features, in_features,
            out.q.data() + r * in_features, out.scale.data() + r * nb);
    }
}

void quantize_rows_q8_group(const float * x, int64_t rows, int64_t in_features, int group_size,
    Q8GroupInputBuffer & out) {
    const int64_t ng = in_features / group_size;
    out.rows = rows;
    out.in_features = in_features;
    out.group_size = group_size;
    out.q.resize(static_cast<size_t>(rows * in_features));
    out.scale.resize(static_cast<size_t>(rows * ng));
    out.sum.resize(static_cast<size_t>(rows * ng));
    const bool use_avx2 = cpu_supports_avx2_fma();
    for (int64_t r = 0; r < rows; ++r) {
#if defined(MING_FAST_AVX2)
        if (use_avx2) {
            quantize_row_q8_group_major_avx2(x + r * in_features, r, rows, in_features, group_size,
                out.q.data(), out.scale.data(), out.sum.data() + r * ng);
            continue;
        }
#endif
        quantize_row_q8_group_major_scalar(x + r * in_features, r, rows, in_features, group_size,
            out.q.data(), out.scale.data(), out.sum.data() + r * ng);
    }
}

void gemm_q8_slice(const PackedQ8Matrix & w, const Q8InputBuffer & x,
    float * y, const float * residual, const float * gelu_table,
    int epilogue, int64_t ch_begin, int64_t ch_end, bool allow_avx2) {
    if (ch_begin >= ch_end) return;
#if defined(MING_FAST_AVX2)
    if (allow_avx2 && cpu_supports_avx2_fma() && (ch_begin % 2 == 0) && (ch_end % 2 == 0)) {
        switch (x.rows) {
            case 1: gemm_q8_avx2_2xM<1>(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end); return;
            case 2: gemm_q8_avx2_2xM<2>(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end); return;
            case 3: gemm_q8_avx2_2xM<3>(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end); return;
            case 4: gemm_q8_avx2_2xM<4>(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end); return;
            case 5: gemm_q8_avx2_2xM<5>(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end); return;
            case 6: gemm_q8_avx2_2xM<6>(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end); return;
            default: break;
        }
    }
#else
    (void) allow_avx2;
#endif
    gemm_q8_scalar(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end);
}

namespace {

#if defined(MING_FAST_AVX2)
template <int SubBlocks, int Epilogue>
bool dispatch_q4g_avx2_rows(const PackedQ4GMatrix & w, const Q8GroupInputBuffer & x,
    float * y, const float * residual, const float * gelu_table,
    int64_t ch_begin, int64_t ch_end, int64_t y_stride, int64_t y_ch_offset) {
    const bool mult4 = ((ch_end - ch_begin) % 4) == 0;
    switch (x.rows) {
        case 1:
            if (mult4) gemm_q4g_avx2_4xM<1, SubBlocks, Epilogue>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset);
            else gemm_q4g_avx2_1xM<1, SubBlocks, Epilogue>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset);
            return true;
        case 2:
            if (mult4) gemm_q4g_avx2_4xM<2, SubBlocks, Epilogue>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset);
            else gemm_q4g_avx2_1xM<2, SubBlocks, Epilogue>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset);
            return true;
        case 3: gemm_q4g_avx2_1xM<3, SubBlocks, Epilogue>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset); return true;
        case 4: gemm_q4g_avx2_1xM<4, SubBlocks, Epilogue>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset); return true;
        case 5: gemm_q4g_avx2_1xM<5, SubBlocks, Epilogue>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset); return true;
        case 6: gemm_q4g_avx2_1xM<6, SubBlocks, Epilogue>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset); return true;
        default: return false;
    }
}

template <int SubBlocks>
bool dispatch_q4g_avx2(const PackedQ4GMatrix & w, const Q8GroupInputBuffer & x,
    float * y, const float * residual, const float * gelu_table, int epilogue,
    int64_t ch_begin, int64_t ch_end, int64_t y_stride, int64_t y_ch_offset) {
    if (epilogue == 0) return dispatch_q4g_avx2_rows<SubBlocks, 0>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset);
    if (epilogue == 1) return dispatch_q4g_avx2_rows<SubBlocks, 1>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset);
    return dispatch_q4g_avx2_rows<SubBlocks, 2>(w, x, y, residual, gelu_table, ch_begin, ch_end, y_stride, y_ch_offset);
}
#endif

void gemm_q4g_slice_strided(const PackedQ4GMatrix & w, const Q8GroupInputBuffer & x,
    float * y, const float * residual, const float * gelu_table,
    int epilogue, int64_t ch_begin, int64_t ch_end, int64_t y_stride, int64_t y_ch_offset, bool allow_avx2 = true) {
    if (ch_begin >= ch_end) return;
#if defined(MING_FAST_AVX2)
    if (allow_avx2 && cpu_supports_avx2_fma()) {
        if (w.group_size == 64) {
            if (dispatch_q4g_avx2<2>(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end, y_stride, y_ch_offset)) return;
        } else if (w.group_size == 128) {
            if (dispatch_q4g_avx2<4>(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end, y_stride, y_ch_offset)) return;
        }
    }
#else
    (void) allow_avx2;
#endif
    gemm_q4g_scalar(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end, y_stride, y_ch_offset);
}

}  // namespace

void gemm_q4g_slice(const PackedQ4GMatrix & w, const Q8GroupInputBuffer & x,
    float * y, const float * residual, const float * gelu_table,
    int epilogue, int64_t ch_begin, int64_t ch_end, bool allow_avx2) {
    gemm_q4g_slice_strided(w, x, y, residual, gelu_table, epilogue, ch_begin, ch_end, w.out_features, 0, allow_avx2);
}

struct DirectPatchNetwork::Impl {
    struct Block {
        std::vector<float> norm1, norm2;
        FastLinear qkv, attn_out, ff_in, ff_out;
    };

    struct Workspace {
        std::vector<float> x;          // [6, hidden]
        std::vector<float> qkv_small;  // [3, 3*hidden]
        std::vector<float> qkv_full;   // [6, 3*hidden]
        std::vector<float> attn_ctx;   // [6, hidden]
        std::vector<float> ff_hidden;  // [6, intermediate]
        std::vector<float> output;     // [3, output_dim]
        std::vector<float> cond_emb;   // [hidden]
        std::vector<float> hist_emb;   // [hidden]
        std::vector<float> hist_qkv;   // [3*hidden]
        std::vector<float> lat_emb;    // [hidden]
        Q8InputBuffer q8_in;
        Q8GroupInputBuffer q4g_in;
    };

    Impl(const engine::assets::TensorSource & source,
        const engine::modules::MingPatchNetworkConfig & config,
        const std::string & prefix, int64_t latent_dim, int64_t llm_dim,
        bool conditioned, FlowWeightFormat format, engine::core::ExecutionContext & execution)
        : threads_(std::max(1, execution.config().threads)),
          conditioned_(conditioned),
          format_(format),
          latent_dim_(latent_dim),
          llm_dim_(llm_dim),
          hidden_(config.hidden_size),
          heads_(config.heads),
          head_dim_(config.hidden_size / config.heads),
          intermediate_(config.intermediate_size),
          output_dim_(conditioned ? latent_dim : llm_dim),
          steps_(conditioned ? 3 : 2) {
        if (hidden_ % heads_ != 0 || hidden_ % 128 != 0 || intermediate_ % 128 != 0 || head_dim_ > 64) {
            throw std::runtime_error("DirectPatchNetwork requires hidden/intermediate % 128 == 0 and head_dim <= 64");
        }
        x_emb_w_ = source.require_f32(prefix + ".x_embedder.weight", {hidden_, latent_dim_});
        x_emb_b_ = source.require_f32(prefix + ".x_embedder.bias", {hidden_});
        final_norm_ = source.require_f32(prefix + ".final_layer.norm_final.weight", {hidden_});
        output_proj_ = load_fast_linear(source, prefix + ".final_layer.linear.weight",
            prefix + ".final_layer.linear.bias", output_dim_, hidden_, format_);
        if (conditioned_) {
            c_emb_w_ = source.require_f32(prefix + ".c_embedder.cond_embedder.weight", {hidden_, llm_dim_});
            c_emb_b_ = source.require_f32(prefix + ".c_embedder.cond_embedder.bias", {hidden_});
            t_in_w_ = source.require_f32(prefix + ".t_embedder.time_mlp.0.weight", {hidden_, 256});
            t_in_b_ = source.require_f32(prefix + ".t_embedder.time_mlp.0.bias", {hidden_});
            t_out_w_ = source.require_f32(prefix + ".t_embedder.time_mlp.2.weight", {hidden_, hidden_});
            t_out_b_ = source.require_f32(prefix + ".t_embedder.time_mlp.2.bias", {hidden_});
            uncond_emb_.resize(static_cast<size_t>(hidden_));
            for (int64_t i = 0; i < hidden_; ++i) uncond_emb_[i] = round_bf16_scalar(c_emb_b_[i]);
        } else {
            cls_ = source.require_f32(prefix + ".word_embedder.weight", {1, hidden_});
            engine::core::round_f32_to_bf16_in_place(cls_);
        }
        const auto inv_freq = source.require_f32(prefix + ".rotary_embed.inv_freq");
        rope_table_ = build_rope_table_exact(inv_freq, steps_, head_dim_);
        gelu_table_ = make_gelu_bf16_table(execution);
        blocks_.reserve(static_cast<size_t>(config.depth));
        for (int64_t i = 0; i < config.depth; ++i) {
            const std::string p = prefix + ".blocks." + std::to_string(i);
            Block b;
            b.norm1 = source.require_f32(p + ".norm1.weight", {hidden_});
            b.norm2 = source.require_f32(p + ".norm2.weight", {hidden_});
            b.qkv = load_fast_qkv(source, p, hidden_, format_);
            b.attn_out = load_fast_linear(source, p + ".attn.to_out.0.weight",
                p + ".attn.to_out.0.bias", hidden_, hidden_, format_);
            b.ff_in = load_fast_linear(source, p + ".mlp.ff.0.0.weight",
                p + ".mlp.ff.0.0.bias", intermediate_, hidden_, format_);
            b.ff_out = load_fast_linear(source, p + ".mlp.ff.2.weight",
                p + ".mlp.ff.2.bias", hidden_, intermediate_, format_);
            blocks_.push_back(std::move(b));
        }
        init_workspace();
        if (!conditioned_) {
            cls_qkv_.resize(static_cast<size_t>(3 * hidden_));
            #pragma omp parallel num_threads(threads_)
            {
                const int tid = omp_get_thread_num();
                const int nth = omp_get_num_threads();
                compute_single_row_block0_qkv_in_parallel(cls_.data(), cls_qkv_.data(), tid, nth);
            }
        }
    }

    void init_workspace() {
        const int64_t max_rows = 6;
        ws_.x.resize(static_cast<size_t>(max_rows * hidden_));
        ws_.qkv_small.resize(static_cast<size_t>(3 * 3 * hidden_));
        ws_.qkv_full.resize(static_cast<size_t>(max_rows * 3 * hidden_));
        ws_.attn_ctx.resize(static_cast<size_t>(max_rows * hidden_));
        ws_.ff_hidden.resize(static_cast<size_t>(max_rows * intermediate_));
        ws_.output.resize(static_cast<size_t>(3 * output_dim_));
        ws_.cond_emb.resize(static_cast<size_t>(hidden_));
        ws_.hist_emb.resize(static_cast<size_t>(hidden_));
        ws_.hist_qkv.resize(static_cast<size_t>(3 * hidden_));
        ws_.lat_emb.resize(static_cast<size_t>(hidden_));
        ws_.q8_in.q.resize(static_cast<size_t>(max_rows * intermediate_));
        ws_.q8_in.scale.resize(static_cast<size_t>(max_rows * (intermediate_ / 32)));
        const int gs = (format_ == FlowWeightFormat::Q4_1_G128) ? 128 : 64;
        ws_.q4g_in.q.resize(static_cast<size_t>(max_rows * intermediate_));
        ws_.q4g_in.scale.resize(static_cast<size_t>(max_rows * (intermediate_ / gs)));
        ws_.q4g_in.sum.resize(static_cast<size_t>(max_rows * (intermediate_ / gs)));
    }

    const std::vector<float> & get_time_emb(float t) {
        uint32_t bits;
        std::memcpy(&bits, &t, sizeof(bits));
        for (const auto & entry : time_cache_) {
            if (entry.first == bits) return entry.second;
        }
        const float scaled = round_bf16_scalar(1000.0f * t);
        std::vector<float> emb(256);
        const float step = static_cast<float>(std::log(10000.0) / 127.0);
        for (size_t i = 0; i < 128; ++i) {
            const float phase = scaled * std::exp(-step * static_cast<float>(i));
            emb[i] = round_bf16_scalar(std::sin(phase));
            emb[128 + i] = round_bf16_scalar(std::cos(phase));
        }
        std::vector<float> h(static_cast<size_t>(hidden_));
        linear_f32_row(t_in_w_.data(), t_in_b_.data(), emb.data(), h.data(), hidden_, 256);
        for (auto & v : h) {
            v = round_bf16_scalar(v / (1.0f + std::exp(-v)));
        }
        std::vector<float> out(static_cast<size_t>(hidden_));
        linear_f32_row(t_out_w_.data(), t_out_b_.data(), h.data(), out.data(), hidden_, hidden_);
        time_cache_.emplace_back(bits, std::move(out));
        return time_cache_.back().second;
    }

    void set_quant_header(int64_t rows, int64_t width) {
        if (format_ == FlowWeightFormat::Q8_0) {
            ws_.q8_in.rows = rows;
            ws_.q8_in.in_features = width;
        } else {
            ws_.q4g_in.rows = rows;
            ws_.q4g_in.in_features = width;
            ws_.q4g_in.group_size = (format_ == FlowWeightFormat::Q4_1_G128) ? 128 : 64;
        }
    }

    void quantize_one_row(const float * row, int64_t r, int64_t rows, int64_t width) {
        const bool use_avx2 = cpu_supports_avx2_fma();
        if (format_ == FlowWeightFormat::Q8_0) {
            const int64_t nb = width / 32;
#if defined(MING_FAST_AVX2)
            if (use_avx2) {
                quantize_row_q8_avx2(row, width, ws_.q8_in.q.data() + r * width,
                    ws_.q8_in.scale.data() + r * nb);
                return;
            }
#endif
            quantize_row_q8_scalar(row, width, ws_.q8_in.q.data() + r * width,
                ws_.q8_in.scale.data() + r * nb);
        } else {
            const int gs = (format_ == FlowWeightFormat::Q4_1_G128) ? 128 : 64;
            const int64_t ng = width / gs;
#if defined(MING_FAST_AVX2)
            if (use_avx2) {
                quantize_row_q8_group_major_avx2(row, r, rows, width, gs,
                    ws_.q4g_in.q.data(), ws_.q4g_in.scale.data(), ws_.q4g_in.sum.data() + r * ng);
                return;
            }
#endif
            quantize_row_q8_group_major_scalar(row, r, rows, width, gs,
                ws_.q4g_in.q.data(), ws_.q4g_in.scale.data(), ws_.q4g_in.sum.data() + r * ng);
        }
    }

    // Fused RMSNorm + BF16 + Quantize on a single row using an L1 stack buffer (saves 1 barrier + norm_buf traffic).
    void norm_and_quantize_one_row(const float * x_row, const float * norm_w,
        int64_t r, int64_t rows, int64_t width) {
        float tmp[4096];
        rms_norm_bf16_row(x_row, norm_w, tmp, width);
        quantize_one_row(tmp, r, rows, width);
    }

    void quantize_rows_parallel(const float * src, int64_t rows, int64_t width, int tid, int nth) {
        if (tid == 0) set_quant_header(rows, width);
        for (int64_t r = tid; r < rows; r += nth) {
            quantize_one_row(src + r * width, r, rows, width);
        }
    }

    void norm_and_quantize_rows_parallel(const float * src, const float * norm_w,
        int64_t rows, int64_t width, int tid, int nth) {
        if (tid == 0) set_quant_header(rows, width);
        for (int64_t r = tid; r < rows; r += nth) {
            norm_and_quantize_one_row(src + r * width, norm_w, r, rows, width);
        }
    }

    void run_linear_slice(const FastLinear & w, float * y, const float * residual,
        int epilogue, int tid, int nth) {
        const int64_t out_ch = w.out_features;
        const int64_t ch_begin = (out_ch / 2) * tid / nth * 2;
        const int64_t ch_end = (out_ch / 2) * (tid + 1) / nth * 2;
        if (w.format == FlowWeightFormat::Q8_0) {
            gemm_q8_slice(w.q8, ws_.q8_in, y, residual, gelu_table_.data(), epilogue, ch_begin, ch_end);
        } else {
            gemm_q4g_slice(w.q4g, ws_.q4g_in, y, residual, gelu_table_.data(), epilogue, ch_begin, ch_end);
        }
    }

    void compute_single_row_block0_qkv_in_parallel(const float * row_emb, float * out_qkv, int tid, int nth) {
        if (tid == 0) {
            set_quant_header(1, hidden_);
            norm_and_quantize_one_row(row_emb, blocks_[0].norm1.data(), 0, 1, hidden_);
        }
        #pragma omp barrier
        run_linear_slice(blocks_[0].qkv, out_qkv, nullptr, 0, tid, nth);
    }

    // Core transformer forward pass called INSIDE an already-active #pragma omp parallel region.
    void forward_blocks_in_parallel(int64_t batch, const float * cached_fixed_qkv,
        bool share_pos2_across_batch, int tid, int nth) {
        const int64_t steps = steps_;
        const int64_t total_rows = batch * steps;
        const int64_t hidden = hidden_;
        const int64_t qkv_stride = 3 * hidden;
        const int64_t heads = heads_;
        const int64_t dim = head_dim_;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(dim));
        const int64_t output_pos = conditioned_ ? (steps - 1) : 0;
        const size_t depth = blocks_.size();
        const bool use_avx2_attn = cpu_supports_avx2_fma() && dim == 64;

        for (size_t bi = 0; bi < depth; ++bi) {
            const auto & blk = blocks_[bi];
            const bool output_only = (bi + 1 == depth);
            const int64_t q_steps = output_only ? 1 : steps;
            const int64_t q_rows = batch * q_steps;

            if (bi == 0 && cached_fixed_qkv != nullptr) {
                int64_t small_rows = 0;
                if (conditioned_) {
                    small_rows = (batch == 2 && share_pos2_across_batch) ? 3 : (batch == 1 ? 2 : 4);
                    if (tid == 0) set_quant_header(small_rows, hidden);
                    // Parallelize the 2 or 3 rows across threads!
                    const int64_t src_idx[3] = {0, (batch == 2 ? 3 : 2), 2};
                    for (int64_t r = tid; r < small_rows; r += nth) {
                        norm_and_quantize_one_row(ws_.x.data() + src_idx[r] * hidden,
                            blk.norm1.data(), r, small_rows, hidden);
                    }
                } else {
                    small_rows = batch;
                    if (tid == 0) set_quant_header(small_rows, hidden);
                    for (int64_t b = tid; b < batch; b += nth) {
                        norm_and_quantize_one_row(ws_.x.data() + (b * 2 + 1) * hidden,
                            blk.norm1.data(), b, small_rows, hidden);
                    }
                }
                #pragma omp barrier
                run_linear_slice(blk.qkv, ws_.qkv_small.data(), nullptr, 0, tid, nth);
                #pragma omp barrier
                if (tid == 0) {
                    const size_t row_bytes = static_cast<size_t>(qkv_stride) * sizeof(float);
                    if (conditioned_) {
                        std::memcpy(ws_.qkv_full.data() + 0 * qkv_stride, ws_.qkv_small.data() + 0 * qkv_stride, row_bytes);
                        std::memcpy(ws_.qkv_full.data() + 1 * qkv_stride, cached_fixed_qkv, row_bytes);
                        const float * lat_qkv = ws_.qkv_small.data() + (batch == 2 ? 2 : 1) * qkv_stride;
                        std::memcpy(ws_.qkv_full.data() + 2 * qkv_stride, lat_qkv, row_bytes);
                        if (batch == 2) {
                            std::memcpy(ws_.qkv_full.data() + 3 * qkv_stride, ws_.qkv_small.data() + 1 * qkv_stride, row_bytes);
                            std::memcpy(ws_.qkv_full.data() + 4 * qkv_stride, cached_fixed_qkv, row_bytes);
                            std::memcpy(ws_.qkv_full.data() + 5 * qkv_stride, lat_qkv, row_bytes);
                        }
                    } else {
                        for (int64_t b = 0; b < batch; ++b) {
                            std::memcpy(ws_.qkv_full.data() + (b * 2 + 0) * qkv_stride, cached_fixed_qkv, row_bytes);
                            std::memcpy(ws_.qkv_full.data() + (b * 2 + 1) * qkv_stride,
                                ws_.qkv_small.data() + b * qkv_stride, row_bytes);
                        }
                    }
                }
                #pragma omp barrier
            } else {
                norm_and_quantize_rows_parallel(ws_.x.data(), blk.norm1.data(), total_rows, hidden, tid, nth);
                #pragma omp barrier
                run_linear_slice(blk.qkv, ws_.qkv_full.data(), nullptr, 0, tid, nth);
                #pragma omp barrier
            }

            // Fused RoPE + Multi-Head Scaled Dot-Product Attention + BF16 round (in L1 stack memory).
            const int64_t total_heads = batch * heads;
            for (int64_t bh = tid; bh < total_heads; bh += nth) {
                const int64_t b = bh / heads;
                const int64_t h = bh % heads;
                float q_rot[3][64];
                float k_rot[3][64];
                const float * v_ptr[3];
                for (int64_t s = 0; s < steps; ++s) {
                    const float * row_qkv = ws_.qkv_full.data() + (b * steps + s) * qkv_stride;
                    const float * factors = rope_table_.data() + s * dim;
                    apply_rope_vec(row_qkv + h * dim, factors, q_rot[s], dim);
                    apply_rope_vec(row_qkv + hidden + h * dim, factors, k_rot[s], dim);
                    v_ptr[s] = row_qkv + 2 * hidden + h * dim;
                }
                const int64_t sq_start = output_only ? output_pos : 0;
                const int64_t sq_end = output_only ? (output_pos + 1) : steps;
                for (int64_t sq = sq_start; sq < sq_end; ++sq) {
                    float scores[3];
                    float max_score = -1e30f;
                    for (int64_t sk = 0; sk < steps; ++sk) {
                        float dot = 0.0f;
#if defined(MING_FAST_AVX2)
                        if (use_avx2_attn) {
                            dot = dot64_avx2(q_rot[sq], k_rot[sk]);
                        } else
#endif
                        {
                            for (int64_t d = 0; d < dim; ++d) dot += q_rot[sq][d] * k_rot[sk][d];
                        }
                        scores[sk] = dot * attn_scale;
                        if (scores[sk] > max_score) max_score = scores[sk];
                    }
                    float sum_exp = 0.0f;
                    for (int64_t sk = 0; sk < steps; ++sk) {
                        scores[sk] = std::exp(scores[sk] - max_score);
                        sum_exp += scores[sk];
                    }
                    const float inv_sum = 1.0f / sum_exp;
                    for (int64_t sk = 0; sk < steps; ++sk) scores[sk] *= inv_sum;
                    const int64_t out_row = output_only ? b : (b * steps + sq);
                    float * dst = ws_.attn_ctx.data() + out_row * hidden + h * dim;
#if defined(MING_FAST_AVX2)
                    if (use_avx2_attn) {
                        weighted_sum_bf16_64_avx2(steps, scores, v_ptr, dst);
                    } else
#endif
                    {
                        for (int64_t d = 0; d < dim; ++d) {
                            float val = 0.0f;
                            for (int64_t sk = 0; sk < steps; ++sk) val += scores[sk] * v_ptr[sk][d];
                            dst[d] = round_bf16_scalar(val);
                        }
                    }
                }
            }
            if (output_only && tid == 0) {
                for (int64_t b = 0; b < batch; ++b) {
                    const float * src = ws_.x.data() + (b * steps + output_pos) * hidden;
                    float * dst = ws_.x.data() + b * hidden;
                    if (src != dst) {
                        std::memmove(dst, src, static_cast<size_t>(hidden) * sizeof(float));
                    }
                }
            }
            #pragma omp barrier

            // Attention out projection + fused residual add.
            quantize_rows_parallel(ws_.attn_ctx.data(), q_rows, hidden, tid, nth);
            #pragma omp barrier
            run_linear_slice(blk.attn_out, ws_.x.data(), ws_.x.data(), 1, tid, nth);
            #pragma omp barrier

            // Fused RMSNorm2 + Quantize -> FF_In + GELU LUT -> Quantize -> FF_Out + residual.
            norm_and_quantize_rows_parallel(ws_.x.data(), blk.norm2.data(), q_rows, hidden, tid, nth);
            #pragma omp barrier
            run_linear_slice(blk.ff_in, ws_.ff_hidden.data(), nullptr, 2, tid, nth);
            #pragma omp barrier
            quantize_rows_parallel(ws_.ff_hidden.data(), q_rows, intermediate_, tid, nth);
            #pragma omp barrier
            run_linear_slice(blk.ff_out, ws_.x.data(), ws_.x.data(), 1, tid, nth);
            #pragma omp barrier
        }

        // Fused Final Norm + Quantize -> Output projection.
        norm_and_quantize_rows_parallel(ws_.x.data(), final_norm_.data(), batch, hidden, tid, nth);
        #pragma omp barrier
        run_linear_slice(output_proj_, ws_.output.data(), nullptr, 0, tid, nth);
    }

    void setup_flow_step_inputs(const float * latent, const float * cond_emb,
        const float * history_emb, const float * t_emb, bool dual) {
        linear_f32_row(x_emb_w_.data(), x_emb_b_.data(), latent, ws_.lat_emb.data(), hidden_, latent_dim_);
        for (int64_t i = 0; i < hidden_; ++i) {
            ws_.x[0 * hidden_ + i] = round_bf16_scalar(t_emb[i] + cond_emb[i]);
        }
        std::memcpy(ws_.x.data() + 1 * hidden_, history_emb, static_cast<size_t>(hidden_) * sizeof(float));
        std::memcpy(ws_.x.data() + 2 * hidden_, ws_.lat_emb.data(), static_cast<size_t>(hidden_) * sizeof(float));
        if (dual) {
            for (int64_t i = 0; i < hidden_; ++i) {
                ws_.x[3 * hidden_ + i] = round_bf16_scalar(t_emb[i] + uncond_emb_[i]);
            }
            std::memcpy(ws_.x.data() + 4 * hidden_, history_emb, static_cast<size_t>(hidden_) * sizeof(float));
            std::memcpy(ws_.x.data() + 5 * hidden_, ws_.lat_emb.data(), static_cast<size_t>(hidden_) * sizeof(float));
        }
    }

    std::vector<float> run(const std::vector<float> & latent, const std::vector<float> & condition,
        const std::vector<float> & history, float time, bool dual) {
        if (latent.empty() || latent.size() % static_cast<size_t>(latent_dim_) != 0 ||
            (conditioned_ && (latent.size() != static_cast<size_t>(latent_dim_) ||
                condition.size() != static_cast<size_t>(llm_dim_) || history.size() != latent.size()))) {
            throw std::runtime_error("Ming patch network input dimensions do not match the checkpoint");
        }
        if (conditioned_) {
            const auto & t_emb = get_time_emb(time);
            linear_f32_row(c_emb_w_.data(), c_emb_b_.data(), condition.data(), ws_.cond_emb.data(), hidden_, llm_dim_);
            linear_f32_row(x_emb_w_.data(), x_emb_b_.data(), history.data(), ws_.hist_emb.data(), hidden_, latent_dim_);
            setup_flow_step_inputs(latent.data(), ws_.cond_emb.data(), ws_.hist_emb.data(), t_emb.data(), dual);
            const int64_t batch = dual ? 2 : 1;
            #pragma omp parallel num_threads(threads_)
            {
                const int tid = omp_get_thread_num();
                const int nth = omp_get_num_threads();
                compute_single_row_block0_qkv_in_parallel(ws_.hist_emb.data(), ws_.hist_qkv.data(), tid, nth);
                #pragma omp barrier
                forward_blocks_in_parallel(batch, ws_.hist_qkv.data(), true, tid, nth);
            }
            return std::vector<float>(ws_.output.begin(), ws_.output.begin() + batch * output_dim_);
        }
        const int64_t total_batch = static_cast<int64_t>(latent.size() / latent_dim_);
        std::vector<float> result;
        result.reserve(static_cast<size_t>(total_batch * output_dim_));
        for (int64_t b0 = 0; b0 < total_batch; b0 += 3) {
            const int64_t chunk = std::min<int64_t>(3, total_batch - b0);
            for (int64_t b = 0; b < chunk; ++b) {
                std::memcpy(ws_.x.data() + (b * 2 + 0) * hidden_, cls_.data(),
                    static_cast<size_t>(hidden_) * sizeof(float));
                linear_f32_row(x_emb_w_.data(), x_emb_b_.data(),
                    latent.data() + (b0 + b) * latent_dim_,
                    ws_.x.data() + (b * 2 + 1) * hidden_, hidden_, latent_dim_);
            }
            #pragma omp parallel num_threads(threads_)
            {
                const int tid = omp_get_thread_num();
                const int nth = omp_get_num_threads();
                forward_blocks_in_parallel(chunk, cls_qkv_.data(), false, tid, nth);
            }
            result.insert(result.end(), ws_.output.begin(), ws_.output.begin() + chunk * output_dim_);
        }
        return result;
    }

    std::vector<float> sample(const std::vector<float> & conditioning,
        const std::vector<float> & history, const std::vector<float> & noise,
        const std::vector<float> & sde_noise, const std::vector<float> & time_grid,
        float cfg, float sigma, float temperature, int cfg_interval) {
        const size_t steps = time_grid.size() - 1;
        if (sde_noise.size() != noise.size() * steps || temperature < 0.0f) {
            throw std::runtime_error("Ming flow requires one SDE-noise frame per sampling step and nonnegative temperature");
        }
        // Ensure time embeddings for all steps are cached before entering the persistent parallel region.
        std::vector<const float *> t_embs(steps);
        for (size_t step = 0; step < steps; ++step) {
            t_embs[step] = get_time_emb(time_grid[step]).data();
        }
        linear_f32_row(c_emb_w_.data(), c_emb_b_.data(), conditioning.data(), ws_.cond_emb.data(), hidden_, llm_dim_);
        linear_f32_row(x_emb_w_.data(), x_emb_b_.data(), history.data(), ws_.hist_emb.data(), hidden_, latent_dim_);

        auto y = noise;
        std::vector<float> cached_delta(noise.size(), 0.0f);
        std::vector<float> guided(noise.size());
        const float noise_scale = sigma * std::sqrt(temperature);

        // ONE single persistent parallel region for ALL 10 Flow steps of the entire patch!
        #pragma omp parallel num_threads(threads_)
        {
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();

            compute_single_row_block0_qkv_in_parallel(ws_.hist_emb.data(), ws_.hist_qkv.data(), tid, nth);
            #pragma omp barrier

            for (size_t step = 0; step < steps; ++step) {
                const bool dual = (cfg_interval <= 1) || (step == 0) || (step + 1 == steps) || (step % cfg_interval == 0);
                const int64_t batch = dual ? 2 : 1;
                if (tid == 0) {
                    setup_flow_step_inputs(y.data(), ws_.cond_emb.data(), ws_.hist_emb.data(), t_embs[step], dual);
                }
                #pragma omp barrier
                forward_blocks_in_parallel(batch, ws_.hist_qkv.data(), true, tid, nth);
                #pragma omp barrier
                if (tid == 0) {
                    const float * prediction = ws_.output.data();
                    if (dual) {
                        for (size_t i = 0; i < guided.size(); ++i) {
                            guided[i] = round_bf16_scalar(prediction[i] - prediction[noise.size() + i]);
                        }
                        if (cfg_interval > 1) cached_delta = guided;
                    } else {
                        guided = cached_delta;
                    }
                    const float dt = round_bf16_scalar(time_grid[step + 1] - time_grid[step]);
                    const float sde_coeff = noise_scale * std::sqrt(std::abs(dt));
                    for (size_t i = 0; i < guided.size(); ++i) {
                        float g = round_bf16_scalar(guided[i] * cfg);
                        g = round_bf16_scalar(g + prediction[i]);
                        g = round_bf16_scalar(g * dt);
                        float next_y = round_bf16_scalar(y[i] + g);
                        if (temperature > 0.0f && sigma > 0.0f) {
                            const float n = round_bf16_scalar(sde_coeff * sde_noise[step * y.size() + i]);
                            next_y = round_bf16_scalar(next_y + n);
                        } else {
                            next_y = round_bf16_scalar(next_y + 0.0f);
                        }
                        y[i] = next_y;
                    }
                }
            }
        }
        return y;
    }

    size_t weight_bytes() const {
        size_t total = (x_emb_w_.size() + x_emb_b_.size() + final_norm_.size() +
            c_emb_w_.size() + c_emb_b_.size() + t_in_w_.size() + t_in_b_.size() +
            t_out_w_.size() + t_out_b_.size() + cls_.size()) * sizeof(float);
        total += output_proj_.bytes();
        for (const auto & b : blocks_) {
            total += (b.norm1.size() + b.norm2.size()) * sizeof(float);
            total += b.qkv.bytes() + b.attn_out.bytes() + b.ff_in.bytes() + b.ff_out.bytes();
        }
        return total;
    }

    int threads_ = 1;
    bool conditioned_ = false;
    FlowWeightFormat format_ = FlowWeightFormat::Q8_0;
    int64_t latent_dim_ = 0, llm_dim_ = 0, hidden_ = 0, heads_ = 0, head_dim_ = 0;
    int64_t intermediate_ = 0, output_dim_ = 0, steps_ = 0;

    std::vector<float> x_emb_w_, x_emb_b_, c_emb_w_, c_emb_b_, t_in_w_, t_in_b_, t_out_w_, t_out_b_;
    std::vector<float> final_norm_, cls_, uncond_emb_, cls_qkv_;
    std::vector<float> rope_table_, gelu_table_;
    std::vector<std::pair<uint32_t, std::vector<float>>> time_cache_;
    std::vector<Block> blocks_;
    FastLinear output_proj_;
    Workspace ws_;
};

DirectPatchNetwork::DirectPatchNetwork(const engine::assets::TensorSource & source,
    const engine::modules::MingPatchNetworkConfig & config,
    const std::string & prefix, int64_t latent_dim, int64_t llm_dim,
    bool conditioned, FlowWeightFormat format, engine::core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(source, config, prefix, latent_dim, llm_dim, conditioned, format, execution)) {}

DirectPatchNetwork::~DirectPatchNetwork() = default;

std::vector<float> DirectPatchNetwork::run(const std::vector<float> & latent,
    const std::vector<float> & condition, const std::vector<float> & history, float time, bool dual) {
    return impl_->run(latent, condition, history, time, dual);
}

std::vector<float> DirectPatchNetwork::sample(const std::vector<float> & conditioning,
    const std::vector<float> & history, const std::vector<float> & noise,
    const std::vector<float> & sde_noise, const std::vector<float> & time_grid,
    float cfg, float sigma, float temperature, int cfg_interval) {
    return impl_->sample(conditioning, history, noise, sde_noise, time_grid, cfg, sigma, temperature, cfg_interval);
}

size_t DirectPatchNetwork::weight_bytes() const {
    return impl_->weight_bytes();
}

}  // namespace ming_cpu
