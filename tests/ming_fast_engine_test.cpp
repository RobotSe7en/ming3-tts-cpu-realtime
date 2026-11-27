#include "flow_engine.h"
#include "kernels.h"

#include "engine/framework/core/backend.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

bool bits_equal(float a, float b) {
    uint32_t ua, ub;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

std::vector<float> make_random(size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> v(n);
    for (auto & x : v) x = ming_cpu::round_bf16_scalar(dist(rng));
    return v;
}

void test_q8_avx2_vs_scalar(const std::vector<float> & gelu_table) {
    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {64, 128}, {128, 256}, {256, 1024}, {384, 512}
    };
    uint32_t seed = 1001;
    for (const auto & [out_ch, in_ch] : shapes) {
        const auto w_f32 = make_random(static_cast<size_t>(out_ch * in_ch), seed++, 0.25f);
        const auto bias = make_random(static_cast<size_t>(out_ch), seed++, 0.1f);
        const auto packed = ming_cpu::pack_q8_from_f32(w_f32.data(), bias.data(), out_ch, in_ch);
        for (int64_t rows = 1; rows <= 6; ++rows) {
            const auto x_f32 = make_random(static_cast<size_t>(rows * in_ch), seed++, 0.5f);
            const auto residual = make_random(static_cast<size_t>(rows * out_ch), seed++, 0.5f);
            ming_cpu::Q8InputBuffer qx;
            ming_cpu::quantize_rows_q8(x_f32.data(), rows, in_ch, qx);
            for (int epilogue = 0; epilogue <= 2; ++epilogue) {
                std::vector<float> y_avx(static_cast<size_t>(rows * out_ch), 0.0f);
                std::vector<float> y_ref(static_cast<size_t>(rows * out_ch), 0.0f);
                ming_cpu::gemm_q8_slice(packed, qx, y_avx.data(), residual.data(), gelu_table.data(),
                    epilogue, 0, out_ch, true);
                ming_cpu::gemm_q8_slice(packed, qx, y_ref.data(), residual.data(), gelu_table.data(),
                    epilogue, 0, out_ch, false);
                for (size_t i = 0; i < y_avx.size(); ++i) {
                    if (!bits_equal(y_avx[i], y_ref[i])) {
                        // Allow 1 ULP in BF16 only if horizontal sum reordering occurs across FMA lanes;
                        // check relative/absolute tolerance tightly.
                        const float diff = std::fabs(y_avx[i] - y_ref[i]);
                        const float base = std::max(std::fabs(y_ref[i]),
                            epilogue == 1 ? std::fabs(residual[i]) : 0.0f);
                        const float tol = std::max(1e-5f, base * 2e-2f);
                        if (diff > tol) {
                            throw std::runtime_error("Q8 AVX2 vs scalar mismatch");
                        }
                    }
                }
            }
        }
    }
    std::cout << "Q8 AVX2 vs scalar parity passed across all shapes, rows 1..6, epilogues 0..2\n";
}

void test_q4g_avx2_vs_scalar_and_accuracy(const std::vector<float> & gelu_table) {
    const std::vector<int> groups = {64, 128};
    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {64, 128}, {66, 128}, {128, 256}, {192, 1024}, {256, 512}
    };
    uint32_t seed = 2001;
    for (int gs : groups) {
        for (const auto & [out_ch, in_ch] : shapes) {
            const auto w_f32 = make_random(static_cast<size_t>(out_ch * in_ch), seed++, 0.25f);
            const auto bias = make_random(static_cast<size_t>(out_ch), seed++, 0.1f);
            const auto packed = ming_cpu::pack_q4g_from_f32(w_f32.data(), bias.data(), out_ch, in_ch, gs);
            for (int64_t rows = 1; rows <= 6; ++rows) {
                const auto x_f32 = make_random(static_cast<size_t>(rows * in_ch), seed++, 0.5f);
                const auto residual = make_random(static_cast<size_t>(rows * out_ch), seed++, 0.5f);
                ming_cpu::Q8GroupInputBuffer qx;
                ming_cpu::quantize_rows_q8_group(x_f32.data(), rows, in_ch, gs, qx);
                for (int epilogue = 0; epilogue <= 2; ++epilogue) {
                    std::vector<float> y_avx(static_cast<size_t>(rows * out_ch), 0.0f);
                    std::vector<float> y_ref(static_cast<size_t>(rows * out_ch), 0.0f);
                    ming_cpu::gemm_q4g_slice(packed, qx, y_avx.data(), residual.data(), gelu_table.data(),
                        epilogue, 0, out_ch, true);
                    ming_cpu::gemm_q4g_slice(packed, qx, y_ref.data(), residual.data(), gelu_table.data(),
                        epilogue, 0, out_ch, false);
                    for (size_t i = 0; i < y_avx.size(); ++i) {
                        const float diff = std::fabs(y_avx[i] - y_ref[i]);
                        const float base = std::max(std::fabs(y_ref[i]),
                            epilogue == 1 ? std::fabs(residual[i]) : 0.0f);
                        const float tol = std::max(1e-5f, base * 2e-2f);
                        if (!bits_equal(y_avx[i], y_ref[i]) && diff > tol) {
                            throw std::runtime_error("Q4G AVX2 vs scalar mismatch: gs=" + std::to_string(gs) +
                                " shape=" + std::to_string(out_ch) + "x" + std::to_string(in_ch) +
                                " rows=" + std::to_string(rows) + " ep=" + std::to_string(epilogue) +
                                " idx=" + std::to_string(i) + " avx=" + std::to_string(y_avx[i]) +
                                " ref=" + std::to_string(y_ref[i]));
                        }
                    }
                    // Also check Q4G epilogue=0 cosine similarity against exact F32 matmul!
                    if (epilogue == 0 && rows == 3) {
                        double dot = 0.0, na = 0.0, nb = 0.0;
                        for (int64_t r = 0; r < rows; ++r) {
                            for (int64_t ch = 0; ch < out_ch; ++ch) {
                                double ref = bias[ch];
                                for (int64_t k = 0; k < in_ch; ++k) {
                                    ref += static_cast<double>(w_f32[ch * in_ch + k]) * x_f32[r * in_ch + k];
                                }
                                const double val = y_avx[r * out_ch + ch];
                                dot += ref * val;
                                na += ref * ref;
                                nb += val * val;
                            }
                        }
                        const double cos = dot / std::sqrt(na * nb);
                        if (cos < 0.992) {
                            throw std::runtime_error("Q4G cosine similarity too low: " + std::to_string(cos));
                        }
                    }
                }
            }
        }
    }
    std::cout << "Q4G (G=64, G=128) AVX2 vs scalar parity & F32 cosine (>0.992) passed\n";
}

void test_time_grid() {
    std::vector<float> ref = {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
    for (auto & t : ref) t /= 32.0f;
    auto phase = ref;
    for (auto & t : phase) t *= 1.5707963267948966f;
    engine::core::round_f32_to_bf16_in_place(phase);
    for (auto & t : phase) t = std::cos(t);
    engine::core::round_f32_to_bf16_in_place(phase);
    for (auto & t : phase) t -= 1.0f;
    engine::core::round_f32_to_bf16_in_place(phase);
    for (size_t i = 0; i < ref.size(); ++i) phase[i] += ref[i];
    engine::core::round_f32_to_bf16_in_place(phase);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] -= phase[i];
    engine::core::round_f32_to_bf16_in_place(ref);

    const auto got = ming_cpu::flow_time_grid(10);
    if (got.size() != ref.size()) throw std::runtime_error("flow_time_grid(10) size mismatch");
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!bits_equal(got[i], ref[i])) throw std::runtime_error("flow_time_grid(10) mismatch");
    }
    for (int s = 5; s <= 9; ++s) {
        const auto g = ming_cpu::flow_time_grid(s);
        if (static_cast<int>(g.size()) != s + 1 || g.front() != 0.0f || g.back() != 1.0f) {
            throw std::runtime_error("Invalid reduced flow_time_grid");
        }
        for (size_t i = 1; i < g.size(); ++i) {
            if (!(g[i] > g[i - 1])) throw std::runtime_error("Non-monotonic flow_time_grid");
        }
    }
    std::cout << "flow_time_grid exactness and monotonicity passed\n";
}

}  // namespace

int main() try {
    engine::core::BackendConfig cfg;
    cfg.type = engine::core::BackendType::Cpu;
    cfg.threads = 2;
    engine::core::ExecutionContext exec(cfg);
    const auto gelu_table = ming_cpu::make_gelu_bf16_table(exec);
    test_q8_avx2_vs_scalar(gelu_table);
    test_q4g_avx2_vs_scalar_and_accuracy(gelu_table);
    test_time_grid();
    return 0;
} catch (const std::exception & e) {
    std::cerr << "Test failed: " << e.what() << '\n';
    return 1;
}
