#include "engine/framework/codecs/ming_audio_vae_runtime.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/streaming_audio.h"
#include "engine/models/ming3_tts/assets.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct ParityStats {
    double max_abs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
};

ParityStats compare_audio(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Audio length mismatch: " + std::to_string(a.size()) + " vs " + std::to_string(b.size()));
    }
    double max_abs = 0.0;
    double sum_sq_err = 0.0;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double va = a[i];
        const double vb = b[i];
        const double diff = std::abs(va - vb);
        if (diff > max_abs) max_abs = diff;
        sum_sq_err += diff * diff;
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    ParityStats s;
    s.max_abs = max_abs;
    s.rmse = std::sqrt(sum_sq_err / std::max<size_t>(1, a.size()));
    s.cosine = (norm_a > 0.0 && norm_b > 0.0) ? dot / (std::sqrt(norm_a) * std::sqrt(norm_b)) : 1.0;
    return s;
}

std::vector<float> run_streaming(
    engine::modules::MingAudioDecoder & decoder,
    const std::vector<float> & latents,
    size_t latent_dim,
    size_t frames_per_chunk,
    engine::runtime::StreamingAudioChunkPolicy policy,
    size_t & event_count,
    double & wall_ms) {
    engine::runtime::StreamingAudioConfig cfg;
    cfg.policy = policy;
    cfg.frames_per_chunk = frames_per_chunk;
    cfg.frame_width = latent_dim;
    cfg.left_context_frames = 1;
    cfg.right_context_frames = 1;

    engine::runtime::StreamingAudioController<float> controller;
    event_count = 0;
    std::vector<float> streamed_via_events;
    const auto start = Clock::now();
    controller.begin(cfg,
        [&](const auto & window) {
            return decoder.decode_streaming_window(window, cfg);
        },
        [&](const engine::runtime::StreamEvent & ev) {
            ++event_count;
            if (ev.audio_output) {
                streamed_via_events.insert(streamed_via_events.end(),
                    ev.audio_output->samples.begin(), ev.audio_output->samples.end());
            }
        });
    const size_t patches = latents.size() / latent_dim;
    std::vector<float> patch(latent_dim);
    for (size_t p = 0; p < patches; ++p) {
        std::copy_n(latents.begin() + static_cast<std::ptrdiff_t>(p * latent_dim), latent_dim, patch.begin());
        controller.push(patch);
    }
    const auto final_buf = controller.finish();
    wall_ms = elapsed_ms(start);
    if (final_buf.samples != streamed_via_events) {
        throw std::runtime_error("Streamed event concatenation differs from StreamingAudioController::finish()");
    }
    return final_buf.samples;
}

}  // namespace

int main(int argc, char ** argv) {
    std::filesystem::path model_path;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        for (const auto & candidate : {
                 std::filesystem::path("models/ming-3.0-tts-0.5b-q8_0.gguf"),
                 std::filesystem::path("../models/ming-3.0-tts-0.5b-q8_0.gguf"),
                 std::filesystem::path("../../models/ming-3.0-tts-0.5b-q8_0.gguf")}) {
            if (std::filesystem::is_regular_file(candidate)) {
                model_path = candidate;
                break;
            }
        }
    }
    if (model_path.empty() || !std::filesystem::is_regular_file(model_path)) {
        std::cout << "SKIP: Ming-3.0-TTS GGUF not found; skipping live streaming VAE parity test\n";
        return 0;
    }

    auto assets = engine::models::ming3_tts::load_ming3_tts_assets(model_path);
    engine::core::BackendConfig exec_cfg;
    exec_cfg.type = engine::core::BackendType::Cpu;
    exec_cfg.threads = 4;
    engine::core::ExecutionContext exec(exec_cfg);

    engine::modules::MingAudioDecoder decoder(assets->model_weights, assets->config.codec, exec);
    const size_t latent_dim = static_cast<size_t>(assets->config.codec.decoder.latent_dim);
    // 64 patches = 256 STFT frames = 2 full 128-slot ring-buffer wraparounds (5.12s audio)
    const size_t patches = 64;
    std::mt19937 rng(2026);
    std::normal_distribution<float> dist(0.0f, 1.5f);
    std::vector<float> latents(patches * latent_dim);
    for (float & v : latents) v = dist(rng);

    // Warmup + timed offline decode
    (void) decoder.decode(latents);
    const auto t0 = Clock::now();
    const auto offline = decoder.decode(latents);
    const double offline_ms = elapsed_ms(t0);

    // Warmup + timed streaming decode (grow policy, frames_per_chunk=8)
    size_t events_grow = 0;
    double ms_grow = 0.0;
    (void) run_streaming(decoder, latents, latent_dim, 8,
        engine::runtime::StreamingAudioChunkPolicy::Grow, events_grow, ms_grow);
    const auto stream_grow = run_streaming(decoder, latents, latent_dim, 8,
        engine::runtime::StreamingAudioChunkPolicy::Grow, events_grow, ms_grow);
    const auto stats_grow = compare_audio(offline, stream_grow);

    // Second streaming pass with odd fixed chunk size = 3 (tests stream reset + odd wraparound)
    size_t events_odd = 0;
    double ms_odd = 0.0;
    const auto stream_odd = run_streaming(decoder, latents, latent_dim, 3,
        engine::runtime::StreamingAudioChunkPolicy::Fixed, events_odd, ms_odd);
    const auto stats_odd = compare_audio(offline, stream_odd);

    const auto stats_grow_vs_odd = compare_audio(stream_grow, stream_odd);
    for (int q = 0; q < 4; ++q) {
        size_t s0 = q * (offline.size() / 4);
        size_t s1 = (q + 1) * (offline.size() / 4);
        double qmax = 0.0;
        for (size_t i = s0; i < s1; ++i) qmax = std::max(qmax, std::abs(double(offline[i]) - double(stream_grow[i])));
        std::cout << "  Quarter " << q << " (frames " << (s0/882) << ".." << (s1/882) << "): max_abs=" << qmax << "\n";
    }
    size_t max_idx = 0;
    double max_d = 0.0;
    double sig_rms = 0.0;
    for (size_t i = 0; i < offline.size(); ++i) {
        sig_rms += double(offline[i]) * offline[i];
        double d = std::abs(double(offline[i]) - double(stream_grow[i]));
        if (d > max_d) { max_d = d; max_idx = i; }
    }
    sig_rms = std::sqrt(sig_rms / offline.size());
    std::cout << "Offline decode (" << patches << " patches, " << offline.size() << " samples, rms=" << sig_rms << "): "
              << offline_ms << " ms\n";
    std::cout << "Stream grow(8) (" << events_grow << " events): " << ms_grow << " ms"
              << " | max_abs=" << stats_grow.max_abs << " (at sample " << max_idx << ", frame " << (max_idx / 882) << ")"
              << " rmse=" << stats_grow.rmse
              << " cosine=" << stats_grow.cosine << "\n";
    std::cout << "Stream fixed(3) (" << events_odd << " events): " << ms_odd << " ms"
              << " | max_abs=" << stats_odd.max_abs
              << " rmse=" << stats_odd.rmse
              << " cosine=" << stats_odd.cosine << "\n";
    std::cout << "Stream grow(8) vs fixed(3): max_abs=" << stats_grow_vs_odd.max_abs
              << " rmse=" << stats_grow_vs_odd.rmse << " cosine=" << stats_grow_vs_odd.cosine << "\n";

    if (stats_grow_vs_odd.max_abs > 1e-6 || stats_grow_vs_odd.rmse > 1e-7) {
        std::cerr << "FAIL: grow(8) and fixed(3) streaming schedules are not invariant!\n";
        return 1;
    }
    if (stats_grow.cosine < 0.999 || stats_grow.rmse > 3e-3) {
        std::cerr << "FAIL: grow(8) streaming VAE diverged from offline decode!\n";
        return 1;
    }
    if (stats_odd.cosine < 0.999 || stats_odd.rmse > 3e-3) {
        std::cerr << "FAIL: fixed(3) streaming VAE diverged from offline decode!\n";
        return 1;
    }
    std::cout << "PASS: Stateful sliding-window KV-cache VAE matches offline decode across 2 ring wraps!\n";
    return 0;
}
