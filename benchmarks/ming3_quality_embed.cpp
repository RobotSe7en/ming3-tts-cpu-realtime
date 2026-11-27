#include "engine/models/ming3_tts/assets.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/reference_audio_frontend.h"
#include "engine/framework/modules/speaker_encoders/campplus_encoder.h"
#include "engine/framework/io/json.h"

#include <omp.h>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) try {
    if (argc != 3) throw std::runtime_error("usage: ming3_quality_embed MODEL WAV_PATH_LIST");
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    auto assets = engine::models::ming3_tts::load_ming3_tts_assets(argv[1]);
    engine::core::BackendConfig backend;
    backend.type = engine::core::BackendType::Cpu;
    backend.threads = 4;
    engine::modules::CampplusEncoderConfig config;
    config.weight_layout = engine::modules::CampplusEncoderWeightLayout::Fused;
    config.normalize_partial_segment_by_full_length = true;
    auto encoder = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
        assets->speaker_weights, backend, config);
    std::ifstream paths(argv[2]);
    if (!paths) throw std::runtime_error("Cannot read WAV list");
    std::cout << std::setprecision(9);
    std::string path;
    while (std::getline(paths, path)) {
        if (path.empty()) continue;
        const auto audio = engine::audio::read_wav_f32(std::filesystem::path(path));
        const auto waveform = engine::audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            audio.samples, audio.sample_rate, audio.channels, 16000);
        const auto features = engine::audio::ReferenceAudioFrontend::campplus_fbank_16k(waveform);
        const auto result = encoder.embed_from_features(features.values, features.frames, features.feature_dim);
        std::cout << "{\"path\":" << engine::io::json::stringify_string(path) << ",\"embedding\":[";
        for (size_t i = 0; i < result.embedding.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << result.embedding[i];
        }
        std::cout << "]}" << std::endl;
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
