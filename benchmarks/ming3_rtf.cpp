#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#ifdef TTS_MING_PROFILE
#include "tts_stage_timer.h"
#endif

#include <omp.h>
#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double cpu_seconds() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 +
        usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 7 && argc != 10) {
        throw std::runtime_error(
            "usage: ming3_rtf MODEL THREADS TEXT_FILE OUTPUT_DIR WARMUPS ITERATIONS "
            "[qwen3_tts REFERENCE_WAV REFERENCE_TEXT_FILE]");
    }
    const std::string family = argc == 10 ? argv[7] : "ming3_tts";
    if (family != "ming3_tts" && family != "qwen3_tts") {
        throw std::runtime_error("Unsupported model family");
    }
    const int threads = std::stoi(argv[2]);
    const int warmups = std::stoi(argv[5]);
    const int iterations = std::stoi(argv[6]);
    if (threads < 1 || warmups < 1 || iterations < 1) {
        throw std::runtime_error("threads, warmups and iterations must be positive");
    }
    std::ifstream input(argv[3]);
    if (!input) {
        throw std::runtime_error("Cannot read benchmark text");
    }
    std::string text((std::istreambuf_iterator<char>(input)), {});
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    if (text.empty()) {
        throw std::runtime_error("Benchmark text is empty");
    }
    const std::filesystem::path output_dir(argv[4]);
    std::filesystem::create_directories(output_dir);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "{\"event\":\"loading\",\"threads\":" << threads << "}" << std::endl;

    const auto load_started = Clock::now();
    auto registry = engine::runtime::make_default_registry();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = argv[1];
    load_request.family_hint = family;
    auto model = registry.load(load_request);
    engine::runtime::SessionOptions session_options;
    session_options.backend.type = engine::core::BackendType::Cpu;
    session_options.backend.threads = threads;
    engine::runtime::TaskSpec task;
    task.task = engine::runtime::VoiceTaskKind::Tts;
    task.mode = engine::runtime::RunMode::Offline;
    auto session_base = model->create_task_session(task, session_options);
    auto* session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession*>(session_base.get());
    if (!session) {
        throw std::runtime_error("Expected an offline TTS session");
    }
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{text, "zh"};
    request.options = {{"language", "zh"}, {"seed", "1895"}};
    if (const char* tmpl = std::getenv("MING_TEMPLATE_NAME"); tmpl && *tmpl) {
        request.options["template_name"] = tmpl;
    }
    if (const char* inst = std::getenv("MING_INSTRUCT"); inst && *inst) {
        request.options["instruct"] = inst;
    }
    if (const char* vid = std::getenv("MING_VOICE_ID"); vid && *vid) {
        request.options["voice_id"] = vid;
    }
    if (family == "qwen3_tts") {
        const auto reference = engine::audio::read_wav_f32(std::filesystem::path(argv[8]));
        std::ifstream transcript_input(argv[9]);
        if (!transcript_input) throw std::runtime_error("Cannot read reference transcript");
        std::string transcript((std::istreambuf_iterator<char>(transcript_input)), {});
        while (!transcript.empty() && (transcript.back() == '\n' || transcript.back() == '\r')) {
            transcript.pop_back();
        }
        if (transcript.empty()) throw std::runtime_error("Reference transcript is empty");
        request.options["reference_text"] = transcript;
        request.voice = engine::runtime::VoiceCondition{};
        request.voice->speaker = engine::runtime::VoiceReference{};
        request.voice->speaker->audio = engine::runtime::AudioBuffer{
            reference.sample_rate, reference.channels, reference.samples};
    }
    engine::runtime::SessionPreparationRequest preparation;
    preparation.text = request.text_input;
    preparation.voice = request.voice;
    preparation.options = request.options;
    session->prepare(preparation);
    std::cout << "{\"event\":\"loaded\",\"load_s\":" << elapsed(load_started)
              << "}" << std::endl;

    for (int index = 0; index < warmups + iterations; ++index) {
        const bool warmup = index < warmups;
        const int iteration = warmup ? index + 1 : index - warmups + 1;
        const char* phase = warmup ? "warmup" : "measured";
        std::cerr << "Starting " << phase << " " << iteration << std::endl;
#ifdef TTS_MING_PROFILE
        tts_profile::reset();
#endif
        const double cpu_started = cpu_seconds();
        const auto started = Clock::now();
        auto result = session->run(request);
        const double synthesis_s = elapsed(started);
        const double used_cpu_s = cpu_seconds() - cpu_started;
        if (!result.audio_output) {
            throw std::runtime_error("Model returned no audio");
        }
        const auto& audio = *result.audio_output;
        if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty() ||
            audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
            throw std::runtime_error("Model returned invalid audio dimensions");
        }
        double sum_squared = 0.0;
        for (float sample : audio.samples) {
            if (!std::isfinite(sample)) {
                throw std::runtime_error("Model returned nonfinite audio");
            }
            sum_squared += static_cast<double>(sample) * sample;
        }
        if (sum_squared == 0.0) {
            throw std::runtime_error("Model returned entirely silent audio");
        }
        const double audio_s = static_cast<double>(audio.samples.size()) /
            audio.channels / audio.sample_rate;
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::cout << "{\"event\":\"synthesis\",\"phase\":\"" << phase
                  << "\",\"iteration\":" << iteration
                  << ",\"synthesis_s\":" << synthesis_s
                  << ",\"audio_s\":" << audio_s
                  << ",\"rtf\":" << synthesis_s / audio_s
                  << ",\"sample_rate\":" << audio.sample_rate
                  << ",\"channels\":" << audio.channels
                  << ",\"sample_values\":" << audio.samples.size()
                  << ",\"cpu_percent_one_core_100\":" << used_cpu_s / synthesis_s * 100.0
                  << ",\"peak_rss_kib\":" << usage.ru_maxrss;
#ifdef TTS_MING_PROFILE
        std::cout << ",\"stages\":";
        tts_profile::write_json(std::cout);
#endif
        std::cout << "}" << std::endl;
        if (!warmup) {
            engine::audio::write_pcm16_wav(
                output_dir / ("run_" + std::to_string(iteration) + ".wav"),
                audio.sample_rate, audio.channels, audio.samples);
        }
    }
    return 0;
} catch (const std::exception& exc) {
    std::cerr << "Benchmark failed: " << exc.what() << std::endl;
    return 1;
}
