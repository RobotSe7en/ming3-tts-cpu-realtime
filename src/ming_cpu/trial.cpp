#include "trial.h"
#include "engine/framework/core/backend.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ming_cpu::trial {
namespace {

long integer(const char * name, long fallback, long minimum, long maximum) {
    const char * value = std::getenv(name);
    if (!value) return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed < minimum || parsed > maximum) {
        throw std::runtime_error(std::string("Invalid experiment setting: ") + name);
    }
    return parsed;
}

}  // namespace

const Settings & settings() {
    static const Settings value = [] {
        Settings s;
        s.flow_steps = integer("MING_FLOW_STEPS", 10, 5, 10);
        if (s.flow_steps != 5 && s.flow_steps != 6 && s.flow_steps != 7 && s.flow_steps != 10) {
            throw std::runtime_error("MING_FLOW_STEPS must be 5, 6, 7 or 10");
        }
        s.cfg_interval = integer("MING_CFG_INTERVAL", 1, 1, 10);
        s.vae_cache_bytes = integer("MING_VAE_CACHE_MIB", 0, 0, 1536) * int64_t{1024 * 1024};
        s.packed_q8 = integer("MING_PACKED_Q8", 0, 0, 1) != 0;
        s.preserve_noise_stream = integer("MING_PRESERVE_NOISE_STREAM", 0, 0, 1) != 0;
        if (const char * mode = std::getenv("MING_CFG_REUSE_MODE")) {
            const std::string name(mode);
            if (name != "delta" && name != "unconditional") throw std::runtime_error("Invalid MING_CFG_REUSE_MODE");
            s.cfg_reuse_delta = name == "delta";
        }
        if (const char * threshold = std::getenv("MING_CFG_REUSE_THRESHOLD")) {
            char * end = nullptr;
            s.cfg_reuse_threshold = std::strtof(threshold, &end);
            if (*threshold == '\0' || *end != '\0' || !std::isfinite(s.cfg_reuse_threshold) ||
                s.cfg_reuse_threshold < 0.0f || s.cfg_reuse_threshold > 2.0f) {
                throw std::runtime_error("Invalid MING_CFG_REUSE_THRESHOLD");
            }
        }
        return s;
    }();
    return value;
}

std::vector<float> time_grid(int steps) {
    std::vector<float> time;
    switch (steps) {
        case 5: time = {0, 2, 4, 8, 16, 32}; break;
        case 6: time = {0, 2, 4, 6, 8, 16, 32}; break;
        case 7: time = {0, 2, 4, 6, 8, 16, 24, 32}; break;
        case 10: time = {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32}; break;
        default: throw std::runtime_error("Unsupported EPSS step count");
    }
    for (auto & t : time) t /= 32.0f;
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

}  // namespace ming_cpu::trial
