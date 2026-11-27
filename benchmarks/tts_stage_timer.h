#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace tts_profile {

enum class Stage : size_t {
    Other, Frontend, ArPrefill, ArDecode, Projection, ProjectionGraph,
    Noise, Flow, FlowGraph, Vae, VaeGraph, Istft, VaeGraphBuild, ReferenceEncode,
    CfgDual, CfgSingle, Count
};

inline constexpr std::array<const char*, static_cast<size_t>(Stage::Count)> names{
    "session_other", "frontend", "ar_prefill", "ar_decode",
    "latent_projection_overhead", "latent_projection_graph", "random_noise",
    "flow_overhead", "flow_graph", "vae_overhead", "vae_graph", "istft",
    "vae_graph_build", "reference_encode", "cfg_dual", "cfg_single"
};

struct Measurement {
    int64_t inclusive_ns = 0;
    int64_t exclusive_ns = 0;
    uint64_t calls = 0;
};

class Scope;
inline thread_local Scope* active = nullptr;
inline thread_local std::array<Measurement, names.size()> measurements{};

// All scopes run on the calling synthesis thread; synchronous worker time is
// included in wall time. Child durations are subtracted to prevent double counting.
class Scope {
public:
    explicit Scope(Stage stage)
        : stage_(stage), parent_(active), started_(Clock::now()) {
        active = this;
    }
    ~Scope() {
        const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - started_).count();
        auto& result = measurements[static_cast<size_t>(stage_)];
        result.inclusive_ns += duration;
        result.exclusive_ns += duration - child_ns_;
        ++result.calls;
        active = parent_;
        if (parent_) parent_->child_ns_ += duration;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    using Clock = std::chrono::steady_clock;
    Stage stage_;
    Scope* parent_;
    Clock::time_point started_;
    int64_t child_ns_ = 0;
};

template <typename Function>
decltype(auto) measure(Stage stage, Function&& function) {
    Scope scope(stage);
    return std::forward<Function>(function)();
}

inline void reset() {
    if (active) throw std::logic_error("Cannot reset active stage measurements");
    measurements = {};
}

inline void write_json(std::ostream& output) {
    output << '{';
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) output << ',';
        const auto& result = measurements[i];
        output << '"' << names[i] << "\":{\"exclusive_s\":" << result.exclusive_ns / 1e9
               << ",\"inclusive_s\":" << result.inclusive_ns / 1e9
               << ",\"calls\":" << result.calls << '}';
    }
    output << '}';
}

}  // namespace tts_profile
