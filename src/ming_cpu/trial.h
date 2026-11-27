#pragma once

#include <cstdint>
#include <vector>

#ifdef TTS_MING_PROFILE
#include "tts_stage_timer.h"
#define MING_TRIAL_MEASURE(stage, expression) \
    tts_profile::measure(tts_profile::Stage::stage, [&] { return (expression); })
#define MING_TRIAL_SCOPE(stage, name) tts_profile::Scope name(tts_profile::Stage::stage)
#else
#define MING_TRIAL_MEASURE(stage, expression) (expression)
#define MING_TRIAL_SCOPE(stage, name)
#endif

namespace ming_cpu::trial {

struct Settings {
    int flow_steps = 10;
    int cfg_interval = 1;
    float cfg_reuse_threshold = 0.0f;
    int64_t vae_cache_bytes = 0;
    bool packed_q8 = false;
    bool preserve_noise_stream = false;
    bool cfg_reuse_delta = true;
};

const Settings & settings();
// Official EPSS nodes followed by Ming's original BF16 sway transform.
std::vector<float> time_grid(int steps);

}  // namespace ming_cpu::trial
