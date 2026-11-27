cmake_minimum_required(VERSION 3.20)
project(TtsCpuBenchmarks LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(AUDIOCPP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.cache/audio.cpp"
    CACHE PATH "Unmodified audio.cpp checkout")
set(AUDIOCPP_MODEL_SET custom CACHE STRING "audio.cpp model selection")
set(AUDIOCPP_MODELS ming3_tts,qwen3_tts CACHE STRING "Models to compile")
set(ENGINE_ENABLE_CUDA OFF CACHE BOOL "CUDA backend")
set(ENGINE_ENABLE_VULKAN OFF CACHE BOOL "Vulkan backend")
set(ENGINE_ENABLE_METAL OFF CACHE BOOL "Metal backend")
set(ENGINE_BUILD_WARMBENCH OFF CACHE BOOL "Upstream benchmark targets")
find_package(OpenMP REQUIRED)
add_subdirectory("${AUDIOCPP_SOURCE_DIR}" audiocpp EXCLUDE_FROM_ALL)

add_executable(ming3_rtf ming3_rtf.cpp)
target_link_libraries(ming3_rtf PRIVATE engine_runtime OpenMP::OpenMP_CXX)

option(TTS_BUILD_MING_PROFILE "Build the instrumented Ming stage benchmark" OFF)
set(AUDIOCPP_PROFILE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.cache/audio.cpp-profile"
    CACHE PATH "audio.cpp worktree with the stage timing patch applied")
if(TTS_BUILD_MING_PROFILE)
    # Supply complete instrumented objects before the static upstream archive.
    # The archive's corresponding original objects are then not extracted.
    add_library(ming3_stage_objects OBJECT
        "${AUDIOCPP_PROFILE_SOURCE_DIR}/src/models/ming3_tts/generator.cpp"
        "${AUDIOCPP_PROFILE_SOURCE_DIR}/src/models/ming3_tts/session.cpp"
        "${AUDIOCPP_PROFILE_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp"
        "${AUDIOCPP_PROFILE_SOURCE_DIR}/src/framework/codecs/ming_audio_vae_runtime.cpp")
    target_include_directories(ming3_stage_objects PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(ming3_stage_objects PRIVATE engine_runtime OpenMP::OpenMP_CXX)
    add_executable(ming3_profile_rtf ming3_rtf.cpp $<TARGET_OBJECTS:ming3_stage_objects>)
    target_compile_definitions(ming3_profile_rtf PRIVATE TTS_MING_PROFILE=1)
    target_link_libraries(ming3_profile_rtf PRIVATE engine_runtime OpenMP::OpenMP_CXX)
    option(TTS_BUILD_MING_OP_PROFILE "Include GGML CPU operator timings in the profile binary" OFF)
    if(TTS_BUILD_MING_OP_PROFILE)
        add_library(ming3_op_profile_core OBJECT
            "${AUDIOCPP_PROFILE_SOURCE_DIR}/external/ggml/src/ggml-cpu/ggml-cpu.c")
        foreach(property INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS)
            get_target_property(cpu_${property} ggml-cpu ${property})
            if(cpu_${property})
                set_property(TARGET ming3_op_profile_core PROPERTY ${property} "${cpu_${property}}")
            endif()
        endforeach()
        target_link_libraries(ming3_op_profile_core PRIVATE ggml-base OpenMP::OpenMP_C)
        target_compile_definitions(ming3_op_profile_core PRIVATE
            _GNU_SOURCE _XOPEN_SOURCE=600 GGML_USE_OPENMP GGML_USE_LLAMAFILE
            GGML_USE_CPU_REPACK GGML_SCHED_MAX_COPIES=4)
        target_compile_options(ming3_op_profile_core PRIVATE -march=native)
        target_sources(ming3_profile_rtf PRIVATE $<TARGET_OBJECTS:ming3_op_profile_core>)
    endif()
endif()

option(TTS_BUILD_MING_ENGINE "Build the Ming specialized CPU engine" OFF)
set(AUDIOCPP_MING_ENGINE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.cache/audio.cpp-ming-engine"
    CACHE PATH "audio.cpp worktree with the specialized Ming engine patch")
if(TTS_BUILD_MING_ENGINE)
    add_library(ming_cpu_kernels STATIC "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu/kernels.cpp")
    target_include_directories(ming_cpu_kernels PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu")
    target_link_libraries(ming_cpu_kernels PUBLIC engine_runtime OpenMP::OpenMP_CXX)
    if(GGML_LLAMAFILE)
        target_include_directories(ming_cpu_kernels PRIVATE "${AUDIOCPP_SOURCE_DIR}/external/ggml/src")
        target_compile_definitions(ming_cpu_kernels PRIVATE MING_USE_GGML_GEMM=1)
    endif()
    add_library(ming3_engine_objects OBJECT
        "${AUDIOCPP_MING_ENGINE_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp"
        "${AUDIOCPP_MING_ENGINE_SOURCE_DIR}/src/framework/audio/fft.cpp")
    target_compile_definitions(ming3_engine_objects PRIVATE ENGINE_SPEECHFFT_MAX_THREADS=4)
    target_link_libraries(ming3_engine_objects PRIVATE ming_cpu_kernels)
    add_executable(ming3_engine_rtf ming3_rtf.cpp $<TARGET_OBJECTS:ming3_engine_objects>)
    target_link_libraries(ming3_engine_rtf PRIVATE ming_cpu_kernels)
    add_executable(ming3_cpu_linear_test "${CMAKE_CURRENT_LIST_DIR}/../tests/ming_cpu_linear_test.cpp")
    target_link_libraries(ming3_cpu_linear_test PRIVATE ming_cpu_kernels)
    enable_testing()
    add_test(NAME ming3_cpu_linear_parity COMMAND ming3_cpu_linear_test)
    if(TTS_BUILD_MING_PROFILE)
        add_library(ming3_engine_profile_objects OBJECT
            "${AUDIOCPP_PROFILE_SOURCE_DIR}/src/models/ming3_tts/generator.cpp"
            "${AUDIOCPP_PROFILE_SOURCE_DIR}/src/models/ming3_tts/session.cpp"
            "${AUDIOCPP_PROFILE_SOURCE_DIR}/src/framework/codecs/ming_audio_vae_runtime.cpp"
            "${AUDIOCPP_MING_ENGINE_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp"
            "${AUDIOCPP_MING_ENGINE_SOURCE_DIR}/src/framework/audio/fft.cpp")
        target_include_directories(ming3_engine_profile_objects PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
        target_compile_definitions(ming3_engine_profile_objects PRIVATE TTS_MING_PROFILE=1 ENGINE_SPEECHFFT_MAX_THREADS=4)
        target_link_libraries(ming3_engine_profile_objects PRIVATE ming_cpu_kernels)
        add_executable(ming3_engine_profile_rtf ming3_rtf.cpp $<TARGET_OBJECTS:ming3_engine_profile_objects>)
        target_compile_definitions(ming3_engine_profile_rtf PRIVATE TTS_MING_PROFILE=1)
        target_link_libraries(ming3_engine_profile_rtf PRIVATE ming_cpu_kernels)
    endif()
endif()

option(TTS_BUILD_MING_CPU_OPT "Build the optimized CPU Ming benchmark" OFF)
set(AUDIOCPP_MING_OPT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.cache/audio.cpp-ming-opt"
    CACHE PATH "audio.cpp worktree with the CPU Ming optimization patch")
if(TTS_BUILD_MING_CPU_OPT)
    add_library(ming3_cpu_opt_objects OBJECT
        "${AUDIOCPP_MING_OPT_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp")
    target_link_libraries(ming3_cpu_opt_objects PRIVATE engine_runtime OpenMP::OpenMP_CXX)
    add_executable(ming3_cpu_opt_rtf ming3_rtf.cpp $<TARGET_OBJECTS:ming3_cpu_opt_objects>)
    target_link_libraries(ming3_cpu_opt_rtf PRIVATE engine_runtime OpenMP::OpenMP_CXX)
    add_executable(ming3_cpu_rope_test "${CMAKE_CURRENT_LIST_DIR}/../tests/ming_cpu_rope_test.cpp")
    target_compile_definitions(ming3_cpu_rope_test PRIVATE
        TTS_MING_OPT_SOURCE="${AUDIOCPP_MING_OPT_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp")
    target_link_libraries(ming3_cpu_rope_test PRIVATE engine_runtime OpenMP::OpenMP_CXX)
    enable_testing()
    add_test(NAME ming3_cpu_rope_parity COMMAND ming3_cpu_rope_test)
endif()

option(TTS_BUILD_MING_TUNED "Build experimental Ming sampling and caching paths" OFF)
set(AUDIOCPP_MING_TUNE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.cache/audio.cpp-ming-tune"
    CACHE PATH "audio.cpp worktree with the Ming tuning patch")
if(TTS_BUILD_MING_TUNED)
    if(NOT TTS_BUILD_MING_ENGINE)
        message(FATAL_ERROR "TTS_BUILD_MING_TUNED requires TTS_BUILD_MING_ENGINE")
    endif()
    add_library(ming_trial_support STATIC
        "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu/trial.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu/packed_q8.cpp")
    target_link_libraries(ming_trial_support PUBLIC ming_cpu_kernels)
    add_library(ming3_tuned_objects OBJECT
        "${AUDIOCPP_MING_TUNE_SOURCE_DIR}/src/models/ming3_tts/generator.cpp"
        "${AUDIOCPP_MING_TUNE_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp"
        "${AUDIOCPP_MING_TUNE_SOURCE_DIR}/src/framework/codecs/ming_audio_vae_runtime.cpp"
        "${AUDIOCPP_MING_TUNE_SOURCE_DIR}/src/framework/audio/fft.cpp")
    target_compile_definitions(ming3_tuned_objects PRIVATE ENGINE_SPEECHFFT_MAX_THREADS=4)
    target_link_libraries(ming3_tuned_objects PRIVATE ming_trial_support)
    add_executable(ming3_tuned_rtf ming3_rtf.cpp $<TARGET_OBJECTS:ming3_tuned_objects>)
    target_link_libraries(ming3_tuned_rtf PRIVATE ming_trial_support)
    add_executable(ming3_trial_test "${CMAKE_CURRENT_LIST_DIR}/../tests/ming_trial_test.cpp")
    target_link_libraries(ming3_trial_test PRIVATE ming_trial_support)
    enable_testing()
    add_test(NAME ming3_trial_schedules COMMAND ming3_trial_test)
    add_executable(ming3_packed_q8_test "${CMAKE_CURRENT_LIST_DIR}/../tests/ming_packed_q8_test.cpp")
    target_link_libraries(ming3_packed_q8_test PRIVATE ming_trial_support)
    add_test(NAME ming3_packed_q8_parity COMMAND ming3_packed_q8_test)
    add_executable(ming3_quality_embed ming3_quality_embed.cpp $<TARGET_OBJECTS:ming3_tuned_objects>)
    target_link_libraries(ming3_quality_embed PRIVATE ming_trial_support)
    if(TTS_BUILD_MING_PROFILE)
        add_library(ming3_tuned_profile_objects OBJECT
            "${AUDIOCPP_MING_TUNE_SOURCE_DIR}/src/models/ming3_tts/generator.cpp"
            "${AUDIOCPP_MING_TUNE_SOURCE_DIR}/src/models/ming3_tts/session.cpp"
            "${AUDIOCPP_MING_TUNE_SOURCE_DIR}/src/framework/codecs/ming_audio_vae_runtime.cpp"
            "${AUDIOCPP_MING_TUNE_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp"
            "${AUDIOCPP_MING_TUNE_SOURCE_DIR}/src/framework/audio/fft.cpp")
        target_include_directories(ming3_tuned_profile_objects PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
        target_compile_definitions(ming3_tuned_profile_objects PRIVATE TTS_MING_PROFILE=1 ENGINE_SPEECHFFT_MAX_THREADS=4)
        target_link_libraries(ming3_tuned_profile_objects PRIVATE ming_trial_support)
        add_executable(ming3_tuned_profile_rtf ming3_rtf.cpp $<TARGET_OBJECTS:ming3_tuned_profile_objects>)
        target_compile_definitions(ming3_tuned_profile_rtf PRIVATE TTS_MING_PROFILE=1)
        target_link_libraries(ming3_tuned_profile_rtf PRIVATE ming_trial_support)
    endif()
endif()

option(TTS_BUILD_MING_FAST "Build the direct persistent C++ Ming engine with load-time Q4 repack" OFF)
set(AUDIOCPP_MING_FAST_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.cache/audio.cpp-ming-fast"
    CACHE PATH "audio.cpp worktree with the fast Ming engine patch")
if(TTS_BUILD_MING_FAST)
    if(NOT TTS_BUILD_MING_ENGINE)
        message(FATAL_ERROR "TTS_BUILD_MING_FAST requires TTS_BUILD_MING_ENGINE")
    endif()
    add_library(ming_fast_support STATIC
        "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu/flow_engine.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu/repack_weights.cpp")
    target_include_directories(ming_fast_support PRIVATE
        "${AUDIOCPP_SOURCE_DIR}/external/ggml/src")
    target_link_libraries(ming_fast_support PUBLIC ming_cpu_kernels)
    add_library(ming3_fast_objects OBJECT
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/models/ming3_tts/generator.cpp"
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/models/ming3_tts/session.cpp"
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp"
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/framework/codecs/ming_audio_vae_runtime.cpp"
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/framework/audio/fft.cpp")
    target_compile_definitions(ming3_fast_objects PRIVATE ENGINE_SPEECHFFT_MAX_THREADS=4)
    target_link_libraries(ming3_fast_objects PRIVATE ming_fast_support)
    add_executable(ming3_fast_rtf ming3_rtf.cpp $<TARGET_OBJECTS:ming3_fast_objects>)
    target_link_libraries(ming3_fast_rtf PRIVATE ming_fast_support)
    add_executable(ming3_fast_engine_test "${CMAKE_CURRENT_LIST_DIR}/../tests/ming_fast_engine_test.cpp")
    target_link_libraries(ming3_fast_engine_test PRIVATE ming_fast_support)
    enable_testing()
    add_test(NAME ming3_fast_engine_parity COMMAND ming3_fast_engine_test)
    if(TTS_BUILD_MING_PROFILE)
        add_library(ming3_fast_profile_objects OBJECT
            "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/models/ming3_tts/generator.cpp"
            "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/models/ming3_tts/session.cpp"
            "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp"
            "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/framework/codecs/ming_audio_vae_runtime.cpp"
            "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/framework/audio/fft.cpp")
        target_include_directories(ming3_fast_profile_objects PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
        target_compile_definitions(ming3_fast_profile_objects PRIVATE TTS_MING_PROFILE=1 ENGINE_SPEECHFFT_MAX_THREADS=4)
        target_link_libraries(ming3_fast_profile_objects PRIVATE ming_fast_support)
        add_executable(ming3_fast_profile_rtf ming3_rtf.cpp $<TARGET_OBJECTS:ming3_fast_profile_objects>)
        target_compile_definitions(ming3_fast_profile_rtf PRIVATE TTS_MING_PROFILE=1)
        target_link_libraries(ming3_fast_profile_rtf PRIVATE ming_fast_support)
    endif()
endif()

