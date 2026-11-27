cmake_minimum_required(VERSION 3.20)
project(Ming3CpuRealtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

set(AUDIOCPP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.cache/audio.cpp"
    CACHE PATH "Unmodified audio.cpp checkout")
set(AUDIOCPP_MODEL_SET custom CACHE STRING "audio.cpp model selection")
set(AUDIOCPP_MODELS ming3_tts CACHE STRING "Models to compile")
set(ENGINE_ENABLE_CUDA OFF CACHE BOOL "CUDA backend")
set(ENGINE_ENABLE_VULKAN OFF CACHE BOOL "Vulkan backend")
set(ENGINE_ENABLE_METAL OFF CACHE BOOL "Metal backend")
set(ENGINE_BUILD_WARMBENCH OFF CACHE BOOL "Upstream benchmark targets")

find_package(OpenMP REQUIRED)
add_subdirectory("${AUDIOCPP_SOURCE_DIR}" audiocpp EXCLUDE_FROM_ALL)

# Unmodified upstream baseline harness (for ABBA benchmarking)
add_executable(ming3_rtf ming3_rtf.cpp)
target_link_libraries(ming3_rtf PRIVATE engine_runtime OpenMP::OpenMP_CXX)

option(TTS_BUILD_MING_FAST "Build the direct persistent C++ Ming CPU engine, server and parity tests" ON)
option(TTS_BUILD_MING_PROFILE "Build the stage-instrumented fast Ming benchmark" ON)
set(AUDIOCPP_MING_FAST_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.cache/audio.cpp-ming-fast"
    CACHE PATH "audio.cpp worktree with src/patches/ming_fast_cpu.patch applied")

if(TTS_BUILD_MING_FAST)
    add_library(ming_cpu_kernels STATIC "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu/kernels.cpp")
    target_include_directories(ming_cpu_kernels PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu")
    target_link_libraries(ming_cpu_kernels PUBLIC engine_runtime OpenMP::OpenMP_CXX)
    if(GGML_LLAMAFILE)
        target_include_directories(ming_cpu_kernels PRIVATE "${AUDIOCPP_SOURCE_DIR}/external/ggml/src")
        target_compile_definitions(ming_cpu_kernels PRIVATE MING_USE_GGML_GEMM=1)
    endif()

    add_library(ming_fast_support STATIC
        "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu/flow_engine.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../src/ming_cpu/repack_weights.cpp")
    target_include_directories(ming_fast_support PRIVATE
        "${AUDIOCPP_SOURCE_DIR}/external/ggml/src")
    target_link_libraries(ming_fast_support PUBLIC ming_cpu_kernels)

    # Supply patched objects before static upstream archives so the fast engine
    # overrides the corresponding Ming runtime units without modifying .cache/audio.cpp.
    add_library(ming3_fast_objects OBJECT
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/models/ming3_tts/generator.cpp"
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/models/ming3_tts/session.cpp"
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/framework/modules/speech_decoders/ming_flow_runtime.cpp"
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/framework/codecs/ming_audio_vae_runtime.cpp"
        "${AUDIOCPP_MING_FAST_SOURCE_DIR}/src/framework/audio/fft.cpp")
    target_compile_definitions(ming3_fast_objects PRIVATE ENGINE_SPEECHFFT_MAX_THREADS=4)
    target_link_libraries(ming3_fast_objects PRIVATE ming_fast_support)

    # Fast CLI benchmark harness
    add_executable(ming3_fast_rtf ming3_rtf.cpp $<TARGET_OBJECTS:ming3_fast_objects>)
    target_link_libraries(ming3_fast_rtf PRIVATE ming_fast_support)

    # Native C++ HTTP/OpenAI server powered by the fast CPU engine + stateful streaming VAE
    file(GLOB AUDIOCPP_FAST_RECIPE_PLUGINS CONFIGURE_DEPENDS
        "${AUDIOCPP_SOURCE_DIR}/app/recipe/plugins/*.cpp")
    add_executable(ming3_fast_server
        $<TARGET_OBJECTS:ming3_fast_objects>
        "${AUDIOCPP_SOURCE_DIR}/app/common/base64.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/common/build_info.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/main.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/config.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/frontend.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/http.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/parallel_http.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/model_memory.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/multipart.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/server_protocol.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/server_ui.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/runtime.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/parallel_runtime.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/server/ui_assets.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/cli/args.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/cli/request.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/streaming/pcm_source.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/streaming/streaming.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/recipe/streaming_tempo.cpp"
        "${AUDIOCPP_SOURCE_DIR}/app/recipe/recipe.cpp"
        ${AUDIOCPP_FAST_RECIPE_PLUGINS})
    target_include_directories(ming3_fast_server PRIVATE
        "${CMAKE_BINARY_DIR}/audiocpp/generated")
    target_link_libraries(ming3_fast_server PRIVATE ming_fast_support ggml OpenMP::OpenMP_CXX)

    # Parity tests
    enable_testing()
    add_executable(ming3_cpu_linear_test "${CMAKE_CURRENT_LIST_DIR}/../tests/ming_cpu_linear_test.cpp")
    target_link_libraries(ming3_cpu_linear_test PRIVATE ming_cpu_kernels)
    add_test(NAME ming3_cpu_linear_parity COMMAND ming3_cpu_linear_test)

    add_executable(ming3_fast_engine_test "${CMAKE_CURRENT_LIST_DIR}/../tests/ming_fast_engine_test.cpp")
    target_link_libraries(ming3_fast_engine_test PRIVATE ming_fast_support)
    add_test(NAME ming3_fast_engine_parity COMMAND ming3_fast_engine_test)

    add_executable(ming3_stream_vae_test
        "${CMAKE_CURRENT_LIST_DIR}/../tests/ming_stream_vae_test.cpp"
        $<TARGET_OBJECTS:ming3_fast_objects>)
    target_link_libraries(ming3_stream_vae_test PRIVATE ming_fast_support)
    add_test(NAME ming3_stream_vae_parity COMMAND ming3_stream_vae_test
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/..")

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
