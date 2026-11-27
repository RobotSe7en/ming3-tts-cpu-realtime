#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AUDIOCPP_REPO="https://github.com/0xShug0/audio.cpp.git"
AUDIOCPP_COMMIT="7d20b414da088c3ffe56bd8803398d9d158cb847"
MODEL_URL="https://modelscope.cn/models/audio-cpp/Ming-3.0-TTS-0.5B-GGUF/resolve/master/ming-3.0-tts-0.5b-q8_0.gguf"
MODEL_PATH="models/ming-3.0-tts-0.5b-q8_0.gguf"
MODEL_SHA256="fcf2aa25e8e9cf2065a936e506581735c64d6255fb7869f09fbb61a2dde09aa1"
JOBS="${JOBS:-2}"
SKIP_MODEL=0

for arg in "$@"; do
    case "$arg" in
        --skip-model) SKIP_MODEL=1 ;;
        -h|--help)
            echo "Usage: $0 [--skip-model]"
            echo "Environment: JOBS=2 (default parallel build jobs)"
            exit 0
            ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "[1/4] Preparing pinned audio.cpp baseline ($AUDIOCPP_COMMIT)..."
mkdir -p .cache
if [ ! -d ".cache/audio.cpp/.git" ]; then
    git clone --depth 1 --filter=blob:none --sparse "$AUDIOCPP_REPO" .cache/audio.cpp
    git -C .cache/audio.cpp fetch --depth 1 origin "$AUDIOCPP_COMMIT"
    git -C .cache/audio.cpp checkout --detach "$AUDIOCPP_COMMIT"
    git -C .cache/audio.cpp sparse-checkout set app include src external model_specs tools webui/native/demo_voices
else
    echo "      Reusing existing .cache/audio.cpp ($(git -C .cache/audio.cpp rev-parse --short HEAD))"
fi

echo "[2/4] Preparing patched ming-fast worktree..."
if [ ! -d ".cache/audio.cpp-ming-fast" ]; then
    git -C .cache/audio.cpp worktree add --no-checkout --detach "$REPO_ROOT/.cache/audio.cpp-ming-fast" HEAD
    git -C .cache/audio.cpp-ming-fast sparse-checkout set app include src external model_specs tools webui/native/demo_voices
    git -C .cache/audio.cpp-ming-fast checkout --detach HEAD
    git -C .cache/audio.cpp-ming-fast apply "$REPO_ROOT/src/patches/ming_fast_cpu.patch"
else
    echo "      Reusing existing .cache/audio.cpp-ming-fast"
fi

if [ "$SKIP_MODEL" -eq 0 ]; then
    echo "[3/4] Checking Ming-3.0-TTS 0.5B Q8_0 GGUF model..."
    mkdir -p models
    if [ -f "$MODEL_PATH" ]; then
        CUR_SHA="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
        if [ "$CUR_SHA" != "$MODEL_SHA256" ]; then
            echo "      Checksum mismatch ($CUR_SHA), re-downloading..."
            rm -f "$MODEL_PATH"
        fi
    fi
    if [ ! -f "$MODEL_PATH" ]; then
        curl --fail --location --retry 3 --output "$MODEL_PATH" "$MODEL_URL"
        CUR_SHA="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
        if [ "$CUR_SHA" != "$MODEL_SHA256" ]; then
            echo "ERROR: Downloaded model SHA-256 mismatch: $CUR_SHA != $MODEL_SHA256" >&2
            exit 1
        fi
    fi
    echo "      Verified $MODEL_PATH"
else
    echo "[3/4] Skipping model download (--skip-model)"
fi

echo "[4/4] Building server, benchmark binaries and running parity tests (JOBS=$JOBS)..."
cmake -S benchmarks -B .cache/bench-build -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/bench-build \
    --target ming3_fast_server ming3_rtf ming3_fast_rtf ming3_fast_profile_rtf \
             ming3_cpu_linear_test ming3_fast_engine_test ming3_stream_vae_test \
    -j "$JOBS"
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE MING_VAE_QUANT=q4_0 \
    ctest --test-dir .cache/bench-build --output-on-failure

echo ""
echo "Setup complete! Start the real-time streaming TTS server with:"
echo "  make server            # foreground on http://0.0.0.0:8080"
echo "  make install-systemd   # persistent systemd --user background services"
