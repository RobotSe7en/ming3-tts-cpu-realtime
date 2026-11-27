#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_BIN="$REPO_ROOT/.cache/bench-build/ming3_fast_server"
CONFIG_PATH="$REPO_ROOT/deploy/server.json"
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
THREADS="${THREADS:-4}"

if [ ! -x "$SERVER_BIN" ]; then
    echo "ERROR: $SERVER_BIN not found. Run ./scripts/setup.sh or 'make build' first." >&2
    exit 1
fi

cleanup() {
    echo ""
    echo "[start_server] Stopping backend and gateway..."
    kill "${GATEWAY_PID:-}" "${BACKEND_PID:-}" 2>/dev/null || true
    wait "${GATEWAY_PID:-}" "${BACKEND_PID:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[start_server] Starting ming3_fast_server (Plan A, ${THREADS} threads) on 127.0.0.1:18080..."
(
    cd "$REPO_ROOT/.cache/audio.cpp"
    OMP_NUM_THREADS="$THREADS" \
    OMP_THREAD_LIMIT="$THREADS" \
    OMP_DYNAMIC=FALSE \
    OPENBLAS_NUM_THREADS="$THREADS" \
    MKL_NUM_THREADS="$THREADS" \
    MING_FLOW_QUANT="${MING_FLOW_QUANT:-q4g128}" \
    MING_AGG_QUANT="${MING_AGG_QUANT:-q4g128}" \
    MING_AR_QUANT="${MING_AR_QUANT:-q8}" \
    MING_VAE_QUANT="${MING_VAE_QUANT:-q4_0}" \
    MING_FLOW_STEPS="${MING_FLOW_STEPS:-8}" \
    MING_CFG_INTERVAL="${MING_CFG_INTERVAL:-1}" \
    MING_REUSE_ANCHOR_LATENT="${MING_REUSE_ANCHOR_LATENT:-1}" \
    exec "$SERVER_BIN" --config "$CONFIG_PATH"
) &
BACKEND_PID=$!

for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "ERROR: ming3_fast_server exited prematurely." >&2
        exit 1
    fi
    sleep 1
done

echo "[start_server] Starting AYTS/1 streaming gateway on http://${GATEWAY_HOST}:${GATEWAY_PORT}..."
python3 "$REPO_ROOT/deploy/tts_gateway.py" \
    --host "$GATEWAY_HOST" \
    --port "$GATEWAY_PORT" \
    --backend-host 127.0.0.1 \
    --backend-port 18080 &
GATEWAY_PID=$!

wait "$GATEWAY_PID"
