#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

if [ ! -x "$REPO_ROOT/.cache/bench-build/ming3_fast_server" ]; then
    echo "ERROR: ming3_fast_server not built yet. Run './scripts/setup.sh' or 'make build' first." >&2
    exit 1
fi

sed "s|/path/to/ming3-tts.cpp|$REPO_ROOT|g" \
    "$REPO_ROOT/deploy/ming3-tts.service.example" > "$UNIT_DIR/ming3-tts.service"
sed "s|/path/to/ming3-tts.cpp|$REPO_ROOT|g" \
    "$REPO_ROOT/deploy/ming3-tts-gateway.service.example" > "$UNIT_DIR/ming3-tts-gateway.service"

systemctl --user daemon-reload
systemctl --user enable --now ming3-tts.service
for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

systemctl --user enable --now ming3-tts-gateway.service
for _ in $(seq 1 15); do
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

echo "Installed and started:"
systemctl --user status ming3-tts.service ming3-tts-gateway.service --no-pager -l
echo ""
echo "Tip: run 'sudo loginctl enable-linger $USER' once so services stay running after SSH logout."
