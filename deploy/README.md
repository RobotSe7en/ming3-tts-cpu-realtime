# Real-Time HTTP & Streaming TTS Deployment Guide

This directory contains the production serving configuration for **Ming-3.0-TTS 0.5B** on x86 CPU (Plan A + 24-layer sliding-window KV-cache streaming VAE).

## Architecture

```text
Client ──► http://0.0.0.0:8080   (deploy/tts_gateway.py, Python stdlib + in-process libsoxr, ~14 MiB RAM)
              │
              ├─ POST /v1/audio/speech/stream ──► AYTS/1 24 kHz mono s16le real-time stream (default prebuffer 1000 ms)
              ├─ POST /v1/audio/speech        ──► OpenAI-compatible 44.1 kHz WAV (shares hot streaming model, zero reload)
              └─ GET  /health, /v1/models, /v1/audio/voices ──► proxied
              │
              ▼
          http://127.0.0.1:18080 (ming3_fast_server, single resident Plan A streaming model, ~1.4 GiB RAM / 0 swap)
```

## Quick Start

```bash
# Foreground tryout
make server

# Persistent systemd --user services (auto-start on boot with linger enabled)
make install-systemd
```

## Endpoints & Examples

### 1. Frame-Level Streaming (`POST /v1/audio/speech/stream`)

Returns an HTTP/1.1 chunked **`AYTS/1`** stream (`Content-Type: application/x-ayni-pcm-stream`):

- **16-byte header** (little-endian): `"AYTS"` (`4B`) + `version=1` (`u32`) + `sample_rate` (`u32`, default `24000`) + `channels=1` (`u32`).
- **Frames**: `type` (`u32`) + `len` (`u32`) + `payload`:
  - `type = 1`: s16le mono PCM chunk (up to `15360 B` = `0.32 s` at 24 kHz).
  - `type = 2`: end-of-stream frame (`len = 4`, payload = `u32 total_pcm_bytes`).

```bash
curl --no-buffer --noproxy '*' \
  'http://127.0.0.1:8080/v1/audio/speech/stream' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ming3-tts",
    "input": "你好，这是流式语音合成测试。",
    "voice": "灵小柔",
    "lang_code": "Chinese",
    "response_format": "pcm",
    "sample_rate": 24000
  }' \
  --output speech.ayts
```

Optional request fields:
- `prebuffer_ms` (`0..5000`, default `1000`): audio buffered before releasing the first burst (`1000` gives ~1.7 s TTFT and ~1.16 s initial playback margin for `0 ms` underrun on long paragraphs; `0` gives ~1.1–1.3 s TTFT).
- `sample_rate`: `16000`, `22050`, `24000` (default), `44100`, or `48000` (stateful in-process `libsoxr`, falls back to `ffmpeg`).
- `response_format: "raw_pcm"` (or `stream_format: "raw"`): headerless s16le PCM stream (`audio/pcm`).
- `options`: e.g. `{"stream_chunk_policy": "grow", "stream_frames_per_event": 8}` for sub-second TTFT (`~0.87 s` with `prebuffer_ms: 0`).

### 2. OpenAI-Compatible Offline WAV (`POST /v1/audio/speech`)

```bash
curl -sS 'http://127.0.0.1:8080/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d '{"model":"ming3-tts","input":"窗外下起了小雨，桌上的热茶冒着白气。","voice":"灵小柔"}' \
  --output out.wav
```

Python `openai` SDK:
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
with client.audio.speech.with_streaming_response.create(
    model="ming3-tts",
    voice="lingxiaorou",  # aliases: "灵小柔", "alloy", "uncond", "emma", "james"
    input="窗外下起了小雨，桌上的热茶冒着白气。",
) as resp:
    resp.stream_to_file("out.wav")
```

Zero-shot voice cloning (automatically overrides default preset):
```bash
REF_B64=$(base64 -w0 ref.wav)
curl -sS 'http://127.0.0.1:8080/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"ming3-tts\",\"input\":\"克隆音色测试。\",\"voice_ref\":{\"type\":\"base64\",\"data\":\"${REF_B64}\"},\"reference_text\":\"参考音频对应文本\"}" \
  --output cloned.wav
```

## Operations & Streaming Diagnostics

```bash
systemctl --user status ming3-tts.service ming3-tts-gateway.service
systemctl --user restart ming3-tts.service ming3-tts-gateway.service
journalctl --user -u ming3-tts-gateway.service -f
```

Every streaming request logs a single diagnostic line:
```text
[stream] client=127.0.0.1 chars=107 voice=灵小柔 sr=24000 prebuf=1000ms ttft=1702ms wall=17.949s dur=18.900s rtf=0.9497 bursts=123 max_gap=1222ms underrun_0ms=0ms
```
