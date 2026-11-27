#!/usr/bin/env python3
"""HTTP/1.1 streaming gateway exposing AYTS/1 real-time TTS over ming3_fast_server.

Exposes:
- POST /v1/audio/speech/stream : true frame-level streaming TTS (stateful KV-cached
  VAE backend -> in-process libsoxr stateful resampling -> AYTS/1 24 kHz s16le PCM).
- POST /v1/audio/speech : OpenAI-compatible WAV/PCM endpoint sharing the same single
  resident streaming model in RAM (no cold model reloads between offline & streaming).
- All other endpoints (/health, /v1/models, /v1/audio/voices, ...) transparently proxied.
"""

import argparse
import ctypes
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import wave


AYTS_MAGIC = b"AYTS"
AYTS_VERSION = 1
AYTS_FRAME_PCM = 1
AYTS_FRAME_EOF = 2
SUPPORTED_SAMPLE_RATES = {16000, 22050, 24000, 44100, 48000}
BACKEND_STREAM_MODEL = "ming3-tts-stream"
DEFAULT_PREBUFFER_MS = int(os.environ.get("GATEWAY_PREBUFFER_MS", "1000"))


class _SoxrIOSpec(ctypes.Structure):
    _fields_ = [
        ("itype", ctypes.c_int),
        ("otype", ctypes.c_int),
        ("scale", ctypes.c_double),
        ("e", ctypes.c_void_p),
        ("flags", ctypes.c_ulong),
    ]


def _load_libsoxr():
    for name in ("libsoxr.so.0", "libsoxr.so"):
        try:
            lib = ctypes.CDLL(name)
            lib.soxr_create.restype = ctypes.c_void_p
            lib.soxr_create.argtypes = [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(_SoxrIOSpec),
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            lib.soxr_process.restype = ctypes.c_char_p
            lib.soxr_process.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            lib.soxr_delete.argtypes = [ctypes.c_void_p]
            return lib
        except OSError:
            continue
    return None


_LIBSOXR = _load_libsoxr()


class StatefulResampler:
    """Zero-buffering stateful s16le mono resampler using libsoxr (with ffmpeg fallback)."""

    def __init__(self, src_sr: int, dst_sr: int) -> None:
        self.src_sr = src_sr
        self.dst_sr = dst_sr
        self._soxr = None
        self._pending_byte = b""
        if src_sr != dst_sr and _LIBSOXR is not None:
            err = ctypes.c_char_p()
            # SOXR_INT16_I = 3
            io_spec = _SoxrIOSpec(3, 3, 1.0, None, 0)
            handle = _LIBSOXR.soxr_create(
                float(src_sr), float(dst_sr), 1, ctypes.byref(err), ctypes.byref(io_spec), None, None
            )
            if not handle or err.value:
                raise RuntimeError(f"libsoxr create failed: {err.value}")
            self._soxr = handle

    def process(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes:
            return b""
        data = self._pending_byte + pcm_bytes
        if len(data) & 1:
            self._pending_byte = data[-1:]
            data = data[:-1]
        else:
            self._pending_byte = b""
        if not data:
            return b""
        if self.src_sr == self.dst_sr:
            return data
        if self._soxr is not None:
            in_samples = len(data) // 2
            out_cap = max(256, int(math.ceil(in_samples * self.dst_sr / self.src_sr)) + 256)
            out_buf = ctypes.create_string_buffer(out_cap * 2)
            in_done = ctypes.c_size_t(0)
            out_done = ctypes.c_size_t(0)
            err = _LIBSOXR.soxr_process(
                self._soxr,
                ctypes.c_char_p(data),
                in_samples,
                ctypes.byref(in_done),
                out_buf,
                out_cap,
                ctypes.byref(out_done),
            )
            if err:
                raise RuntimeError(f"libsoxr process failed: {err.decode('utf-8', errors='replace')}")
            return out_buf.raw[: out_done.value * 2]
        # Fallback if libsoxr is unavailable
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(self.src_sr), "-ac", "1", "-i", "pipe:0",
            "-f", "s16le", "-ar", str(self.dst_sr), "-ac", "1", "pipe:1",
        ]
        proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg resample failed: {proc.stderr.decode('utf-8', errors='replace').strip()}")
        return proc.stdout

    def flush(self) -> bytes:
        if self._soxr is None:
            return b""
        chunks = []
        out_cap = 4096
        out_buf = ctypes.create_string_buffer(out_cap * 2)
        out_done = ctypes.c_size_t(0)
        while True:
            err = _LIBSOXR.soxr_process(
                self._soxr, None, 0, None, out_buf, out_cap, ctypes.byref(out_done)
            )
            if err:
                raise RuntimeError(f"libsoxr flush failed: {err.decode('utf-8', errors='replace')}")
            if out_done.value == 0:
                break
            chunks.append(out_buf.raw[: out_done.value * 2])
        return b"".join(chunks)

    def close(self) -> None:
        if self._soxr is not None:
            _LIBSOXR.soxr_delete(self._soxr)
            self._soxr = None


def encode_ayts_header(sample_rate: int, channels: int = 1) -> bytes:
    return struct.pack("<4s3I", AYTS_MAGIC, AYTS_VERSION, sample_rate, channels)


def encode_ayts_pcm_frame(pcm_slice: bytes) -> bytes:
    return struct.pack("<2I", AYTS_FRAME_PCM, len(pcm_slice)) + pcm_slice


def encode_ayts_eof_frame(total_pcm_bytes: int) -> bytes:
    return struct.pack("<2II", AYTS_FRAME_EOF, 4, total_pcm_bytes)


def wrap_pcm_as_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def build_backend_stream_payload(req: dict, default_policy: str = "fixed", default_frames: int = 8) -> dict:
    extra_options = dict(req.get("options") or {}) if isinstance(req.get("options"), dict) else {}
    extra_options.setdefault("stream_chunk_policy", default_policy)
    extra_options.setdefault("stream_frames_per_event", default_frames)
    payload: dict = {
        "model": BACKEND_STREAM_MODEL,
        "input": req["input"],
        "response_format": "pcm",
        "stream_format": "audio",
        "options": extra_options,
    }
    for key in ("voice", "voice_ref", "reference_text", "seed", "max_tokens", "speed", "speaking_rate", "language"):
        val = req.get(key)
        if val is not None and val != "":
            payload[key] = val
    return payload


class GatewayServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_cls, backend_host: str, backend_port: int, backend_timeout: int):
        super().__init__(server_address, handler_cls)
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.backend_timeout = backend_timeout


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[gateway] {self.address_string()} - {fmt % args}\n")
        sys.stderr.flush()

    def _send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    def _write_http_chunk(self, data: bytes) -> None:
        if not data:
            return
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _finish_http_chunked(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _read_request_body(self) -> bytes:
        length_hdr = self.headers.get("Content-Length")
        if length_hdr is not None:
            length = int(length_hdr)
            return self.rfile.read(length) if length > 0 else b""
        if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            chunks = []
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                size = int(line.split(b";", 1)[0], 16)
                if size == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)
            return b"".join(chunks)
        return b""

    def _proxy_request(self, method: str, pre_read_body: bytes | None = None) -> None:
        body = pre_read_body if pre_read_body is not None else (
            self._read_request_body() if method in ("POST", "PUT", "PATCH") else b""
        )
        fwd_headers = {}
        for k, v in self.headers.items():
            kl = k.lower()
            if kl in ("host", "connection", "transfer-encoding", "keep-alive", "proxy-connection"):
                continue
            fwd_headers[k] = v
        if body:
            fwd_headers["Content-Length"] = str(len(body))
        try:
            conn = http.client.HTTPConnection(
                self.server.backend_host, self.server.backend_port, timeout=self.server.backend_timeout
            )
            conn.request(method, self.path, body=body if body else None, headers=fwd_headers)
            resp = conn.getresponse()
            self.send_response(resp.status, resp.reason)
            is_chunked = False
            has_length = False
            for k, v in resp.getheaders():
                kl = k.lower()
                if kl in ("connection", "keep-alive", "proxy-connection"):
                    continue
                if kl == "transfer-encoding" and "chunked" in v.lower():
                    is_chunked = True
                if kl == "content-length":
                    has_length = True
                self.send_header(k, v)
            if not is_chunked and not has_length:
                self.send_header("Transfer-Encoding", "chunked")
                is_chunked = True
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    buf = resp.read(8192)
                    if not buf:
                        break
                    if is_chunked:
                        self._write_http_chunk(buf)
                    else:
                        self.wfile.write(buf)
                        self.wfile.flush()
                if is_chunked:
                    self._finish_http_chunked()
            finally:
                conn.close()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self._send_json(502, {"error": f"backend error: {exc}"})

    def _handle_speech_offline_shared(self, raw_body: bytes) -> None:
        """Serve POST /v1/audio/speech (WAV) via the hot streaming model to avoid model swaps."""
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._proxy_request("POST", raw_body)
            return
        if not isinstance(req, dict):
            self._proxy_request("POST", raw_body)
            return

        # Only intercept standard WAV calls for ming3-tts; pass SSE/JSON/custom requests straight through
        model = req.get("model")
        resp_fmt = req.get("response_format", "wav")
        stream_fmt = req.get("stream_format")
        if model not in (None, "", "ming3-tts") or resp_fmt != "wav" or stream_fmt is not None:
            self._proxy_request("POST", raw_body)
            return

        text = req.get("input")
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            self._send_json(400, {"error": "input must be nonempty text without NUL"})
            return

        backend_payload = build_backend_stream_payload(req, default_policy="fixed", default_frames=8)
        body = json.dumps(backend_payload, ensure_ascii=False).encode("utf-8")
        conn = http.client.HTTPConnection(
            self.server.backend_host, self.server.backend_port, timeout=self.server.backend_timeout
        )
        try:
            conn.request(
                "POST",
                "/v1/audio/speech",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            if resp.status != 200:
                err_data = resp.read()
                self.send_response(resp.status, resp.reason)
                self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(err_data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(err_data)
                self.wfile.flush()
                return
            src_sr = int(resp.getheader("X-Audio-Sample-Rate", "44100"))
            pcm_bytes = resp.read()
        except Exception as exc:
            self._send_json(502, {"error": f"backend error: {exc}"})
            return
        finally:
            conn.close()

        wav_bytes = wrap_pcm_as_wav(pcm_bytes, src_sr, 1)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(wav_bytes)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _handle_speech_stream(self) -> None:
        raw_body = self._read_request_body()
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        if not isinstance(req, dict):
            self._send_json(400, {"error": "JSON body must be an object"})
            return

        text = req.get("input")
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            self._send_json(400, {"error": "input must be nonempty text without NUL"})
            return

        sample_rate = req.get("sample_rate", 24000)
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate not in SUPPORTED_SAMPLE_RATES:
            self._send_json(400, {"error": f"sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}"})
            return

        response_format = req.get("response_format", "pcm")
        if response_format not in ("pcm", "raw_pcm"):
            self._send_json(400, {"error": "response_format must be pcm or raw_pcm for /v1/audio/speech/stream"})
            return
        use_ayts = (req.get("stream_format", "ayts") != "raw") and (response_format == "pcm")

        prebuffer_ms = req.get("prebuffer_ms", DEFAULT_PREBUFFER_MS)
        if not isinstance(prebuffer_ms, int) or isinstance(prebuffer_ms, bool) or not 0 <= prebuffer_ms <= 5000:
            self._send_json(400, {"error": "prebuffer_ms must be an integer between 0 and 5000"})
            return
        prebuffer_bytes = (sample_rate * 2 * prebuffer_ms // 1000)
        prebuffer_bytes -= prebuffer_bytes % 2

        # 0.32s max AYTS PCM frame -> 15360 bytes at 24 kHz mono s16le (exact AYTS/1 match)
        max_frame_bytes = (sample_rate * 2 * 32 // 100)
        max_frame_bytes -= max_frame_bytes % 2

        backend_payload = build_backend_stream_payload(req, default_policy="fixed", default_frames=8)
        body = json.dumps(backend_payload, ensure_ascii=False).encode("utf-8")

        conn = http.client.HTTPConnection(
            self.server.backend_host, self.server.backend_port, timeout=self.server.backend_timeout
        )
        resampler: StatefulResampler | None = None
        headers_sent = False
        total_pcm_bytes = 0
        prebuf = bytearray()
        t_start = time.perf_counter()
        ttft_s: float | None = None
        last_emit_t: float | None = None
        play_end_s: float | None = None
        max_gap_s = 0.0
        underrun_s = 0.0
        emitted_bursts = 0

        def emit_pcm(pcm_chunk: bytes) -> None:
            nonlocal headers_sent, total_pcm_bytes, ttft_s, last_emit_t, play_end_s, max_gap_s, underrun_s, emitted_bursts
            if not pcm_chunk:
                return
            now = time.perf_counter() - t_start
            chunk_dur = (len(pcm_chunk) / 2) / sample_rate
            if not headers_sent:
                ttft_s = now
                play_end_s = now + chunk_dur
                self.send_response(200)
                if use_ayts:
                    self.send_header("Content-Type", "application/x-ayni-pcm-stream")
                else:
                    self.send_header("Content-Type", "audio/pcm")
                    self.send_header("X-Audio-Sample-Rate", str(sample_rate))
                    self.send_header("X-Audio-Channels", "1")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                if use_ayts:
                    self._write_http_chunk(encode_ayts_header(sample_rate, 1))
                headers_sent = True
            else:
                if last_emit_t is not None:
                    gap = now - last_emit_t
                    if gap > max_gap_s:
                        max_gap_s = gap
                if play_end_s is not None:
                    if now > play_end_s:
                        underrun_s += now - play_end_s
                        play_end_s = now + chunk_dur
                    else:
                        play_end_s += chunk_dur
            last_emit_t = now
            emitted_bursts += 1

            if use_ayts:
                out_parts = []
                for off in range(0, len(pcm_chunk), max_frame_bytes):
                    piece = pcm_chunk[off:off + max_frame_bytes]
                    out_parts.append(encode_ayts_pcm_frame(piece))
                    total_pcm_bytes += len(piece)
                self._write_http_chunk(b"".join(out_parts))
            else:
                self._write_http_chunk(pcm_chunk)
                total_pcm_bytes += len(pcm_chunk)

        try:
            conn.request(
                "POST",
                "/v1/audio/speech",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            if resp.status != 200:
                err_text = resp.read().decode("utf-8", errors="replace").strip()
                self._send_json(resp.status, {"error": err_text or f"backend HTTP {resp.status}"})
                return

            src_sr = int(resp.getheader("X-Audio-Sample-Rate", "44100"))
            resampler = StatefulResampler(src_sr, sample_rate)

            # Read chunked HTTP stream incrementally using read1() so we never block filling a full buffer
            while True:
                buf = resp.read1(16384) if hasattr(resp, "read1") else resp.read(4096)
                if not buf:
                    break
                resampled = resampler.process(buf)
                if not resampled:
                    continue
                if not headers_sent and prebuffer_bytes > 0:
                    prebuf.extend(resampled)
                    if len(prebuf) >= prebuffer_bytes:
                        emit_pcm(bytes(prebuf))
                        prebuf.clear()
                else:
                    emit_pcm(resampled)

            tail = resampler.flush()
            if not headers_sent and prebuf:
                if tail:
                    prebuf.extend(tail)
                emit_pcm(bytes(prebuf))
                prebuf.clear()
            elif tail:
                emit_pcm(tail)

            if headers_sent:
                if use_ayts:
                    self._write_http_chunk(encode_ayts_eof_frame(total_pcm_bytes))
                self._finish_http_chunked()
                wall = time.perf_counter() - t_start
                audio_dur = (total_pcm_bytes / 2) / sample_rate
                rtf = wall / audio_dur if audio_dur > 0 else 0.0
                voice_lbl = req.get("voice") or "default"
                sys.stderr.write(
                    f"[stream] client={self.address_string()} chars={len(text)} voice={voice_lbl} sr={sample_rate} "
                    f"prebuf={prebuffer_ms}ms ttft={(ttft_s or 0.0)*1000:.0f}ms wall={wall:.3f}s dur={audio_dur:.3f}s "
                    f"rtf={rtf:.4f} bursts={emitted_bursts} max_gap={max_gap_s*1000:.0f}ms underrun_0ms={underrun_s*1000:.0f}ms\n"
                )
                sys.stderr.flush()
            else:
                self._send_json(502, {"error": "backend produced empty audio stream"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            if not headers_sent:
                self._send_json(502, {"error": str(exc)})
            else:
                sys.stderr.write(f"[gateway] stream aborted mid-response: {exc}\n")
                sys.stderr.flush()
        finally:
            if resampler is not None:
                resampler.close()
            conn.close()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:
        self._proxy_request("GET")

    def do_HEAD(self) -> None:
        self._proxy_request("HEAD")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/v1/audio/speech/stream":
            self._handle_speech_stream()
            return
        if parsed.path == "/v1/audio/speech":
            self._handle_speech_offline_shared(self._read_request_body())
            return
        self._proxy_request("POST")


def main() -> None:
    parser = argparse.ArgumentParser(description="AYTS/1 real-time streaming gateway for ming3_fast_server")
    parser.add_argument("--host", default=os.environ.get("GATEWAY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GATEWAY_PORT", "8080")))
    parser.add_argument("--backend-host", default=os.environ.get("BACKEND_HOST", "127.0.0.1"))
    parser.add_argument("--backend-port", type=int, default=int(os.environ.get("BACKEND_PORT", "18080")))
    parser.add_argument("--backend-timeout", type=int, default=int(os.environ.get("BACKEND_TIMEOUT", "180")))
    args = parser.parse_args()

    server = GatewayServer(
        (args.host, args.port),
        GatewayHandler,
        args.backend_host,
        args.backend_port,
        args.backend_timeout,
    )

    def handle_signal(signum, _frame):
        sys.stderr.write(f"[gateway] received signal {signum}, shutting down\n")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    resampler_name = "libsoxr (in-process)" if _LIBSOXR is not None else "ffmpeg (fallback)"
    sys.stderr.write(
        f"[gateway] listening on http://{args.host}:{args.port} -> backend http://{args.backend_host}:{args.backend_port} "
        f"[resampler: {resampler_name}]\n"
    )
    sys.stderr.flush()
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
