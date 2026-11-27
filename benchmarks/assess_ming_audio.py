"""Assess changed Ming WAVs locally with ASR CER and CAMPPlus voice similarity.

These are automated screening measurements, not a listening test or MOS score.
Run this separately from timed TTS benchmarks, inside the same resource limits.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import unicodedata
import wave

ROOT = Path(__file__).resolve().parent.parent


def normalized(text):
    text = re.sub(r'<\|.*?\|>', '', text)
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, x in enumerate(reference, 1):
        current = [i]
        for j, y in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import sherpa_onnx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparisons', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/sensevoice-int8')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = args.model_dir / 'model.int8.onnx'
    tokens = args.model_dir / 'tokens.txt'
    checksums = json.loads((args.model_dir / 'checksums.json').read_text())
    for name, digest in checksums['files'].items():
        if sha256(args.model_dir / name) != digest:
            raise RuntimeError(f'ASR model checksum mismatch: {name}')
    cases = []
    waves = {}
    for path in args.comparisons:
        comparison = json.loads(path.read_text())
        if not comparison['complete']:
            raise RuntimeError(f'Incomplete comparison: {path}')
        case = {'comparison': str(path.resolve()), 'wav': {}}
        for role in ('reference', 'candidate'):
            row = next(r for r in comparison['runs'] if r['role'] == role)
            folder = Path(row['results'])
            wav = (folder / 'audio/run_1.wav').resolve()
            metadata = json.loads((folder / 'metadata.json').read_text())
            case['wav'][role] = str(wav)
            case['text'] = metadata['text']
            waves.setdefault(str(wav), {'text': metadata['text'], 'sha256': sha256(wav)})
        cases.append(case)
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model), tokens=str(tokens), num_threads=4,
        provider='cpu', language='zh', use_itn=False, debug=False)
    for name, record in waves.items():
        with wave.open(name, 'rb') as audio:
            if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
                raise RuntimeError('Quality checks expect mono PCM16 WAVs from the benchmark')
            sample_rate = audio.getframerate()
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)
        transcript = stream.result.text
        reference, hypothesis = normalized(record['text']), normalized(transcript)
        distance = edit_distance(reference, hypothesis)
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        record.update(transcript=transcript, normalized_reference=reference, normalized_hypothesis=hypothesis,
                      edit_distance=distance, cer=distance / max(1, len(reference)),
                      duration_s=len(samples) / sample_rate, rms_dbfs=20 * math.log10(max(rms, 1.0e-12)),
                      clipping_fraction=float(np.mean(np.abs(samples) >= 32760 / 32768)))
        print(f'ASR CER={record["cer"]:.4f}: {name}', flush=True)
    del recognizer, stream
    gc.collect()
    paths = args.output / 'wav_paths.txt'
    paths.write_text('\n'.join(waves) + '\n')
    command = [str(ROOT / '.cache/bench-build/ming3_quality_embed'),
               str(ROOT / 'models/ming-3.0-tts-0.5b-q8_0.gguf'), str(paths.resolve())]
    embeddings_path = args.output / 'embeddings.jsonl'
    with embeddings_path.open('w') as output, (args.output / 'embedding.log').open('w') as log:
        subprocess.run(command, cwd=ROOT / '.cache/audio.cpp', stdout=output, stderr=log, check=True)
    embeddings = {r['path']: np.asarray(r['embedding'], dtype=np.float64)
                  for line in embeddings_path.read_text().splitlines() if line.startswith('{')
                  for r in [json.loads(line)]}
    for case in cases:
        a, b = (embeddings[case['wav'][role]] for role in ('reference', 'candidate'))
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if not math.isfinite(cosine):
            raise RuntimeError('Nonfinite speaker similarity')
        case.update(speaker_cosine=cosine,
                    reference_cer=waves[case['wav']['reference']]['cer'],
                    candidate_cer=waves[case['wav']['candidate']]['cer'])
        case['cer_change'] = case['candidate_cer'] - case['reference_cer']
        case['review_flag'] = case['cer_change'] > 0.03 or cosine < 0.90
    report = {'complete': True, 'automated_screening_only': True, 'human_listening_performed': False,
              'asr': {'model_revision': checksums['revision'], 'model_sha256': sha256(model),
                      'tokens_sha256': sha256(tokens), 'threads': 4, 'language': 'zh', 'use_itn': False,
                      'sherpa_onnx_version': importlib.metadata.version('sherpa-onnx')},
              'speaker': {'model': 'CAMPPlus packaged with the Ming GGUF', 'command': command,
                          'binary_sha256': sha256(Path(command[0])), 'threads': 4},
              'script_sha256': sha256(Path(__file__)), 'cases': cases, 'audio': waves,
              'review_flag_definition': 'CER increases more than 0.03 or CAMPPlus cosine below 0.90; heuristic only'}
    (args.output / 'quality.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print('Quality:', args.output / 'quality.json', flush=True)
    print(json.dumps(cases, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
