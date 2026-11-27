"""Compare a Ming sampling experiment with a frozen reference using ABBA order."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', default='zh_short.txt')
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--reference', choices=('engine', 'tuned'), default='engine')
    parser.add_argument('--reference-steps', type=int, choices=(5, 6, 7, 10), default=10)
    parser.add_argument('--reference-cfg-interval', type=int, default=1)
    parser.add_argument('--reference-reuse-threshold', type=float, default=0.0)
    parser.add_argument('--reference-packed-q8', action='store_true')
    parser.add_argument('--reference-preserve-noise-stream', action='store_true')
    parser.add_argument('--reference-cfg-reuse-mode', choices=('delta', 'unconditional'), default='delta')
    parser.add_argument('--steps', type=int, choices=(5, 6, 7, 10), default=10)
    parser.add_argument('--cfg-interval', type=int, default=1)
    parser.add_argument('--cfg-reuse-threshold', type=float, default=0.0)
    parser.add_argument('--packed-q8', action='store_true')
    parser.add_argument('--vae-cache-mib', type=int, default=0)
    parser.add_argument('--preserve-noise-stream', action='store_true')
    parser.add_argument('--cfg-reuse-mode', choices=('delta', 'unconditional'), default='delta')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    if args.reference == 'engine' and (args.reference_steps != 10 or args.reference_cfg_interval != 1 or
                                       args.reference_reuse_threshold or args.reference_packed_q8 or
                                       args.reference_preserve_noise_stream or args.reference_cfg_reuse_mode != 'delta'):
        parser.error('Reference experiment settings require --reference tuned')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label = f'n{args.steps}-cfg{args.cfg_interval}-p{int(args.packed_q8)}'
    destination = ROOT / f'benchmarks/results/ming-trial-{label}-{stamp}'
    destination.mkdir()
    records = []
    signatures = {}
    reference_settings = None
    identical_required = (args.reference_steps == args.steps and
                          args.reference_cfg_interval == args.cfg_interval and
                          args.reference_reuse_threshold == args.cfg_reuse_threshold and
                          (args.reference_cfg_reuse_mode == args.cfg_reuse_mode or args.cfg_interval == 1) and
                          (args.reference_preserve_noise_stream == args.preserve_noise_stream or args.steps == 10))
    report = {'complete': False, 'arguments': vars(args), 'order': 'ABBA', 'threads': 4,
              'audio_bitwise_equal_required': identical_required, 'runs': records}

    def save():
        (destination / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(role):
        nonlocal reference_settings
        number = len(records) + 1
        variant = args.reference if role == 'reference' else 'tuned'
        command = [sys.executable, str(ROOT / 'benchmarks/run_ming3.py'), '--threads', '4',
                   '--variant', variant, '--warmups', '1', '--iterations', str(args.iterations),
                   '--text-file', str(ROOT / 'benchmarks' / args.fixture)]
        if variant == 'tuned':
            steps = args.reference_steps if role == 'reference' else args.steps
            interval = args.reference_cfg_interval if role == 'reference' else args.cfg_interval
            threshold = args.reference_reuse_threshold if role == 'reference' else args.cfg_reuse_threshold
            packed = args.reference_packed_q8 if role == 'reference' else args.packed_q8
            command.extend(['--flow-steps', str(steps), '--cfg-interval', str(interval),
                            '--cfg-reuse-threshold', str(threshold)])
            if role == 'candidate':
                command.extend(['--vae-cache-mib', str(args.vae_cache_mib)])
            if packed:
                command.append('--packed-q8')
            preserve = args.reference_preserve_noise_stream if role == 'reference' else args.preserve_noise_stream
            if preserve:
                command.append('--preserve-noise-stream')
            mode = args.reference_cfg_reuse_mode if role == 'reference' else args.cfg_reuse_mode
            command.extend(['--cfg-reuse-mode', mode])
        print(f'[{number}/4] {role}: {" ".join(command)}', flush=True)
        process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, check=False)
        (destination / f'launch_{number}.log').write_text(process.stdout)
        if process.returncode:
            raise RuntimeError(f'Run failed: {destination}/launch_{number}.log')
        folder = Path(next(line.removeprefix('Results: ') for line in process.stdout.splitlines()
                           if line.startswith('Results: ')))
        summary = json.loads((folder / 'summary.json').read_text())
        metadata = json.loads((folder / 'metadata.json').read_text())
        if not summary['complete']:
            raise RuntimeError('Incomplete benchmark')
        settings = {key: metadata[key] for key in (
            'verified_model_sha256', 'text', 'seed', 'backend', 'allowed_logical_cpus', 'family')}
        if reference_settings is None:
            reference_settings = settings
        if settings != reference_settings:
            raise RuntimeError('Weights, text, seed, backend or affinity changed')
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((folder / 'audio').glob('run_*.wav'))]
        if len(hashes) != args.iterations or len(set(hashes)) != 1:
            raise RuntimeError('Missing or nondeterministic measured WAVs')
        if role in signatures and signatures[role] != hashes[0]:
            raise RuntimeError('Audio changed between identical configuration blocks')
        signatures[role] = hashes[0]
        if identical_required and role == 'candidate' and signatures['reference'] != hashes[0]:
            raise RuntimeError('An engineering-only experiment changed the audio')
        record = {'role': role, 'variant': variant, 'command': command, 'results': str(folder),
                  'median_rtf': summary['median_rtf'], 'rtfs': [r['rtf'] for r in summary['measured_runs']],
                  'audio_seconds': [r['audio_s'] for r in summary['measured_runs']],
                  'synthesis_seconds': [r['synthesis_s'] for r in summary['measured_runs']],
                  'peak_rss_kib': max(r['peak_rss_kib'] for r in summary['measured_runs']),
                  'peak_cgroup_memory_bytes': summary['cgroup_peak_memory_bytes'],
                  'swap_bytes': summary['cgroup_peak_swap_bytes'], 'audio_sha256': hashes[0],
                  'binary_sha256': metadata['benchmark_binary_sha256']}
        records.append(record)
        save()
        print(f'  RTF={record["median_rtf"]:.6f}, audio={record["audio_seconds"]}', flush=True)

    save()
    try:
        for role in ('reference', 'candidate', 'candidate', 'reference'):
            run(role)
        measurements = {}
        for role in ('reference', 'candidate'):
            rows = [r for r in records if r['role'] == role]
            if len({r['binary_sha256'] for r in rows}) != 1:
                raise RuntimeError('The binary changed between comparison blocks')
            measurements[role] = {
                'median_rtf': statistics.median(v for r in rows for v in r['rtfs']),
                'peak_rss_gib': max(r['peak_rss_kib'] for r in rows) / 1024 ** 2,
                'max_swap_bytes': max(r['swap_bytes'] for r in rows)}
        ref = measurements['reference']['median_rtf']
        candidate = measurements['candidate']['median_rtf']
        report.update(complete=True, measurements=measurements,
                      rtf_reduction_percent=(1 - candidate / ref) * 100,
                      speedup=ref / candidate, measured_runs_per_variant=2 * args.iterations,
                      audio_bitwise_equal=signatures['reference'] == signatures['candidate'],
                      requires_audio_quality_evaluation=not identical_required)
    except BaseException as error:
        report['error'] = str(error)
        save()
        raise
    save()
    print('Comparison:', destination / 'comparison.json', flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
