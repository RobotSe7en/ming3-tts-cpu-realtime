"""Run sequential ABBA benchmarks comparing baseline vs fast Ming CPU engine."""

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
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--reference', choices=('baseline', 'engine'), default='baseline')
    parser.add_argument('--flow-quant', choices=('q8', 'q4g64', 'q4g128'), default='q4g64')
    parser.add_argument('--agg-quant', choices=('q8', 'q4g64', 'q4g128'), default=None)
    parser.add_argument('--ar-quant', choices=('q8', 'f16', 'q4_0', 'iq4_nl'), default='q4_0')
    parser.add_argument('--vae-quant', choices=('q8', 'f16', 'q4_0', 'iq4_nl'), default='q4_0')
    parser.add_argument('--fused-fp32-act', choices=('auto', '0', '1'), default='auto')
    parser.add_argument('--flow-steps', type=int, choices=range(5, 11), default=10)
    parser.add_argument('--cfg-interval', type=int, choices=range(1, 11), default=1)
    parser.add_argument('--reuse-anchor-latent', choices=('0', '1'), default='1')
    parser.add_argument('--template-name', default='')
    parser.add_argument('--instruct', default='')
    parser.add_argument('--tag-suffix', default='')
    parser.add_argument('--skip-regressions', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    agg_quant = args.agg_quant or args.flow_quant
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    tag = f'{args.flow_quant}-ar_{args.ar_quant}-vae_{args.vae_quant}-s{args.flow_steps}-c{args.cfg_interval}'
    if args.tag_suffix:
        tag += f'-{args.tag_suffix}'
    destination = ROOT / f'benchmarks/results/ming-fast-comparison-{tag}-t{args.threads}-{stamp}'
    destination.mkdir(parents=True)
    records = []
    expected_hashes = {}

    def run(role, fixture, iterations):
        index = len(records) + 1
        variant = args.reference if role == 'reference' else 'fast'
        command = [sys.executable, str(ROOT / 'benchmarks/run_ming3.py'),
                   '--threads', str(args.threads), '--variant', variant,
                   '--warmups', '1', '--iterations', str(iterations),
                   '--text-file', str(ROOT / 'benchmarks' / fixture)]
        if args.template_name:
            command.extend(['--template-name', args.template_name])
        if args.instruct:
            command.extend(['--instruct', args.instruct])
        if variant == 'fast':
            command.extend(['--flow-quant', args.flow_quant, '--agg-quant', agg_quant,
                            '--ar-quant', args.ar_quant, '--vae-quant', args.vae_quant,
                            '--fused-fp32-act', args.fused_fp32_act,
                            '--flow-steps', str(args.flow_steps), '--cfg-interval', str(args.cfg_interval),
                            '--reuse-anchor-latent', args.reuse_anchor_latent])
        print(f'[{index}] {role} ({variant}): {fixture}, warmup=1, measured={iterations}', flush=True)
        process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, check=False)
        (destination / f'launch_{index}.log').write_text(process.stdout)
        if process.returncode:
            raise RuntimeError(f'Benchmark failed; inspect {destination}/launch_{index}.log')
        lines = [line.removeprefix('Results: ') for line in process.stdout.splitlines()
                 if line.startswith('Results: ')]
        if len(lines) != 1:
            raise RuntimeError('Expected exactly one result directory')
        folder = Path(lines[0])
        summary = json.loads((folder / 'summary.json').read_text())
        if not summary['complete']:
            raise RuntimeError('Incomplete benchmark')
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((folder / 'audio').glob('run_*.wav'))]
        if len(hashes) != iterations or len(set(hashes)) != 1:
            raise RuntimeError(f'Audio hash mismatch within run: {folder}')
        key = (role, fixture)
        if key in expected_hashes and expected_hashes[key] != hashes[0]:
            raise RuntimeError(f'Non-deterministic audio across blocks for {key}: {folder}')
        expected_hashes[key] = hashes[0]
        record = {
            'role': role, 'variant': variant, 'fixture': fixture, 'results': str(folder),
            'median_rtf': summary['median_rtf'],
            'rtfs': [row['rtf'] for row in summary['measured_runs']],
            'audio_s': [row['audio_s'] for row in summary['measured_runs']],
            'peak_rss_kib': max(row['peak_rss_kib'] for row in summary['measured_runs']),
            'cpu_percent': [row['cpu_percent_one_core_100'] for row in summary['measured_runs']],
            'swap_bytes': summary['cgroup_peak_swap_bytes'],
            'audio_sha256': hashes,
        }
        records.append(record)
        (destination / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
        print(f'  RTF={record["median_rtf"]:.6f}, audio_s={record["audio_s"][0]:.3f}, '
              f'RSS={record["peak_rss_kib"]/1024**2:.3f} GiB', flush=True)

    for role in ('reference', 'candidate', 'candidate', 'reference'):
        run(role, args.fixture, args.iterations)
    if not args.skip_regressions:
        for fixture in ('zh_regression.txt', 'zh_long_regression.txt'):
            run('reference', fixture, 1)
            run('candidate', fixture, 1)

    measurements = {}
    for role in ('reference', 'candidate'):
        rows = [row for row in records if row['role'] == role and row['fixture'] == args.fixture]
        measurements[role] = {
            'median_rtf': statistics.median(value for row in rows for value in row['rtfs']),
            'rtfs': [value for row in rows for value in row['rtfs']],
            'peak_rss_gib': max(row['peak_rss_kib'] for row in rows) / 1024 ** 2,
            'max_swap_bytes': max(row['swap_bytes'] for row in rows),
        }
    ref_rtf = measurements['reference']['median_rtf']
    cand_rtf = measurements['candidate']['median_rtf']
    bitwise_equal = expected_hashes.get(('reference', args.fixture)) == expected_hashes.get(('candidate', args.fixture))
    report = {
        'complete': True,
        'order': 'ABBA',
        'threads': args.threads,
        'warmups_per_block': 1,
        'fixture': args.fixture,
        'arguments': {**vars(args), 'agg_quant': agg_quant},
        'measured_runs_per_variant': 2 * args.iterations,
        'measurements': measurements,
        'rtf_reduction_percent': (1 - cand_rtf / ref_rtf) * 100,
        'speedup': ref_rtf / cand_rtf,
        'audio_bitwise_equal': bitwise_equal,
        'runs': records,
    }
    (destination / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Comparison:', destination / 'comparison.json', flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
