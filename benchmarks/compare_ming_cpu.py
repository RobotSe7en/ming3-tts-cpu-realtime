"""Run sequential ABBA benchmarks and check CPU optimization audio parity."""

from datetime import datetime, timezone
import argparse
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
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--skip-regressions', action='store_true')
    parser.add_argument('--candidate', choices=('cpu-opt', 'engine'), default='cpu-opt')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 'ming-cpu' if args.candidate == 'cpu-opt' else 'ming-engine'
    destination = ROOT / f'benchmarks/results/{prefix}-comparison-t{args.threads}-{stamp}'
    destination.mkdir(parents=True)
    records = []
    expected = {}

    def run(variant, fixture, iterations):
        index = len(records) + 1
        print(f'[{index}] {variant}: {fixture}, warmup=1, measured={iterations}', flush=True)
        command = [sys.executable, str(ROOT / 'benchmarks/run_ming3.py'),
                   '--threads', str(args.threads), '--variant', variant, '--warmups', '1',
                   '--iterations', str(iterations), '--text-file', str(ROOT / 'benchmarks' / fixture)]
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
        if len(hashes) != iterations:
            raise RuntimeError('Missing generated audio')
        if variant == 'baseline':
            if fixture in expected and expected[fixture] != hashes:
                raise RuntimeError('Baseline audio changed between comparison blocks')
            expected[fixture] = hashes
        elif hashes != expected.get(fixture):
            raise RuntimeError(f'Optimization audio differs from baseline: {folder}')
        record = {'variant': variant, 'fixture': fixture, 'results': str(folder),
                  'median_rtf': summary['median_rtf'],
                  'rtfs': [row['rtf'] for row in summary['measured_runs']],
                  'peak_rss_kib': max(row['peak_rss_kib'] for row in summary['measured_runs']),
                  'cpu_percent': [row['cpu_percent_one_core_100'] for row in summary['measured_runs']],
                  'swap_bytes': summary['cgroup_peak_swap_bytes'], 'audio_sha256': hashes}
        records.append(record)
        (destination / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
        print(f'  RTF={record["median_rtf"]:.6f}, audio matches baseline', flush=True)

    for variant in ('baseline', args.candidate, args.candidate, 'baseline'):
        run(variant, args.fixture, args.iterations)
    if not args.skip_regressions:
        for fixture in ('zh_regression.txt', 'zh_long_regression.txt'):
            run('baseline', fixture, 1)
            run(args.candidate, fixture, 1)
    measurements = {}
    for variant in ('baseline', args.candidate):
        rows = [row for row in records if row['variant'] == variant and row['fixture'] == args.fixture]
        measurements[variant] = {
            'median_rtf': statistics.median(value for row in rows for value in row['rtfs']),
            'peak_rss_gib': max(row['peak_rss_kib'] for row in rows) / 1024 ** 2,
            'max_swap_bytes': max(row['swap_bytes'] for row in rows),
        }
    baseline = measurements['baseline']['median_rtf']
    candidate = measurements[args.candidate]['median_rtf']
    report = {'complete': True, 'order': 'ABBA', 'threads': args.threads, 'warmups_per_block': 1,
              'fixture': args.fixture, 'measured_runs_per_variant': 2 * args.iterations,
              'measurements': measurements,
              'rtf_reduction_percent': (1 - candidate / baseline) * 100,
              'speedup': baseline / candidate, 'audio_bitwise_equal': True,
              'runs': records}
    (destination / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Comparison:', destination / 'comparison.json', flush=True)
    print(json.dumps({key: value for key, value in report.items() if key != 'runs'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
