"""Run warmed CPU-only synthesis inside a resource-limited systemd user service."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    'ming3_tts': {
        'filename': 'ming-3.0-tts-0.5b-q8_0.gguf', 'size': 1742280896,
        'sha256': 'fcf2aa25e8e9cf2065a936e506581735c64d6255fb7869f09fbb61a2dde09aa1',
        'source': 'https://modelscope.cn/models/audio-cpp/Ming-3.0-TTS-0.5B-GGUF',
        'label': 'ming3-q8',
    },
    'qwen3_tts': {
        'filename': 'qwen3-tts-12hz-0.6b-base-q8_0.gguf', 'size': 1991217728,
        'sha256': 'c7d0e6c9485a25ae1f6a4c0796f0023230a0051c44cbd2bd05d73770f2cce637',
        'source': 'https://modelscope.cn/models/HereIsMark/audio.cpp-gguf',
        'label': 'qwen3-0.6b-base-q8',
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def capture(*command):
    return subprocess.check_output(command, text=True, cwd=ROOT).strip()


def key_values(path):
    try:
        return {line.split()[0].rstrip(':'): int(line.split()[1])
                for line in path.read_text().splitlines() if len(line.split()) >= 2}
    except OSError:
        return {}


def guest_cpu_ticks():
    # Linux /proc/stat includes guest time in user/nice; keep raw counters so
    # steal time can be inspected without double counting or inferring host load.
    fields = ('user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'guest', 'guest_nice')
    values = Path('/proc/stat').read_text().splitlines()[0].split()[1:]
    return dict(zip(fields, map(int, values)))


def main(default_family='ming3_tts'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--memory-gib', type=int, default=6)
    parser.add_argument('--profile', action='store_true', help='Use the stage-timed Ming binary')
    parser.add_argument('--family', choices=MODELS, default=default_family)
    parser.add_argument('--variant', choices=('baseline', 'cpu-opt', 'engine', 'tuned', 'fast'), default='baseline')
    parser.add_argument('--flow-steps', type=int, choices=range(5, 11), default=10)
    parser.add_argument('--vae-cache-mib', type=int, default=0)
    parser.add_argument('--cfg-interval', type=int, choices=range(1, 11), default=1)
    parser.add_argument('--cfg-reuse-threshold', type=float, default=0.0)
    parser.add_argument('--packed-q8', action='store_true')
    parser.add_argument('--preserve-noise-stream', action='store_true')
    parser.add_argument('--cfg-reuse-mode', choices=('delta', 'unconditional'), default='delta')
    parser.add_argument('--flow-quant', choices=('q8', 'q4g64', 'q4g128'), default='q4g64')
    parser.add_argument('--agg-quant', choices=('q8', 'q4g64', 'q4g128'), default=None)
    parser.add_argument('--ar-quant', choices=('q8', 'f16', 'q4_0', 'iq4_nl'), default='q4_0')
    parser.add_argument('--vae-quant', choices=('q8', 'f16', 'q4_0', 'iq4_nl'), default='q4_0')
    parser.add_argument('--fused-fp32-act', choices=('auto', '0', '1'), default='auto')
    parser.add_argument('--reuse-anchor-latent', choices=('0', '1'), default='1')
    parser.add_argument('--template-name', default='')
    parser.add_argument('--instruct', default='')
    parser.add_argument('--voice-id', default='')
    parser.add_argument('--text-file', type=Path, default=ROOT / 'benchmarks/zh_short.txt')
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or min(args.warmups, args.iterations, args.memory_gib) < 1:
        parser.error('Use 1–4 threads and positive warmups, iterations and memory limit')
    if args.profile and args.family != 'ming3_tts':
        parser.error('--profile currently supports Ming only')
    if args.variant != 'baseline' and (args.family != 'ming3_tts' or (args.profile and args.variant not in ('engine', 'tuned', 'fast'))):
        parser.error('CPU variants support Ming only; profiling supports baseline, engine, tuned and fast')
    if not 0 <= args.vae_cache_mib <= 1536 or not 0.0 <= args.cfg_reuse_threshold <= 2.0:
        parser.error('Use 0–1536 MiB VAE cache and 0–2 CFG reuse threshold')
    if args.variant not in ('tuned', 'fast') and (args.flow_steps != 10 or args.cfg_interval != 1):
        parser.error('--flow-steps and --cfg-interval require --variant tuned or fast')
    if args.variant != 'tuned' and (args.vae_cache_mib or args.cfg_reuse_threshold or args.packed_q8 or
                                    args.preserve_noise_stream or args.cfg_reuse_mode != 'delta'):
        parser.error('Sampling/cache experiment settings require --variant tuned')
    binary = ROOT / ('.cache/bench-build/ming3_profile_rtf' if args.profile
                     else '.cache/bench-build/ming3_rtf')
    if args.variant == 'cpu-opt':
        binary = ROOT / '.cache/bench-build/ming3_cpu_opt_rtf'
    if args.variant == 'engine':
        binary = ROOT / ('.cache/bench-build/ming3_engine_profile_rtf' if args.profile
                         else '.cache/bench-build/ming3_engine_rtf')
    if args.variant == 'tuned':
        binary = ROOT / ('.cache/bench-build/ming3_tuned_profile_rtf' if args.profile
                         else '.cache/bench-build/ming3_tuned_rtf')
    if args.variant == 'fast':
        binary = ROOT / ('.cache/bench-build/ming3_fast_profile_rtf' if args.profile
                         else '.cache/bench-build/ming3_fast_rtf')
    model_config = MODELS[args.family]
    model = ROOT / 'models' / model_config['filename']
    text = args.text_file.resolve()
    if not binary.is_file() or not model.is_file() or not text.is_file():
        parser.error('Build the benchmark binary and download the Q8 model first')
    if model.stat().st_size != model_config['size'] or file_sha256(model) != model_config['sha256']:
        parser.error('Model size or SHA-256 does not match the selected checkpoint')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    variant = model_config['label'] + ('-profile' if args.profile else '')
    if args.variant != 'baseline':
        variant += '-' + args.variant
    output = ROOT / f'benchmarks/results/{variant}-t{args.threads}-{stamp}'
    output.mkdir(parents=True)
    unit = f'tts-{args.family}-t{args.threads}-{stamp.lower()}-{os.getpid()}'
    inference = [str(binary), str(model), str(args.threads), str(text),
                 str(output / 'audio'), str(args.warmups), str(args.iterations)]
    reference = ROOT / '.cache/audio.cpp/webui/native/demo_voices/demo_2_man.wav'
    reference_text = ROOT / 'benchmarks/qwen_reference.txt'
    if args.family == 'qwen3_tts':
        inference.extend([args.family, str(reference), str(reference_text)])
    environment = [f'OMP_NUM_THREADS={args.threads}', f'OMP_THREAD_LIMIT={args.threads}',
                   'OMP_DYNAMIC=FALSE', f'OPENBLAS_NUM_THREADS={args.threads}',
                   f'MKL_NUM_THREADS={args.threads}']
    if args.template_name:
        environment.append(f'MING_TEMPLATE_NAME={args.template_name}')
    if args.instruct:
        environment.append(f'MING_INSTRUCT={args.instruct}')
    if args.voice_id:
        environment.append(f'MING_VOICE_ID={args.voice_id}')
    if args.variant == 'tuned':
        environment.extend([f'MING_FLOW_STEPS={args.flow_steps}',
                            f'MING_VAE_CACHE_MIB={args.vae_cache_mib}',
                            f'MING_CFG_INTERVAL={args.cfg_interval}',
                            f'MING_CFG_REUSE_THRESHOLD={args.cfg_reuse_threshold}',
                            f'MING_PACKED_Q8={int(args.packed_q8)}',
                            f'MING_PRESERVE_NOISE_STREAM={int(args.preserve_noise_stream)}',
                            f'MING_CFG_REUSE_MODE={args.cfg_reuse_mode}'])
    if args.variant == 'fast':
        agg_quant = args.agg_quant or args.flow_quant
        environment.extend([f'MING_FLOW_QUANT={args.flow_quant}',
                            f'MING_AGG_QUANT={agg_quant}',
                            f'MING_AR_QUANT={args.ar_quant}',
                            f'MING_VAE_QUANT={args.vae_quant}',
                            f'MING_FLOW_STEPS={args.flow_steps}',
                            f'MING_CFG_INTERVAL={args.cfg_interval}',
                            f'MING_REUSE_ANCHOR_LATENT={args.reuse_anchor_latent}'])
        if args.fused_fp32_act != 'auto':
            environment.append(f'MING_FUSED_FP32_ACT={args.fused_fp32_act}')
    command = ['systemd-run', '--user', '--wait', '--pipe', '--collect',
               f'--unit={unit}', f'--working-directory={ROOT / ".cache/audio.cpp"}',
               '-p', f'MemoryMax={args.memory_gib}G', '-p', 'MemorySwapMax=0',
               '-p', f'CPUQuota={args.threads * 100}%', '-p', 'TasksMax=256',
               '-p', 'Nice=10', '-p', 'RuntimeMaxSec=20min',
               '/usr/bin/env', *environment, *inference]
    metadata = {
        'started_utc': stamp, 'configuration': {**vars(args), 'text_file': str(text)}, 'command': command,
        'text': text.read_text().strip(), 'language': 'zh', 'seed': 1895,
        'task': 'tts', 'mode': 'offline', 'backend': 'cpu',
        'other_generation_options': 'upstream defaults',
        'timing_boundary': 'session->run; excludes loading, warmup and WAV writing',
        'rtf_definition': 'synthesis wall-clock seconds / actual generated audio seconds',
        'family': args.family,
        'model_source': model_config['source'],
        'model_size_bytes': model.stat().st_size,
        'verified_model_sha256': model_config['sha256'],
        'upstream_commit': capture('git', '-C', str(ROOT / '.cache/audio.cpp'), 'rev-parse', 'HEAD'),
        'upstream_worktree_changes': capture('git', '-C', str(ROOT / '.cache/audio.cpp'), 'status', '--short'),
        'lscpu': json.loads(capture('lscpu', '--json')),
        'allowed_logical_cpus': sorted(os.sched_getaffinity(0)),
        'compiler': capture('g++', '--version').splitlines()[0],
        'ggml_cpu_compile_flags': (ROOT / '.cache/bench-build/audiocpp/ggml/src/CMakeFiles/ggml-cpu.dir/flags.make').read_text(),
        'kernel': capture('uname', '-sr'),
        'virtualization': subprocess.run(['systemd-detect-virt'], text=True, capture_output=True).stdout.strip(),
        'guest_cpu_ticks_before': guest_cpu_ticks(),
        'meminfo_before_kib': key_values(Path('/proc/meminfo')),
        'benchmark_binary_sha256': file_sha256(binary),
    }
    if args.family == 'qwen3_tts':
        metadata['variant'] = '0.6B Base, reference-conditioned voice cloning'
        metadata['reference_audio'] = str(reference)
        metadata['reference_audio_sha256'] = file_sha256(reference)
        metadata['reference_text'] = reference_text.read_text().strip()
        metadata['reference_cache'] = 'default; identical reference reused for warmup and measured runs'
    if args.profile:
        metadata['stage_timer_header'] = (ROOT / 'benchmarks/tts_stage_timer.h').read_text()
        stage_patch = ROOT / 'benchmarks/ming3_stage_timing.patch'
        if stage_patch.is_file():
            metadata['stage_timing_patch'] = stage_patch.read_text()
        if (ROOT / '.cache/audio.cpp-profile').is_dir():
            metadata['profile_worktree_changes'] = capture(
                'git', '-C', str(ROOT / '.cache/audio.cpp-profile'), 'diff')
    if args.variant == 'cpu-opt':
        metadata['cpu_optimization_patch'] = capture(
            'git', '-C', str(ROOT / '.cache/audio.cpp-ming-opt'), 'diff')
    if args.variant in ('engine', 'tuned', 'fast'):
        worktree = {'engine': 'audio.cpp-ming-engine', 'tuned': 'audio.cpp-ming-tune',
                    'fast': 'audio.cpp-ming-fast'}[args.variant]
        metadata['engine_patch'] = capture(
            'git', '-C', str(ROOT / '.cache' / worktree), 'diff')
        source_files = ['kernels.cpp', 'kernels.h']
        if args.variant == 'tuned':
            source_files.extend(['trial.cpp', 'trial.h', 'packed_q8.cpp', 'packed_q8.h'])
        if args.variant == 'fast':
            source_files.extend(['flow_engine.cpp', 'flow_engine.h', 'repack_weights.cpp', 'repack_weights.h'])
        metadata['engine_sources'] = {
            str(path.relative_to(ROOT)): {'sha256': file_sha256(path), 'source': path.read_text()}
            for path in (ROOT / 'src/ming_cpu' / name for name in source_files)}
        metadata['engine_fft_worker_limit'] = 4
        metadata['engine_build_configuration'] = (ROOT / 'benchmarks/CMakeLists.txt').read_text()
        metadata['engine_kernel_compile_flags'] = (
            ROOT / '.cache/bench-build/CMakeFiles/ming_cpu_kernels.dir/flags.make').read_text()
    if args.variant == 'fast':
        agg_quant = args.agg_quant or args.flow_quant
        metadata['fast_engine'] = {
            'flow_quant': args.flow_quant, 'agg_quant': agg_quant,
            'ar_quant': args.ar_quant, 'vae_quant': args.vae_quant,
            'flow_steps': args.flow_steps, 'cfg_interval': args.cfg_interval,
            'reuse_anchor_latent': args.reuse_anchor_latent,
        }
    if args.variant == 'tuned':
        metadata['other_generation_options'] = 'upstream defaults except sampling_experiment'
        metadata['sampling_experiment'] = {
            'flow_steps': args.flow_steps, 'cfg_interval': args.cfg_interval,
            'cfg_reuse_threshold': args.cfg_reuse_threshold,
            'requires_audio_quality_evaluation': args.flow_steps != 10 or args.cfg_interval != 1 or
                                                 args.cfg_reuse_threshold != 0.0,
            'vae_cache_budget_mib': args.vae_cache_mib,
            'packed_q8': args.packed_q8,
            'preserve_noise_stream': args.preserve_noise_stream,
            'cfg_reuse_mode': args.cfg_reuse_mode,
            'rng_policy': ('N SDE frames, advance as 10 steps to keep patch-index initial noise aligned'
                           if args.preserve_noise_stream else
                           'N SDE frames and N-frame RNG advance, matching official flow_steps behavior'),
        }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(f'Results: {output}', flush=True)
    peak_memory = 0
    peak_swap = 0
    last_cpu_stats = {}
    last_memory_events = {}
    abort_reason = None
    low_memory_samples = 0
    group = None
    started = time.monotonic()
    with (output / 'events.jsonl').open('w') as events, \
            (output / 'runtime.log').open('w') as log, \
            (output / 'resources.csv').open('w') as samples:
        process = subprocess.Popen(command, stdout=events, stderr=log, cwd=ROOT)
        writer = csv.writer(samples)
        writer.writerow(['elapsed_s', 'cgroup_memory_bytes', 'cgroup_peak_bytes',
                         'cgroup_swap_bytes', 'cpu_usage_usec', 'mem_available_kib', 'load1',
                         'tasks_current', 'tasks_peak', 'guest_steal_ticks_total'])
        try:
            while process.poll() is None:
                if group is None:
                    lookup = subprocess.run(
                        ['systemctl', '--user', 'show', unit + '.service',
                         '--property=ControlGroup', '--value'],
                        text=True, capture_output=True, check=False)
                    if lookup.returncode == 0 and lookup.stdout.strip():
                        group = Path('/sys/fs/cgroup') / lookup.stdout.strip().lstrip('/')
                memory = {}
                if group is not None:
                    for name in ('memory.current', 'memory.peak', 'memory.swap.current',
                                 'pids.current', 'pids.peak'):
                        try:
                            memory[name] = int((group / name).read_text())
                        except OSError:
                            pass
                    last_cpu_stats = key_values(group / 'cpu.stat') or last_cpu_stats
                    last_memory_events = key_values(group / 'memory.events') or last_memory_events
                peak_memory = max(peak_memory, memory.get('memory.peak', 0))
                peak_swap = max(peak_swap, memory.get('memory.swap.current', 0))
                available = key_values(Path('/proc/meminfo')).get('MemAvailable', 0)
                writer.writerow([round(time.monotonic() - started, 3),
                                 memory.get('memory.current'), memory.get('memory.peak'),
                                 memory.get('memory.swap.current'), last_cpu_stats.get('usage_usec'),
                                 available, Path('/proc/loadavg').read_text().split()[0],
                                 memory.get('pids.current'), memory.get('pids.peak'), guest_cpu_ticks().get('steal')])
                samples.flush()
                low_memory_samples = low_memory_samples + 1 if available < 4 * 1024 * 1024 else 0
                if low_memory_samples >= 3:
                    abort_reason = 'System available memory stayed below 4 GiB for three samples'
                    subprocess.run(['systemctl', '--user', 'stop', unit + '.service'], check=False)
                time.sleep(1)
        except BaseException:
            subprocess.run(['systemctl', '--user', 'stop', unit + '.service'], check=False)
            process.wait()
            raise
        returncode = process.wait()
    records = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()
               if line.startswith('{')]
    measured = [record for record in records if record.get('phase') == 'measured']
    complete = returncode == 0 and len(measured) == args.iterations
    summary = {'complete': complete, 'returncode': returncode, 'abort_reason': abort_reason,
               'measured_runs': measured, 'cgroup_peak_memory_bytes': peak_memory,
               'cgroup_peak_swap_bytes': peak_swap, 'cgroup_cpu_stats_last_sample': last_cpu_stats,
               'cgroup_memory_events_last_sample': last_memory_events,
               'meminfo_after_kib': key_values(Path('/proc/meminfo')),
               'guest_cpu_ticks_after': guest_cpu_ticks()}
    if complete:
        summary['median_rtf'] = statistics.median(item['rtf'] for item in measured)
        summary['aggregate_rtf'] = (sum(item['synthesis_s'] for item in measured) /
                                    sum(item['audio_s'] for item in measured))
        if args.profile:
            total_s = sum(item['synthesis_s'] for item in measured)
            total_audio_s = sum(item['audio_s'] for item in measured)
            stage_names = measured[0]['stages'].keys()
            totals = {name: sum(item['stages'][name]['exclusive_s'] for item in measured)
                      for name in stage_names}
            summary['stages'] = {
                name: {'mean_exclusive_s': seconds / len(measured),
                       'share_percent': seconds / total_s * 100,
                       'rtf_contribution': seconds / total_audio_s,
                       'calls_per_run': [item['stages'][name]['calls'] for item in measured]}
                for name, seconds in totals.items()}
            summary['stage_unaccounted_mean_s'] = (total_s - sum(totals.values())) / len(measured)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    if not complete:
        raise SystemExit('Benchmark incomplete; inspect runtime.log and partial events')


if __name__ == '__main__':
    main()
