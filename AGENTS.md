# Repository Guidelines

## Project Goal & Current Scope

This repository improves x86 CPU TTS inference, currently Ming-3.0-TTS 0.5B. RTF is the sole optimization metric. Preserve valid synthesis and constrain CPU and memory use on the TrueNAS VM.

## Project Structure & Module Organization

Baseline source stays unmodified. Profiling and CPU optimization use separate worktrees:

- `benchmarks/`: C++ harness, Python resource monitor, timing patch, CMake configuration, and text fixtures.
- `benchmarks/results/`: per-run metadata, synthesis timings, resource samples, and summaries.
- `src/patches/`: replayable optimization patches; `tests/`: numerical parity tests.
- `.cache/audio.cpp/`: upstream source; `.cache/audio.cpp-profile/` and `.cache/audio.cpp-ming-opt/`: experimental worktrees.
- `.cache/bench-build/`: generated binaries and build files.
- `models/`: downloaded GGUF weights, excluded from version control.

Keep weights, generated audio, and build output out of version control. Setup instructions belong in `README.md`.

## Build, Test, and Development Commands

Run from the repository root:

- `cmake -S benchmarks -B .cache/bench-build -DCMAKE_BUILD_TYPE=Release`: configure the CPU baseline.
- `cmake --build .cache/bench-build --target ming3_rtf -j 2`: build the measurement harness.
- `python3 benchmarks/run_ming3.py --threads 2 --warmups 1 --iterations 3`: run the resource-limited benchmark.
- `python3 benchmarks/run_ming3.py --variant cpu-opt --threads 4`: benchmark the optimized CPU path.
- `python3 benchmarks/compare_ming_cpu.py --fixture zh_paragraph.txt --iterations 2 --skip-regressions`: alternate baseline and optimized long-text runs.

See `README.md` for worktree setup, build limits, and profiling. Never derive worker counts from the VM's reported 128 logical CPUs.

## Coding Style & Naming Conventions

Use four spaces in C++ and Python. Keep orchestration separate from inference changes. Gate ISA-specific kernels by CPU features and retain a fallback. Avoid thread oversubscription.

## Testing & RTF Benchmarks

CTest's `ming3_cpu_rope_parity` checks bitwise numerical parity across shapes and repeated executions. No coverage threshold exists. Benchmarks reject invalid audio.

Calculate `RTF = synthesis wall-clock seconds / generated audio seconds`; derive audio duration from sample count and sample rate.

Exclude loading, warmup, and WAV writes. Keep hardware, weights, text, decoding settings, threads, and affinity identical. Alternate baseline and candidate runs and report median RTF. Validate stage sums and audio hashes. Test both short text and automatically chunked paragraphs. Preserve baselines and record commands, CPU details, runtime versions, revisions, and patches.

## Commit & Pull Request Guidelines

No Git history establishes commit conventions. Use focused, imperative subjects, such as `Reduce vocoder RTF`. Pull requests should explain the change, link relevant issues, list validation commands, and include reproducible baseline-versus-candidate RTF results for performance claims.
