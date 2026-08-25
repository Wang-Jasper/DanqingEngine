Renderer §8 Benchmark Collection
================================

This folder contains everything needed to reproduce the §8 tables and
figures of `paper.md` on any Windows machine in a *single* command.

The scripts are designed to be portable in the sense that:

* Only the standard Windows toolchain is required (PowerShell 5.1+ is
  shipped with every Windows 10/11 install; no `pwsh` is needed).
* All build inputs are pulled by CMake `FetchContent`; no manual SDK
  installs except Vulkan SDK and Visual Studio Build Tools.
* The Python aggregator depends only on the standard library; matplotlib
  is *optional* and only needed if you want the PNG figures.

Quick start
-----------

```powershell
# 0. one-time machine setup (skip if already done):
#    - Install Vulkan SDK (https://vulkan.lunarg.com/)
#    - Install Visual Studio 2022 with "Desktop development with C++"
#    - Install CMake >= 3.20 (https://cmake.org/download/)
#    - (optional) Install Python 3.9+ with `pip install matplotlib`

git clone <repo>
cd Renderer

# 1. one-shot: build + benchmark + collect
powershell -ExecutionPolicy Bypass -File benchmarks\collect_all.ps1

# 2. (optional) render the markdown report + PNG figures
python benchmarks\aggregate_report.py benchmarks\out\<TIMESTAMP>
```

That produces, under `benchmarks/out/<TIMESTAMP>/`:

| File                       | Purpose                                          |
|----------------------------|--------------------------------------------------|
| `env.json`                 | Host fingerprint (CPU/GPU/driver/CMake/git SHA). |
| `build.json`               | Cold + incremental build time, binary size.      |
| `first_frame.json`         | Cold-start latency, full headless run wall time. |
| `bench_csv/*.csv`          | Raw per-run CSVs straight from `BenchmarkRunner`.|
| `RAW_*.csv`                | Flat join of every run with `preset, repeat`.    |
| `SUMMARY_*.csv`            | Per-preset aggregates over `-Repeats` runs.      |
| `script_overhead.csv`      | Placeholder for figure 8.4 (see TODO below).     |
| `culling.csv`              | Placeholder for figure 8.5 (see TODO below).     |
| `report.md`                | Filled-in tables 8.2 and 8.3.                    |
| `fig_8_2_*.png`            | Figure 8.2 — fps bar across presets.             |
| `fig_8_3_*.png`            | Figure 8.3 — Stress1000 per-run CPU/GPU curve.   |

Common flags
------------

```powershell
# Re-run benchmarks only (don't repeat the multi-minute cold build)
benchmarks\collect_all.ps1 -SkipBuild

# Smaller / quicker pass for sanity checking
benchmarks\collect_all.ps1 -Repeats 1 -MaxFrames 1500

# Release build (paper §8 baseline is Debug; switch only for ablation)
benchmarks\collect_all.ps1 -Config Release

# Custom subset
benchmarks\collect_all.ps1 -Presets Stress1000,LightStress
```

Sampling protocol
-----------------

Hard-coded into `src/renderer/BenchmarkRunner.h` and **identical across
machines**:

* 5 s warmup (timestamps still run, samples discarded)
* 10 s sampling, every frame contributes one sample
* Window: 1280×720, vsync off, ImGui drawn as in editor mode
* CSV column names match exactly what `BenchmarkRunner::writeCsv` emits.

So fps / frame-time numbers from any two machines are directly
comparable — only the host fingerprint in `env.json` differs.

What still requires engine changes
----------------------------------

Two figures in §8 cannot be filled in automatically from a single
command-line run because the engine currently lacks the corresponding CLI
hooks; the scripts therefore emit *placeholder* CSVs with a TODO header:

* **Figure 8.4 — script overhead**: needs `--script-stress N` in
  `src/main.cpp` to spawn N entities each carrying a no-op
  `ScriptComponent`. Once implemented, extend `collect_all.ps1`'s
  `script_overhead.csv` block to drive `N ∈ {0, 1, 10, 50, 100}`.
* **Figure 8.5 — culling strategy**: needs `--culling
  {none|frustum|bvh}` plus `--entity-stress M` in `src/main.cpp`. The
  underlying flags (`frustumCullingEnabled`, BVH branch in
  `DeferredRenderer.cpp`) already exist, only the CLI surface is
  missing.

Adding both flags is mechanical (≤ 30 lines in `main.cpp` + the
corresponding setters on `DeferredRenderer`); we leave the empty CSV
schema in place so that, the day those CLI hooks land, no further
changes to the collection scripts are required.

Linux / macOS port
------------------

Porting `collect_all.ps1` and `measure_build.ps1` to bash is mechanical:
the only platform-specific calls are `Get-CimInstance` (read host
fingerprint), `Start-Process` (timeout supervision) and PowerShell
transcription. The Python aggregator already works as-is on every
platform.
