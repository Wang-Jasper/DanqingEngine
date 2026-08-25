"""
benchmarks/aggregate_report.py
==============================

Cross-platform aggregator that consumes a single output directory produced by
``benchmarks/collect_all.ps1`` and renders the §8 tables / figures described in
``paper.md``.

Usage
-----
    python ./benchmarks/aggregate_report.py benchmarks/out/<TS>

What it does
------------
1. Loads ``env.json``, ``build.json``, ``first_frame.json``, ``SUMMARY_*.csv``,
   ``RAW_*.csv`` from the output directory.
2. Emits ``report.md`` containing the three §8 tables filled in with measured
   numbers.
3. If ``matplotlib`` is available, also renders the figures
   (``fig_8_2_fps_bar.png``, ``fig_8_3_frame_curve.png``).
4. If the optional figures (script overhead / culling) carry no data the
   corresponding plotting step is skipped with a printed NOTE rather than an
   error -- this keeps the script resilient on any machine.

Notes
-----
* This script intentionally has zero non-stdlib hard dependencies; matplotlib
  and pandas are imported lazily so that the markdown report can still be
  produced on a minimal Python install.
* The script is platform-agnostic: it can also consume directories produced by
  a future Linux/macOS port of ``collect_all.ps1``.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return json.loads(raw.decode("utf-8", errors="replace"))


def load_csv(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        return [row for row in reader if row]


def find_one(out_dir: Path, pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(out_dir / pattern)))
    return Path(matches[-1]) if matches else None


# --------------------------------------------------------------------------
# Markdown emission
# --------------------------------------------------------------------------
def fmt(v, digits: int = 2) -> str:
    if v in (None, "", "n/a"):
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def render_table_8_2(summary_rows: list[dict]) -> str:
    head = (
        "| Preset | Runs | fps avg | fps p95 | frame ms avg | "
        "frame ms p95 | GPU ms avg | CPU ms avg | "
        "Draws total | Geometry | LightVolume | Lighting | ImGui | Samples |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = []
    for r in summary_rows:
        body.append(
            "| {preset} | {runs} | {fps_avg} | {fps_p95} | {f_avg} | {f_p95} "
            "| {g_avg} | {c_avg} | {dc} | {dc_geom} | {dc_lv} | {dc_lit} | {dc_gui} | {ns} |".format(
                preset=r.get("preset", "?"),
                runs=r.get("runs", "?"),
                fps_avg=fmt(r.get("fps_avg_mean")),
                fps_p95=fmt(r.get("fps_p95_mean")),
                f_avg=fmt(r.get("frame_ms_avg"), 3),
                f_p95=fmt(r.get("frame_ms_p95"), 3),
                g_avg=fmt(r.get("gpu_ms_avg"), 3),
                c_avg=fmt(r.get("cpu_ms_avg"), 3),
                dc=fmt(r.get("draws_avg"), 1),
                dc_geom=fmt(r.get("geometry_avg"), 1),
                dc_lv=fmt(r.get("light_volume_avg"), 1),
                dc_lit=fmt(r.get("lighting_avg"), 1),
                dc_gui=fmt(r.get("imgui_avg"), 1),
                ns=r.get("samples", "?"),
            )
        )
    return head + "\n".join(body) + "\n"


def render_table_8_3(env: dict, build: dict, first_frame: dict) -> str:
    rows = [
        ("CPU", env.get("cpu_name", "n/a")),
        ("GPU", env.get("gpu_name", "n/a")),
        ("GPU driver", env.get("gpu_driver", "n/a")),
        ("RAM (GB)", env.get("ram_total_gb", "n/a")),
        ("OS", env.get("os_caption", "n/a")),
        ("CMake", env.get("cmake_version", "n/a")),
        ("Build config", env.get("config", "n/a")),
        ("CMake configure (s)", build.get("cmake_configure_sec", "n/a")),
        ("First (cold) build (s)", build.get("first_build_sec", "n/a")),
        ("Incremental build (s)", build.get("incremental_build_sec", "n/a")),
        ("Final exe size (MB)", build.get("exe_size_mb", "n/a")),
        ("Build dir size (MB)", build.get("build_dir_size_mb", "n/a")),
        ("src + shaders (MB)", build.get("src_plus_shaders_mb", "n/a")),
        ("Repo total (MB)", build.get("repo_total_mb", "n/a")),
        ("Cold start to exit (ms, mean of 3)",
         _mean_of_runs(first_frame, "cold_start_to_exit_ms_run")),
        ("Headless full benchmark (ms)",
         first_frame.get("headless_full_run_ms", "n/a")),
    ]
    out = ["| Metric | Value |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    return "\n".join(out) + "\n"


def _mean_of_runs(d: dict, prefix: str) -> str:
    vals = [v for k, v in d.items() if k.startswith(prefix)]
    if not vals:
        return "n/a"
    return fmt(sum(vals) / len(vals), 1)


# --------------------------------------------------------------------------
# Optional plotting (matplotlib lazy import)
# --------------------------------------------------------------------------
def maybe_plot_figures(out_dir: Path,
                       summary_rows: list[dict],
                       raw_rows: list[dict]) -> list[str]:
    notes: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError:
        notes.append("matplotlib not installed; skipping figures.")
        return notes

    if summary_rows:
        # Figure 8.2: fps avg + p95 bar
        presets = [r["preset"] for r in summary_rows]
        avg = [float(r.get("fps_avg_mean") or 0) for r in summary_rows]
        p95 = [float(r.get("fps_p95_mean") or 0) for r in summary_rows]
        x = range(len(presets))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x, avg, label="fps_avg")
        # lower-bar overlay = p95 (lower-tail proxy)
        ax.bar(x, p95, label="fps_p95", alpha=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(presets, rotation=20)
        ax.set_ylabel("fps")
        ax.set_title("Figure 8.2 — fps mean vs p95 across stress presets")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig_8_2_fps_bar.png", dpi=140)
        plt.close(fig)
    else:
        notes.append("SUMMARY missing; skipping fig 8.2.")

    if raw_rows:
        # Figure 8.3: per-run frame_ms_avg vs sample count for Stress1000.
        # We don't have per-frame raw samples (only aggregates), so we plot
        # per-run aggregates as scatter, which is still informative.
        s1k = [r for r in raw_rows if r.get("preset") == "Stress1000"]
        if s1k:
            fig, ax = plt.subplots(figsize=(7, 4))
            xs = list(range(1, len(s1k) + 1))
            ax.plot(xs, [float(r.get("cpu_ms_avg") or 0) for r in s1k],
                    marker="o", label="CPU ms")
            ax.plot(xs, [float(r.get("gpu_ms_avg") or 0) for r in s1k],
                    marker="s", label="GPU ms")
            ax.set_xlabel("repeat #")
            ax.set_ylabel("ms")
            ax.set_title("Figure 8.3 — Stress1000: CPU/GPU per repeat")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "fig_8_3_stress1000_curve.png", dpi=140)
            plt.close(fig)
        else:
            notes.append("Stress1000 absent; skipping fig 8.3.")

    # Figures 8.4 / 8.5 require additional CLI hooks in the engine; once
    # implemented, the corresponding CSVs (script_overhead.csv, culling.csv)
    # can be loaded and plotted here.
    for csv_name, fig_label in [
        ("script_overhead.csv", "fig_8_4_script_overhead.png"),
        ("culling.csv",        "fig_8_5_culling.png"),
    ]:
        p = out_dir / csv_name
        if not p.exists():
            continue
        rows = [r for r in load_csv(p) if r and not list(r.values())[0].startswith("#")]
        if not rows:
            notes.append(f"{csv_name} placeholder only; skipping {fig_label}.")
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        x_key = list(rows[0].keys())[0]
        y_key = "cpu_ms_avg"
        ax.plot([r[x_key] for r in rows],
                [float(r.get(y_key, 0)) for r in rows], marker="o")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(fig_label)
        fig.tight_layout()
        fig.savefig(out_dir / fig_label, dpi=140)
        plt.close(fig)

    return notes


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: aggregate_report.py <out_dir>")
        return 2
    out_dir = Path(argv[1]).resolve()
    if not out_dir.exists():
        print(f"out dir not found: {out_dir}")
        return 1

    env = load_json(out_dir / "env.json")
    build = load_json(out_dir / "build.json")
    first_frame = load_json(out_dir / "first_frame.json")
    summary_path = find_one(out_dir, "SUMMARY_*.csv")
    raw_path = find_one(out_dir, "RAW_*.csv")
    summary_rows = load_csv(summary_path) if summary_path else []
    raw_rows = load_csv(raw_path) if raw_path else []

    md = []
    md.append(f"# Renderer §8 benchmark report — {out_dir.name}\n")
    md.append("## Environment\n")
    md.append("```\n" + json.dumps(env, indent=2, ensure_ascii=False) + "\n```\n")
    md.append("## Table 8.2 — Performance under five stress presets\n")
    md.append(render_table_8_2(summary_rows) if summary_rows
              else "_no SUMMARY file found_\n")
    md.append("\n## Table 8.3 — Build & engine footprint\n")
    md.append(render_table_8_3(env, build, first_frame))

    notes = maybe_plot_figures(out_dir, summary_rows, raw_rows)
    if notes:
        md.append("\n## Notes\n")
        for n in notes:
            md.append(f"- {n}\n")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
