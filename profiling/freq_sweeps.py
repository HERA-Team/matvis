#!/usr/bin/env python
"""Run and tabulate the sweeps behind the frequency-restructuring question.

``matvis profile`` measures one configuration. These sweeps vary the axes that
decide whether restructuring the per-frequency loop is worth doing -- number of
channels, number of unique beams, integrations per run, the beam's own channel
count, and the coordinate-rotation method -- and print the derived quantities
side by side.

Usage::

    profiling/freq_sweeps.py run amortisation      # or: beams bcrs prod cpu all
    profiling/freq_sweeps.py report                # tabulate whatever has run

Results land in ``--outdir`` as one ``sweep-<name>.json`` per sweep, alongside
the per-run ``summary-stats-*.json`` that ``matvis profile`` writes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

DEFAULT_OUTDIR = Path(__file__).parent / "results"


@lru_cache(maxsize=1)
def _cli_options() -> str:
    """The installed `matvis profile` help text, for feature detection."""
    return subprocess.run(
        ["matvis", "profile", "--help"], capture_output=True, text=True
    ).stdout


def spline_order(order: int) -> list[str]:
    """``--spline-order`` if this build has it, nothing if it does not.

    Cubic gridded-beam interpolation is a newer option; without it there is one
    interpolation order and the sweep collapses to a single row rather than
    failing.
    """
    if "--spline-order" not in _cli_options():
        return []
    return ["--spline-order", str(order)]


GPU = [
    "--gpu",
    "--interpolated-beam",
    "--single-precision",
    "--gpu-event-timing",
    "--coord-method",
    "CoordinateRotationERFA",
]
CPU = [
    "--cpu",
    "--interpolated-beam",
    "--single-precision",
    "--coord-method",
    "CoordinateRotationERFA",
]
# Small enough to iterate on; large enough that overheads are not the story.
DEV = ["-a", "64", "-b", "64", "-s", "200000"]
# The production slice, as defined on the Performance docs page.
PROD = ["-a", "350", "-b", "350", "-s", "1000000", "--nchunks", "30"]


def run_one(outdir: Path, tag: str, args: list[str], timeout=7200) -> dict | None:
    """Run one `matvis profile` config and return its JSON summary."""
    outdir.mkdir(parents=True, exist_ok=True)
    known = {p.name for p in outdir.glob("summary-stats-*.json")}
    print(f"\n=== {tag}\n    {' '.join(args)}", flush=True)
    proc = subprocess.run(
        ["matvis", "profile", *args, "-o", str(outdir)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:]
        print("    FAILED:", *tail, sep="\n      ", flush=True)
        return None

    new = {p.name for p in outdir.glob("summary-stats-*.json")} - known
    if new:
        path = outdir / sorted(new)[0]
    else:
        # The label is deterministic, so a repeated config overwrites its file.
        path = max(outdir.glob("summary-stats-*.json"), key=lambda p: p.stat().st_mtime)
    data = json.loads(path.read_text())
    data["_tag"] = tag
    data["_args"] = args
    return data


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------


def sweep_amortisation(outdir):
    """Nfreq x ntimes: how much of a run is setup that a restructure could share."""
    out = []
    for nfreq in (1, 2, 4):
        for ntimes in (2, 8, 32):
            r = run_one(
                outdir,
                f"amort_nf{nfreq}_nt{ntimes}",
                [*GPU, *DEV, "-f", str(nfreq), "-t", str(ntimes), "--repeat", "3"],
            )
            if r:
                out.append(r)
    return out


def sweep_beams(outdir):
    """Nbeams x spline order x beam channel count: where setup actually goes."""
    out = []
    orders = (1, 3) if "--spline-order" in _cli_options() else (1,)
    for nbeams in (1, 16, 64):
        for order in orders:
            for beam_nfreq in (0, 16):
                r = run_one(
                    outdir,
                    f"beams_nb{nbeams}_o{order}_bf{beam_nfreq}",
                    [
                        *GPU,
                        "-a",
                        "64",
                        "-s",
                        "200000",
                        "-b",
                        str(nbeams),
                        *spline_order(order),
                        "--beam-nfreq",
                        str(beam_nfreq),
                        "-f",
                        "2",
                        "-t",
                        "8",
                        "--repeat",
                        "3",
                    ],
                )
                if r:
                    out.append(r)
    return out


def sweep_bcrs(outdir):
    """Coordinate-rotation method and BCRS refresh interval."""
    out = []
    cases = [
        ("CoordinateRotationERFA", 0.0),
        ("CoordinateRotationERFA", 180.0),
        ("CoordinateRotationERFA", 1e9),
        ("GPUCoordinateRotationERFA", 0.0),
        ("GPUCoordinateRotationERFA", 1e9),
        ("CoordinateRotationAstropy", None),
    ]
    for method, ubce in cases:
        args = [
            "--gpu",
            "--interpolated-beam",
            "--single-precision",
            "--gpu-event-timing",
            "--coord-method",
            method,
            *DEV,
            "-f",
            "1",
            "-t",
            "8",
            "--repeat",
            "3",
        ]
        if ubce is not None:
            args += ["--update-bcrs-every", repr(ubce)]
        r = run_one(outdir, f"bcrs_{method}_{ubce}", args)
        if r:
            out.append(r)
    return out


def sweep_prod(outdir):
    """The production slice, at both spline orders and nfreq in {1, 2}."""
    out = []
    orders = (1, 3) if "--spline-order" in _cli_options() else (1,)
    for nfreq in (1, 2):
        for order in orders:
            r = run_one(
                outdir,
                f"prod_nf{nfreq}_o{order}",
                [
                    *GPU,
                    *PROD,
                    *spline_order(order),
                    "-f",
                    str(nfreq),
                    "-t",
                    "4",
                    "--repeat",
                    "3",
                ],
            )
            if r:
                out.append(r)
    return out


def sweep_cpu(outdir):
    """The CPU backend, whose per-stage timings are unambiguous."""
    out = []
    base = ["-a", "64", "-b", "64", "-s", "50000"]
    for nfreq in (1, 2):
        for ntimes in (2, 8):
            r = run_one(
                outdir,
                f"cpu_nf{nfreq}_nt{ntimes}",
                [*CPU, *base, "-f", str(nfreq), "-t", str(ntimes), "--repeat", "2"],
            )
            if r:
                out.append(r)
    for ubce in (0.0, 1e9):
        r = run_one(
            outdir,
            f"cpu_bcrs_{ubce}",
            [
                *CPU,
                *base,
                "-f",
                "1",
                "-t",
                "8",
                "--update-bcrs-every",
                repr(ubce),
                "--repeat",
                "2",
            ],
        )
        if r:
            out.append(r)
    return out


SWEEPS = {
    "amortisation": sweep_amortisation,
    "beams": sweep_beams,
    "bcrs": sweep_bcrs,
    "prod": sweep_prod,
    "cpu": sweep_cpu,
}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _d(row, key, default=float("nan")):
    return row["derived"].get(key, default)


def _ev(row, stage):
    return row["derived"].get("event_timing_ms", {}).get(stage, float("nan"))


def report_gpu_table(rows, title):
    """Print the per-stage GPU table for one sweep."""
    print(f"\n## {title}")
    print(
        f"{'tag':>28} {'wall/int s':>11} {'gpu/int s':>10} {'rot+ovh ms':>11} "
        f"{'chunk ms':>9} {'beam ms':>8} {'tau ms':>7} {'z ms':>6} "
        f"{'matprod ms':>11} {'peak GB':>8} {'spread':>7}"
    )
    for r in rows:
        wall, gput = (
            _d(r, "steady_wall_per_integration"),
            _d(r, "gpu_time_per_integration"),
        )
        print(
            f"{r['_tag']:>28} {wall:11.4f} {gput:10.4f} {(wall - gput) * 1e3:11.2f} "
            f"{_ev(r, 'chunk_total'):9.2f} {_ev(r, 'beam'):8.2f} {_ev(r, 'tau'):7.2f} "
            f"{_ev(r, 'z'):6.2f} {_ev(r, 'matprod'):11.2f} "
            f"{_d(r, 'peak_device_bytes', 0) / 1024**3:8.2f} "
            f"{_d(r, 'repeat_wall_spread_frac', 0):7.1%}"
        )
    print(
        "\n  rot+ovh is wall minus the summed chunk events: coords.rotate() plus\n"
        "  host-side loop overhead, i.e. an UPPER bound on the rotation cost."
    )


def report_setup(rows):
    """Print the per-frequency setup split for one sweep."""
    print("\n   setup, per frequency:")
    print(
        f"{'tag':>28} {'total s':>9} {'indep s':>9} {'dep s':>9} "
        f"{'beam interp s':>14} {'hoist saving':>13} {'nchunks':>9}"
    )
    for r in rows:
        per_freq = r.get("run_stats_per_freq") or []
        binterp = sum(
            s["setup_breakdown"].get("beam_wrangle_freq_dependent", 0.0)
            for s in per_freq
        ) / max(len(per_freq), 1)
        chunks = sorted({s.get("nchunks") for s in per_freq})
        print(
            f"{r['_tag']:>28} {_d(r, 'setup_time_per_freq'):9.3f} "
            f"{_d(r, 'setup_freq_independent_per_freq'):9.3f} "
            f"{_d(r, 'setup_freq_dependent_per_freq'):9.3f} {binterp:14.3f} "
            f"{_d(r, 'projected_setup_hoist_saving'):13.2%} "
            f"{','.join(str(c) for c in chunks):>9}"
        )
    print(
        "\n  A chunk count that varies between channels of one run means the\n"
        "  per-chunk timings of that run are not comparable."
    )


def report_cpu(rows):
    """Print the CPU backend's host-timed stage split."""
    print("\n## CPU backend: host-timed stages per integration")
    for r in rows:
        stages = r["derived"].get("stage_seconds_per_integration", {})
        if not stages:
            continue
        total = sum(stages.values())
        cfg = r["config"]
        print(
            f"\n{r['_tag']}  (nfreq={cfg['nfreq']} ntimes={cfg['ntimes']} "
            f"update_bcrs_every={cfg.get('update_bcrs_every')})"
        )
        for k, v in stages.items():
            print(f"   {k:>13}: {v * 1e3:9.3f} ms ({v / total:5.1%})")
        shared = stages.get("rotate", 0) + stages.get("select_chunk", 0)
        print(f"   -> frequency-independent stages: {shared / total:.1%} of the loop")


def do_report(outdir: Path):
    """Tabulate every sweep whose JSON is present in `outdir`."""
    any_found = False
    for name in SWEEPS:
        path = outdir / f"sweep-{name}.json"
        if not path.exists():
            continue
        any_found = True
        rows = json.loads(path.read_text())
        if name == "cpu":
            report_cpu(rows)
            report_setup(rows)
        else:
            report_gpu_table(rows, name)
            report_setup(rows)
    if not any_found:
        print(f"No sweep-*.json found in {outdir}; run a sweep first.")


def main():
    """Parse arguments and either run sweeps or report them."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one or more sweeps")
    r.add_argument("which", choices=[*SWEEPS, "all"])
    r.add_argument("--outdir", default=str(DEFAULT_OUTDIR))

    p = sub.add_parser("report", help="tabulate sweeps already run")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))

    a = ap.parse_args()
    outdir = Path(a.outdir)

    if a.cmd == "report":
        do_report(outdir)
        return

    for name in list(SWEEPS) if a.which == "all" else [a.which]:
        rows = SWEEPS[name](outdir)
        (outdir / f"sweep-{name}.json").write_text(json.dumps(rows, indent=1))
        print(f"\n--> {outdir}/sweep-{name}.json ({len(rows)} runs)")
    do_report(outdir)


if __name__ == "__main__":
    sys.exit(main())
