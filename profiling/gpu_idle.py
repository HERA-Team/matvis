#!/usr/bin/env python
"""Report how much of a matvis run the GPU spends *idle*, and where.

The CUDA-event timings collected by ``matvis profile --gpu-event-timing``
measure spans *on the stream*: an event recorded before a stage and one after
it bracket everything that happened in between, including any time the GPU sat
idle waiting for the host to enqueue more work. That makes them useless for
telling "the device needs 60 ms for this stage" apart from "the device needs
40 ms and then waits 20 ms for the host".

This script closes that gap using an nsys trace. It takes the union of all
kernel and memcpy intervals as the device's busy time, subtracts it from the
wall span of each integration, and attributes the remaining idle time to the
NVTX range the host was inside at the time. Pipeline stalls therefore show up
against the stage that caused them.

Usage
-----
Profile a run and analyse it in one step::

    profiling/gpu_idle.py --run -- -a 350 -b 350 -s 1000000 -t 3 --nchunks 30

(everything after ``--`` is passed to ``matvis profile``; the GPU, beam,
precision and coordinate-method flags used by the canonical benchmarks are
added automatically unless you override them).

Or analyse a trace you already have::

    nsys profile -t cuda,nvtx -o mytrace uv run matvis profile ...
    profiling/gpu_idle.py mytrace.nsys-rep

Interpreting the output
-----------------------
``idle`` is the headroom available to any optimization that only makes the host
faster (deeper queueing, fewer launches, removing a synchronization); it does
not shrink when you move to a faster GPU, so on a faster card it is a *larger*
fraction of the run. ``busy`` is what a faster GPU shrinks. To predict a run on
a card that is ``f`` times faster on this workload, scale ``busy`` by ``1/f``
and leave ``idle`` alone.
"""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Everything the canonical benchmarks set, so a --run invocation only has to
# name the problem size.
DEFAULT_SIM_ARGS = [
    "--gpu",
    "--interpolated-beam",
    "--single-precision",
    "--coord-method",
    "CoordinateRotationERFA",
    "-f",
    "1",
]


def run_nsys(out: Path, sim_args: list[str]) -> Path:
    """Profile ``matvis profile`` under nsys and return the report path."""
    if shutil.which("nsys") is None:
        raise SystemExit("nsys not found on PATH")

    args = list(sim_args)
    if not any(a == "--gpu" for a in args):
        args = DEFAULT_SIM_ARGS + args

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nsys",
        "profile",
        "-t",
        "cuda,nvtx",
        "--sample=none",
        "-o",
        str(out.with_suffix("")),
        "-f",
        "true",
        "uv",
        "run",
        "matvis",
        "profile",
        *args,
    ]
    print(" ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return out.with_suffix(".nsys-rep")


def to_sqlite(report: Path) -> Path:
    """Export an .nsys-rep to sqlite (no-op if it already is one)."""
    if report.suffix == ".sqlite":
        return report
    db = report.with_suffix(".sqlite")
    if not db.exists() or db.stat().st_mtime < report.stat().st_mtime:
        subprocess.run(
            [
                "nsys",
                "export",
                "-t",
                "sqlite",
                "-f",
                "true",
                "-o",
                str(db),
                str(report),
            ],
            check=True,
        )
    return db


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping [start, end) intervals; input must be sorted."""
    out: list[tuple[int, int]] = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def load(db: Path):
    """Return (busy intervals, NVTX ranges) from an nsys sqlite export."""
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT id, value FROM StringIds"))

    acts: list[tuple[int, int]] = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        try:
            acts += list(con.execute(f"SELECT start, end FROM {table}"))
        except sqlite3.OperationalError:
            pass
    if not acts:
        raise SystemExit(f"no CUDA activity in {db}")
    acts.sort()

    ranges = []
    try:
        rows = con.execute(
            "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
        )
        for s, e, text, tid in rows:
            ranges.append((s, e, text or names.get(tid, "?")))
    except sqlite3.OperationalError:
        pass
    ranges.sort()
    return merge(acts), ranges


def analyse(busy: list[tuple[int, int]], ranges: list[tuple[int, int, str]]):
    """Split the trace into integrations and attribute idle time within each."""
    # Each integration starts at a `rotate` range; the first ones belong to the
    # warmup simulation, which the caller discards.
    starts = [s for s, _, name in ranges if name == "rotate"]
    if not starts:
        raise SystemExit("no 'rotate' NVTX ranges: run with cuda,nvtx tracing")
    ends = starts[1:] + [max(e for _, e, _ in ranges)]
    bounds = list(zip(starts, ends, strict=True))

    by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for s, e, name in ranges:
        if name != "rotate":
            by_name[name].append((s, e))
    by_name["rotate"] = [(s, e) for s, e, n in ranges if n == "rotate"]
    keys = {n: [i[0] for i in v] for n, v in by_name.items()}

    busy_starts = [b[0] for b in busy]

    out = []
    for lo, hi in bounds:
        i = bisect.bisect_left(busy_starts, lo)
        if i and busy[i - 1][1] > lo:
            i -= 1
        b_tot = 0
        gaps = []
        prev_end = lo
        while i < len(busy) and busy[i][0] < hi:
            s, e = max(busy[i][0], lo), min(busy[i][1], hi)
            if s > prev_end:
                gaps.append((prev_end, s))
            b_tot += max(0, e - s)
            prev_end = max(prev_end, e)
            i += 1
        if prev_end < hi:
            gaps.append((prev_end, hi))

        attr: dict[str, float] = defaultdict(float)
        for gs, ge in gaps:
            owner = "(outside any range)"
            for name, ivs in by_name.items():
                j = bisect.bisect_right(keys[name], gs) - 1
                if j >= 0 and ivs[j][1] >= ge:
                    # innermost wins: rotate/sum_chunks never nest with stages
                    owner = name
                    break
            attr[owner] += ge - gs

        out.append(
            {
                "span_ms": (hi - lo) / 1e6,
                "busy_ms": b_tot / 1e6,
                "idle_ms": (hi - lo - b_tot) / 1e6,
                "idle_by_range_ms": {k: v / 1e6 for k, v in sorted(attr.items())},
            }
        )
    return out


def main(argv=None):
    """Profile and/or analyse a run, and print the idle breakdown."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", nargs="?", help=".nsys-rep or .sqlite to analyse")
    p.add_argument("--run", action="store_true", help="profile a run first")
    p.add_argument("--out", default="profiling/results/gpu_idle", help="trace prefix")
    p.add_argument("--json", help="also write the per-integration table here")
    p.add_argument(
        "--skip",
        type=int,
        default=None,
        help="integrations to discard as warmup (default: all but the last N-1 "
        "of the final simulation, i.e. everything before the timed loop settles)",
    )
    args, rest = p.parse_known_args(argv)
    sim_args = [a for a in rest if a != "--"]

    if args.run:
        report = run_nsys(Path(args.out), sim_args)
    elif args.report:
        report = Path(args.report)
    else:
        p.error("give a report to analyse, or --run")

    rows = analyse(*load(to_sqlite(report)))

    # The warmup simulation and the first timed integration both carry one-time
    # costs; keep the settled tail.
    skip = args.skip if args.skip is not None else max(1, len(rows) // 2)
    settled = rows[skip:] or rows[-1:]

    print(f"{len(rows)} integrations found, reporting the last {len(settled)}\n")
    print(f"{'#':>3} {'span ms':>10} {'busy ms':>10} {'idle ms':>10} {'idle %':>8}")
    for i, r in enumerate(settled, start=skip):
        print(
            f"{i:>3} {r['span_ms']:10.1f} {r['busy_ms']:10.1f} "
            f"{r['idle_ms']:10.1f} {100 * r['idle_ms'] / r['span_ms']:7.2f}%"
        )

    n = len(settled)
    tot = {"span": 0.0, "busy": 0.0, "idle": 0.0}
    per_range: dict[str, float] = defaultdict(float)
    for r in settled:
        tot["span"] += r["span_ms"] / n
        tot["busy"] += r["busy_ms"] / n
        tot["idle"] += r["idle_ms"] / n
        for k, v in r["idle_by_range_ms"].items():
            per_range[k] += v / n

    print(
        f"\nmean per integration: span {tot['span']:.1f} ms, "
        f"busy {tot['busy']:.1f} ms, idle {tot['idle']:.1f} ms "
        f"({100 * tot['idle'] / tot['span']:.2f}%)"
    )
    print("\nidle attributed to the NVTX range the host was in:")
    for name, v in sorted(per_range.items(), key=lambda kv: -kv[1]):
        print(f"  {name:>22}: {v:8.2f} ms  ({100 * v / tot['span']:5.2f}% of wall)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "integrations": settled,
                    "mean": tot,
                    "idle_by_range": dict(per_range),
                },
                indent=1,
            )
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
