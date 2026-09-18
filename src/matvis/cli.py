#!/bin/env python

"""Profile the code with a simple scalable sky model.

Running the script will write a summary of the timings of various main blocks of code.
It will also save these results in pickle format to a file annotated with the inputs
(eg. nants, ntimes, nsources, nfreqs).
"""

from __future__ import annotations

import inspect
import json
import linecache
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Literal

import click
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from line_profiler import LineProfiler
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import GaussianBeam
from pyuvdata.telescopes import known_telescope_location
from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
from rich.traceback import Traceback

from matvis import DATA_PATH, HAVE_GPU, coordinates, cpu, simulate_vis

from .core.coords import CoordinateRotation

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)])

if HAVE_GPU:
    from matvis import gpu
    from matvis.gpu import gpu as gpu_module

from matvis.cpu import cpu as cpu_module  # noqa: E402

simcpu = cpu.simulate

if HAVE_GPU:
    simgpu = gpu.simulate

beam_file = DATA_PATH / "NF_HERA_Dipole_small.fits"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matvis")

cns = Console()

# Which setup phases a multi-frequency restructure could hoist out of the
# per-frequency loop, and which are irreducibly per-frequency. `beam_construct`
# is deliberately absent: it is the sum of its two `beam_wrangle_*` sub-phases
# plus a small remainder, which is accounted for separately.
SETUP_FREQ_INDEPENDENT = (
    "validate",
    "chunk_planning",
    "coord_construct",
    "coord_setup",
    "beam_wrangle_freq_independent",
    "z_setup",
    "matprod_setup",
    "vis_alloc",
)
# tau: antpos is pre-scaled by 2*pi*nu/c. beam_setup: uploads/prefilters this
# frequency's beam grid. beam_wrangle_freq_dependent: UVBeam.interp onto it.
SETUP_FREQ_DEPENDENT = (
    "tau_setup",
    "beam_setup",
    "beam_wrangle_freq_dependent",
)


def classify_setup(breakdown: dict) -> dict:
    """Split one call's setup breakdown into hoistable and per-frequency work."""
    indep = sum(breakdown.get(k, 0.0) for k in SETUP_FREQ_INDEPENDENT)
    dep = sum(breakdown.get(k, 0.0) for k in SETUP_FREQ_DEPENDENT)
    # Whatever `beam_construct` cost beyond its two measured sub-phases is
    # BeamInterface bookkeeping, which does not depend on frequency.
    indep += max(
        breakdown.get("beam_construct", 0.0)
        - breakdown.get("beam_wrangle_freq_independent", 0.0)
        - breakdown.get("beam_wrangle_freq_dependent", 0.0),
        0.0,
    )
    return {"freq_independent": indep, "freq_dependent": dep}


def summarize_run(all_stats: list[dict]) -> dict:
    """Aggregate the per-frequency backend stats of one simulate_vis call.

    simulate_vis calls the backend once per frequency, so `all_stats` has one
    entry per channel. Medians are taken across channels; setup, which is paid
    per channel, is summed.
    """
    if not all_stats:
        return {}

    nfreq = len(all_stats)
    ntimes = all_stats[0]["ntimes"]
    setup_total = sum(st["setup_time"] for st in all_stats)
    splits = [classify_setup(st.get("setup_breakdown", {})) for st in all_stats]

    out = {
        "nfreq_calls": nfreq,
        "setup_time_total": setup_total,
        "setup_time_per_freq": setup_total / nfreq,
        "setup_freq_independent_per_freq": float(
            np.mean([sp["freq_independent"] for sp in splits])
        ),
        "setup_freq_dependent_per_freq": float(
            np.mean([sp["freq_dependent"] for sp in splits])
        ),
        "steady_wall_per_integration": float(
            np.median([st["steady_time_per_integration"] for st in all_stats])
        ),
    }

    loop = out["steady_wall_per_integration"]
    total = nfreq * (out["setup_time_per_freq"] + ntimes * loop)
    # Issue #134: hoisting the frequency-independent setup out of the loop saves
    # it (nfreq - 1) times. This is the ceiling for that part of the work; it
    # says nothing about the in-loop stages.
    out["projected_setup_hoist_saving"] = (
        (nfreq - 1) * out["setup_freq_independent_per_freq"] / total if total else 0.0
    )

    gpu_times = [
        st["steady_gpu_time_per_integration"]
        for st in all_stats
        if "steady_gpu_time_per_integration" in st
    ]
    if gpu_times:
        out["gpu_time_per_integration"] = float(np.median(gpu_times))
        # The host timer and the per-chunk GPU events bound different,
        # overlapping windows, so clamp at zero rather than report a negative.
        out["host_overhead_per_integration"] = max(
            out["steady_wall_per_integration"] - out["gpu_time_per_integration"], 0.0
        )

    events = [st["event_timing_ms"] for st in all_stats if "event_timing_ms" in st]
    if events:
        out["event_timing_ms"] = {
            stage: float(np.median([ev[stage]["median"] for ev in events]))
            for stage in events[0]
        }

    peaks = [st["peak_device_bytes"] for st in all_stats if "peak_device_bytes" in st]
    if peaks:
        out["peak_device_bytes"] = int(max(peaks))

    totals = [st["stage_totals"] for st in all_stats if "stage_totals" in st]
    if totals:
        out["stage_seconds_per_integration"] = {
            stage: float(np.median([tt[stage] for tt in totals])) / ntimes
            for stage in totals[0]
        }

    return out


# These specify which line(s) in the code correspond to which algorithmic step.
STEPS = {
    "Coordinate Rotation": ("coords.rotate(t)", "coords.select_chunk("),
    "Beam Interpolation": ("bmfunc(",),
    "Compute exp(tau)": ("taucalc(",),
    "Compute Z": ("zcalc(",),
    "Compute V": ("matprod(",),
}

profiler = LineProfiler()

main = click.Group(
    help="Profiling and benchmarking utilities for matvis. See `matvis profile "
    "--help` and `matvis hera-profile --help` for the two subcommands."
)


def get_label(**kwargs):
    """Get a label for the output profile files."""
    precision = 2 if kwargs["double_precision"] else 1
    return (
        "A{analytic_beam}_nf{nfreq}_nt{ntimes}_na{nants}_ns{nsource}_nb{nbeams}_"
        "naz{naz}_nza{nza}_g{gpu}_pr{precision}_{matprod_method}_{coord_method}"
    ).format(precision=precision, **kwargs)


def run_profile(
    analytic_beam,
    nfreq,
    ntimes,
    nants,
    nbeams,
    nsource,
    gpu,
    double_precision,
    outdir,
    verbose,
    log_level,
    matprod_method,
    coord_method,
    naz=360,
    nza=180,
    pairs=None,
    nchunks=1,
    source_buffer=1.0,
    gpu_event_timing=False,
    warmup=True,
    update_bcrs_every=0.0,
    repeat=1,
    beam_nfreq=0,
):
    """Run the script."""
    if not HAVE_GPU and gpu:
        raise RuntimeError("Cannot run GPU version without GPU dependencies installed!")

    logger.setLevel(log_level.upper())

    # update_bcrs_every only exists on the ERFA rotators. Passing it to a method
    # that does not accept it would be a silent no-op at best, so check.
    coord_method_params = {}
    coord_cls = CoordinateRotation._methods[coord_method]
    if "update_bcrs_every" in inspect.signature(coord_cls.__init__).parameters:
        coord_method_params["update_bcrs_every"] = update_bcrs_every
    elif update_bcrs_every:
        raise click.UsageError(
            f"--update-bcrs-every is not accepted by {coord_method}; it only "
            "applies to the ERFA coordinate methods."
        )

    (
        ants,
        flux,
        ra,
        dec,
        freqs,
        times,
        cpu_beams,
        beam_idx,
    ) = get_standard_sim_params(
        analytic_beam,
        nfreq,
        ntimes,
        nants,
        nsource,
        nbeams,
        naz=naz,
        nza=nza,
        beam_nfreq=beam_nfreq,
    )

    cns.print(Rule("Running matvis profile"))
    cns.print(f"  NANTS:            {nants:>7}")
    cns.print(f"  NTIMES:           {ntimes:>7}")
    cns.print(f"  NFREQ:            {nfreq:>7}")
    cns.print(f"  BEAM NFREQ:       {beam_nfreq or nfreq:>7}")
    cns.print(f"  NBEAMS:           {nbeams:>7}")
    cns.print(f"  NSOURCE:          {nsource:>7}")
    cns.print(f"  GPU:              {gpu:>7}")
    cns.print(f"  DOUBLE-PRECISION: {double_precision:>7}")
    cns.print(f"  ANALYTIC-BEAM:    {analytic_beam:>7}")
    cns.print(f"  MATPROD METHOD:   {matprod_method:>7}")
    cns.print(f"  COORDROT METHOD:  {coord_method:>7}")
    cns.print(f"  NPAIRS:           {len(pairs) if pairs is not None else nants**2:>7}")
    cns.print(f"  NAZ:              {naz:>7}")
    cns.print(f"  NZA:              {nza:>7}")
    cns.print(f"  GPU-EVENT-TIMING: {gpu_event_timing:>7}")
    cns.print(f"  WARMUP:           {warmup:>7}")
    cns.print(Rule())

    if warmup and gpu:
        # An untimed miniature run with the same precision, beam type and
        # backend methods, so all one-time costs (cupy RawKernel/ufunc
        # compilation, cuBLAS handle+workspace creation, ERFA/IERS caches)
        # are paid before any timing starts.
        nwarm = min(nsource, 10_000)
        logger.info(f"Running warmup simulation ({nwarm} sources, 1 time)...")
        simulate_vis(
            ants=ants,
            # flux is (nsource, nfreq); the warmup runs one channel, so the
            # frequency axis has to be sliced too or simulate_vis rejects it.
            fluxes=flux[:nwarm, :1],
            ra=ra[:nwarm],
            dec=dec[:nwarm],
            freqs=freqs[:1],
            times=times[:1],
            beams=cpu_beams,
            polarized=True,
            precision=2 if double_precision else 1,
            telescope_loc=known_telescope_location("hera"),
            use_gpu=gpu,
            beam_idx=beam_idx,
            matprod_method=f"{'GPU' if gpu else 'CPU'}{matprod_method}",
            coord_method=coord_method,
            antpairs=pairs,
            source_buffer=source_buffer,
        )

    if gpu:
        profiler.add_function(simgpu)
    else:
        profiler.add_function(simcpu)

    backend_module = gpu_module if gpu else cpu_module

    backend_kwargs = {"gpu_event_timing": gpu_event_timing} if gpu else {}

    per_repeat = []
    init_time = time.time()
    for _ in range(repeat):
        # Each repeat is a fresh measurement; the line profiler accumulates
        # across them by design, but the backend stats must not.
        backend_module.reset_run_stats()
        profiler.runcall(
            simulate_vis,
            ants=ants,
            fluxes=flux,
            ra=ra,
            dec=dec,
            freqs=freqs,
            times=times,
            beams=cpu_beams,
            polarized=True,
            precision=2 if double_precision else 1,
            telescope_loc=known_telescope_location("hera"),
            use_gpu=gpu,
            beam_idx=beam_idx,
            matprod_method=f"{'GPU' if gpu else 'CPU'}{matprod_method}",
            coord_method=coord_method,
            coord_method_params=coord_method_params,
            antpairs=pairs,
            min_chunks=nchunks,
            source_buffer=source_buffer,
            **backend_kwargs,
        )
        per_repeat.append(summarize_run(backend_module.ALL_RUN_STATS))
    out_time = time.time()

    outdir = Path(outdir).expanduser().absolute()

    str_id = get_label(
        analytic_beam=analytic_beam,
        nfreq=nfreq,
        ntimes=ntimes,
        nants=nants,
        nbeams=nbeams,
        nsource=nsource,
        gpu=gpu,
        double_precision=double_precision,
        matprod_method=matprod_method,
        coord_method=coord_method,
        naz=naz,
        nza=nza,
    )

    with open(f"{outdir}/full-stats-{str_id}.txt", "w") as fl:
        profiler.print_stats(stream=fl, stripzeros=True)

    if verbose:
        profiler.print_stats()

    line_stats = get_line_based_stats(profiler.get_stats())
    thing_stats = get_summary_stats(line_stats, STEPS)

    # Median across repeats of every scalar, so one slow repeat cannot move the
    # headline. The spread is reported alongside: an effect smaller than it is
    # not measurable with this many repeats.
    derived = {}
    scalar_keys = {
        k for rp in per_repeat for k, v in rp.items() if isinstance(v, (int, float))
    }
    for key in sorted(scalar_keys):
        vals = [rp[key] for rp in per_repeat if key in rp]
        derived[key] = float(np.median(vals))
    for key in ("event_timing_ms", "stage_seconds_per_integration"):
        if per_repeat and key in per_repeat[0]:
            derived[key] = {
                stage: float(
                    np.median([rp[key][stage] for rp in per_repeat if key in rp])
                )
                for stage in per_repeat[0][key]
            }
    if repeat > 1:
        walls = [rp["steady_wall_per_integration"] for rp in per_repeat]
        derived["repeat_wall_times"] = walls
        derived["repeat_wall_spread_frac"] = float(
            (max(walls) - min(walls)) / np.median(walls)
        )

    cns.print()
    cns.print(Rule("Summary of timings"))
    cns.print(f"         Total Time:            {out_time - init_time:.3e} seconds")
    if "steady_wall_per_integration" in derived:
        cns.print(
            f"  Steady-state wall time per integration: "
            f"{derived['steady_wall_per_integration']:.3e} seconds"
        )
    if "gpu_time_per_integration" in derived:
        cns.print(
            f"  GPU time per integration (median):      "
            f"{derived['gpu_time_per_integration']:.3e} seconds"
        )
        cns.print(
            f"  Host overhead per integration:          "
            f"{derived['host_overhead_per_integration']:.3e} seconds"
        )
    if "repeat_wall_spread_frac" in derived:
        cns.print(
            f"  Spread over {repeat} repeats (max-min)/median: "
            f"{derived['repeat_wall_spread_frac']:.1%}"
        )
    if "peak_device_bytes" in derived:
        cns.print(
            f"  Peak device memory:                     "
            f"{derived['peak_device_bytes'] / 1024**3:.2f} GB"
        )

    if "setup_time_per_freq" in derived:
        cns.print()
        cns.print(Rule("Setup, per frequency"))
        cns.print(
            f"  Total setup per frequency:              "
            f"{derived['setup_time_per_freq']:.3e} seconds"
        )
        cns.print(
            f"    frequency-independent (hoistable):    "
            f"{derived['setup_freq_independent_per_freq']:.3e} seconds"
        )
        cns.print(
            f"    frequency-dependent:                  "
            f"{derived['setup_freq_dependent_per_freq']:.3e} seconds"
        )
        cns.print(
            f"  Projected saving from hoisting setup:   "
            f"{derived['projected_setup_hoist_saving']:.2%} of this run"
        )

    if "stage_seconds_per_integration" in derived:
        cns.print()
        cns.print(Rule("Host-timed stages, per integration (CPU backend)"))
        stages = derived["stage_seconds_per_integration"]
        total = sum(stages.values())
        for stage, val in stages.items():
            frac = val / total if total else 0.0
            cns.print(f"  {stage:>14}: {val:.3e} seconds  ({frac:5.1%})")

    cns.print()
    for thing, (hits, _time, time_per_hit, percent, nlines) in thing_stats.items():
        cns.print(
            f"{thing:>19}: {hits:>4} hits, {_time:.3e} seconds, {time_per_hit:.3e} sec/hit, {percent:4.2f}%, {nlines} lines"
        )
    cns.print(Rule())

    with open(f"{outdir}/summary-stats-{str_id}.pkl", "wb") as fl:
        pickle.dump(thing_stats, fl)

    # Machine-readable summary for before/after benchmark comparisons.
    summary = {
        "config": {
            "analytic_beam": analytic_beam,
            "nfreq": nfreq,
            "ntimes": ntimes,
            "nants": nants,
            "nbeams": nbeams,
            "nsource": nsource,
            "gpu": gpu,
            "precision": 2 if double_precision else 1,
            "matprod_method": matprod_method,
            "coord_method": coord_method,
            "naz": naz,
            "nza": nza,
            "nchunks": nchunks,
            "source_buffer": source_buffer,
            "update_bcrs_every": coord_method_params.get("update_bcrs_every"),
            "repeat": repeat,
            "beam_nfreq": beam_nfreq or nfreq,
        },
        "total_time": out_time - init_time,
        "stages": {
            thing: {
                "hits": hits,
                "time": _time,
                "time_per_hit": time_per_hit,
                "percent": percent,
            }
            for thing, (hits, _time, time_per_hit, percent, _) in thing_stats.items()
        },
    }
    summary["derived"] = derived
    # Back-compat: the last frequency's stats, as a bare dict. The per-frequency
    # detail of the final repeat lives alongside it -- simulate_vis calls the
    # backend once per channel, so at nfreq > 1 the bare dict is only one of
    # them.
    if backend_module.LAST_RUN_STATS:
        summary["run_stats"] = dict(backend_module.LAST_RUN_STATS)
    summary["run_stats_per_freq"] = [dict(st) for st in backend_module.ALL_RUN_STATS]
    with open(f"{outdir}/summary-stats-{str_id}.json", "w") as fl:
        json.dump(summary, fl, indent=2)


common_profile_options = [
    click.option(
        "-A/-I",
        "--analytic-beam/--interpolated-beam",
        default=True,
        help="Use an analytic (Gaussian) beam, or a gridded UVBeam requiring interpolation.",
    ),
    click.option("-f", "--nfreq", default=1, help="Number of frequency channels."),
    click.option(
        "-t",
        "--ntimes",
        default=1,
        help="Number of time integrations.",
    ),
    click.option(
        "-b",
        "--nbeams",
        default=1,
        help="Number of unique beams (1, or up to --nants for one beam per antenna).",
    ),
    click.option(
        "-g/-c",
        "--gpu/--cpu",
        default=False,
        help="Run on the GPU or CPU backend.",
    ),
    click.option(
        "--matprod-method",
        default="MatMul",
        type=click.Choice(["MatMul", "VectorDot"]),
        help="Matrix-product strategy; the CPU/GPU prefix is added automatically.",
    ),
    click.option(
        "--coord-method",
        default="CoordinateRotationAstropy",
        type=click.Choice(list(CoordinateRotation._methods.keys())),
        help="Coordinate rotation method.",
    ),
    click.option(
        "--update-bcrs-every",
        default=0.0,
        type=float,
        help=(
            "Seconds between full recomputations of the BCRS source vectors "
            "(ERFA coordinate methods only). The default of 0 recomputes them "
            "at every integration, which is the exact but slowest setting; "
            "~180 keeps errors below ~10 mas for far less work."
        ),
    ),
    click.option(
        "--beam-nfreq",
        default=0,
        type=int,
        help=(
            "Number of frequency channels in the gridded test beam. The default "
            "of 0 gives it exactly the simulated channels, which makes matvis's "
            "per-channel beam interpolation nearly free and is not "
            "representative; a real beam covers the whole band."
        ),
    ),
    click.option(
        "--repeat",
        default=1,
        type=int,
        help=(
            "Run the whole simulation this many times and report the median "
            "and the spread, so an effect can be told from run-to-run noise."
        ),
    ),
    click.option(
        "-v/-V", "--verbose/--not-verbose", default=False, help="Print verbose output"
    ),
    click.option(
        "-l",
        "--log-level",
        default="INFO",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        help="Logging verbosity.",
    ),
    click.option(
        "--nchunks",
        default=1,
        help="Minimum number of source chunks (more may be used automatically "
        "if memory requires it).",
    ),
    click.option(
        "-o",
        "--outdir",
        default=".",
        type=click.Path(file_okay=False, dir_okay=True, exists=True),
        help="Directory to write summary-stats JSON and profiling output to.",
    ),
    click.option(
        "--double-precision/--single-precision",
        default=True,
        help="Use float64/complex128 or float32/complex64 throughout.",
    ),
    click.option(
        "--naz",
        default=360,
        type=int,
        help="Number of azimuth grid points for gridded beams.",
    ),
    click.option(
        "--nza",
        default=180,
        type=int,
        help="Number of zenith-angle grid points for gridded beams.",
    ),
    click.option(
        "--source-buffer",
        default=1.0,
        type=float,
        help="Fraction of nsource to pre-allocate per chunk for sources above the horizon.",
    ),
    click.option(
        "--gpu-event-timing/--no-gpu-event-timing",
        default=False,
        help="Collect per-chunk CUDA-event timings (see the docs Performance page).",
    ),
    click.option(
        "--warmup/--no-warmup",
        default=True,
        help="Run a small untimed simulation first so one-time costs (kernel "
        "compilation, cuBLAS workspace, coordinate caches) don't skew timings.",
    ),
]


def add_common_options(func):
    """Add common profiling options to a function."""
    for option in reversed(common_profile_options):
        func = option(func)
    return func


@main.command()
@click.option("-s", "--nsource", default=1, help="Number of point sources.")
@click.option("-a", "--nants", default=1, help="Number of antennas.")
@add_common_options
def profile(**kwargs):
    """Profile a matvis simulation with a synthetic (random-position) sky model.

    Writes human-readable summaries plus machine-readable summary-stats JSON
    and full-stats text files to --outdir. See the docs Performance page for
    how to interpret the output, and profiling/run-canonical.sh for the
    canonical benchmark configurations used to track performance over time.
    """
    run_profile(**kwargs)


def get_redundancies(bls, ndecimals: int = 2):
    """Find redundant baselines."""
    uvbins = set()
    pairs = []

    # Everything here is in wavelengths
    bls = np.round(bls, decimals=ndecimals)
    nant = bls.shape[0]

    # group redundant baselines
    for i in range(nant):
        for j in range(i + 1, nant):
            u, v = bls[i, j]
            if (u, v) not in uvbins and (-u, -v) not in uvbins:
                uvbins.add((u, v))
                pairs.append([i, j])

    return pairs


@main.command()
@click.option(
    "-a", "--hex-num", default=11, help="Hex-grid parameter for the HERA-like array."
)
@click.option(
    "-s",
    "--nside",
    default=64,
    help="HEALPix nside for the sky model (nsource = 12 x nside^2).",
)
@click.option(
    "-k",
    "--keep-ants",
    type=str,
    default="",
    help="Comma-separated antenna indices to keep (default: all).",
)
@click.option(
    "--outriggers/--no-outriggers",
    default=False,
    help="Include HERA outrigger antennas.",
)
@add_common_options
def hera_profile(hex_num, nside, keep_ants, outriggers, **kwargs):
    """Profile a matvis simulation with a HERA-like array and a HEALPix sky model.

    Unlike `profile` (synthetic random-position sky), this uses a real
    HERA-like antenna layout (hex-packed core, optional outriggers) and a
    full-sky HEALPix source grid, so it's a closer proxy for a production run.
    """
    from py21cmsense.antpos import hera

    antpos = hera(hex_num=hex_num, split_core=True, outriggers=2 if outriggers else 0)
    if keep_ants:
        keep_ants = [int(i) for i in keep_ants.split(",")]
        antpos = antpos[keep_ants]

    bls = antpos[np.newaxis, :, :2] - antpos[:, np.newaxis, :2]
    pairs = np.array(get_redundancies(bls.value))

    run_profile(nsource=12 * nside**2, nants=antpos.shape[0], pairs=pairs, **kwargs)


def get_line_based_stats(lstats) -> tuple[dict, float]:
    """Convert the line-number based stats into line-based stats."""
    time_unit = lstats.unit
    (fn, lineno, name), timings = sorted(lstats.timings.items())[0]
    return get_stats_and_lines(fn, lineno, timings, time_unit)


def get_summary_stats(line_data, ids):
    """Convert a line-by-line set of stats into a summary of major components."""
    # specify contents of lines where important things happen
    thing_stats = {}  # "total": (1, total_time, total_time / 1, 100, len(line_data))}
    for thing, lines in ids.items():
        assoc_lines = [dd for line in lines for dd in line_data if line in dd]

        if not assoc_lines:
            raise RuntimeError(
                f"Could not find any lines for {thing} satisfying '{lines}'. "
                "Possible lines:\n" + "\n".join(list(line_data.keys()))
            )

        # save (hits, time, time/hits, percent, nlines)
        thing_stats[thing] = (
            line_data[assoc_lines[0]][0],
            sum(line_data[ln][1] for ln in assoc_lines),
            sum(line_data[ln][2] for ln in assoc_lines),
            sum(line_data[ln][3] for ln in assoc_lines),
            len(assoc_lines),
        )

    return thing_stats


def get_stats_and_lines(filename, start_lineno, timings, time_unit):
    """Match up timing stats with line content of the code."""
    d = {}
    total_time = 0.0
    linenos = []
    for lineno, nhits, _time in timings:
        total_time += _time
        linenos.append(lineno)

    if not os.path.exists(filename):
        raise ValueError(f"Could not find file: {filename}")

    linecache.clearcache()
    all_lines = linecache.getlines(filename)
    sublines = inspect.getblock(all_lines[start_lineno - 1 :])
    all_linenos = list(range(start_lineno, start_lineno + len(sublines)))

    for lineno, nhits, _time in timings:
        percent = 100 * _time / total_time
        idx = all_linenos.index(lineno)

        d[sublines[idx].rstrip("\n").rstrip("\r")] = (
            nhits,
            _time * time_unit,
            float(_time) / nhits * time_unit,
            percent,
            lineno,
        )

    return d


def get_standard_sim_params(
    use_analytic_beam: bool,
    nfreq,
    ntime,
    nants,
    nsource,
    nbeams,
    naz=360,
    nza=180,
    freq_min=100e6,
    freq_max=200e6,
    beam_nfreq=0,
):
    """Create some standard random simulation parameters for use in profiling.

    Will create a sky with uniformly distributed point sources (half below the horizon).
    """
    # Set the seed so that different runs take about the same time.
    rng = np.random.default_rng()

    # Source locations and frequencies
    freqs = np.linspace(freq_min, freq_max, nfreq)

    # Beam model
    beam = GaussianBeam(diameter=14.0)

    if not use_analytic_beam:
        # The beam's own channels, which are NOT the channels being simulated.
        # Real runs read a beam covering the whole band and matvis interpolates
        # it onto each simulated channel, so a beam that happens to carry
        # exactly the simulated channels (beam_nfreq=0, the historical default)
        # makes that interpolation almost free and hides its cost entirely.
        if beam_nfreq:
            beam_freqs = np.linspace(freq_min - 10e6, freq_max + 10e6, beam_nfreq)
        else:
            beam_freqs = freqs
        beam = beam.to_uvbeam(
            freq_array=beam_freqs,
            axis1_array=np.linspace(0, 2 * np.pi, naz + 1)[:-1],
            axis2_array=np.linspace(0, np.pi, nza + 1),
        )

    beams = [beam] * nbeams

    # Random antenna locations
    x = rng.uniform(size=nants) * 400.0  # Up to 400 metres
    y = rng.uniform(size=nants) * 400.0
    z = np.zeros(nants)
    ants = {i: (x[i], y[i], z[i]) for i in range(nants)}

    # This will make the beam_idx like [0,1,2,3,3,3,3,3,3,3] where nbeams=4 and the
    # array is nants long.
    if nbeams in [1, nants]:
        beam_idx = None
    else:
        beam_idx = np.array(list(range(nbeams)) + [nbeams - 1] * (nants - nbeams))

    times = Time(np.linspace(2459865.0, 2459866.0, ntime), format="jd")

    # The first source always near zenith (makes sure there's always at least one
    # source above the horizon at the first time).
    ra0 = 125.7 * np.pi / 180
    dec0 = -30.72 * np.pi / 180

    if nsource > 1:
        ra = np.random.uniform(low=0.0, high=2 * np.pi, size=nsource - 1)
        dec = np.arccos(1 - 2 * np.random.uniform(size=nsource - 1)) - np.pi / 2
        ra = np.concatenate(([ra0], ra))
        dec = np.concatenate(([dec0], dec))
    else:
        ra = np.array([ra0])
        dec = np.array([dec0])

    flux0 = np.random.random(nsource) * 4
    spec_indx = np.random.normal(0.8, scale=0.05, size=nsource)

    # Calculate source fluxes for matvis
    flux = ((freqs[:, np.newaxis] / freqs[0]) ** spec_indx.T * flux0.T).T

    return (
        ants,
        flux,
        ra,
        dec,
        freqs,
        times,
        beams,
        beam_idx,
    )
