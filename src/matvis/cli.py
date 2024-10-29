#!/bin/env python

"""Profile the code with a simple scalable sky model.

Running the script will write a summary of the timings of various main blocks of code.
It will also save these results in pickle format to a file annotated with the inputs
(eg. nants, ntimes, nsources, nfreqs).
"""
from __future__ import annotations

import click
import inspect
import linecache
import logging
import numpy as np
import os
import pickle
import time
from astropy.coordinates import EarthLocation
from astropy.time import Time
from line_profiler import LineProfiler
from pathlib import Path
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import GaussianBeam
from pyuvdata.telescopes import get_telescope
from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
from rich.traceback import Traceback
from typing import Literal

from matvis import DATA_PATH, HAVE_GPU, coordinates, cpu, simulate_vis

from .core.coords import CoordinateRotation

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)])

if HAVE_GPU:
    from matvis import gpu

simcpu = cpu.simulate

if HAVE_GPU:
    simgpu = gpu.simulate

beam_file = DATA_PATH / "NF_HERA_Dipole_small.fits"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matvis")

cns = Console()

# These specify which line(s) in the code correspond to which algorithmic step.
STEPS = {
    "Coordinate Rotation": ("coords.rotate(t)", "coords.select_chunk("),
    "Beam Interpolation": ("bmfunc(",),
    "Compute exp(tau)": ("taucalc(",),
    "Compute Z": ("zcalc(",),
    "Compute V": ("matprod(",),
}

profiler = LineProfiler()

main = click.Group()


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
):
    """Run the script."""
    if not HAVE_GPU and gpu:
        raise RuntimeError("Cannot run GPU version without GPU dependencies installed!")

    logger.setLevel(log_level.upper())

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
        analytic_beam, nfreq, ntimes, nants, nsource, nbeams, naz=naz, nza=nza
    )

    cns.print(Rule("Running matvis profile"))
    cns.print(f"  NANTS:            {nants:>7}")
    cns.print(f"  NTIMES:           {ntimes:>7}")
    cns.print(f"  NFREQ:            {nfreq:>7}")
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
    cns.print(Rule())

    if gpu:
        profiler.add_function(simgpu)
    else:
        profiler.add_function(simcpu)

    init_time = time.time()
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
        telescope_loc=get_telescope("hera").location,
        use_gpu=gpu,
        beam_idx=beam_idx,
        matprod_method=f"{'GPU' if gpu else 'CPU'}{matprod_method}",
        coord_method=coord_method,
        antpairs=pairs,
        min_chunks=nchunks,
        source_buffer=source_buffer,
    )
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

    cns.print()
    cns.print(Rule("Summary of timings"))
    cns.print(f"         Total Time:            {out_time - init_time:.3e} seconds")
    for thing, (hits, _time, time_per_hit, percent, nlines) in thing_stats.items():
        cns.print(
            f"{thing:>19}: {hits:>4} hits, {_time:.3e} seconds, {time_per_hit:.3e} sec/hit, {percent:4.2f}%, {nlines} lines"
        )
    cns.print(Rule())

    with open(f"{outdir}/summary-stats-{str_id}.pkl", "wb") as fl:
        pickle.dump(thing_stats, fl)


common_profile_options = [
    click.option("-A/-I", "--analytic-beam/--interpolated-beam", default=True),
    click.option("-f", "--nfreq", default=1),
    click.option(
        "-t",
        "--ntimes",
        default=1,
    ),
    click.option(
        "-b",
        "--nbeams",
        default=1,
    ),
    click.option(
        "-g/-c",
        "--gpu/--cpu",
        default=False,
    ),
    click.option(
        "--matprod-method",
        default="MatMul",
        type=click.Choice(["MatMul", "VectorDot"]),
    ),
    click.option(
        "--coord-method",
        default="CoordinateRotationAstropy",
        type=click.Choice(list(CoordinateRotation._methods.keys())),
    ),
    click.option(
        "-v/-V", "--verbose/--not-verbose", default=False, help="Print verbose output"
    ),
    click.option(
        "-l",
        "--log-level",
        default="INFO",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    ),
    click.option(
        "--nchunks",
        default=1,
    ),
    click.option("-o", "--outdir", default="."),
    click.option("--double-precision/--single-precision", default=True),
    click.option("--naz", default=360, type=int),
    click.option("--nza", default=180, type=int),
    click.option("--source-buffer", default=1.0, type=float),
]


def add_common_options(func):
    """Add common profiling options to a function."""
    for option in reversed(common_profile_options):
        func = option(func)
    return func


@main.command()
@click.option(
    "-s",
    "--nsource",
    default=1,
)
@click.option(
    "-a",
    "--nants",
    default=1,
)
@add_common_options
def profile(**kwargs):
    """Run the script."""
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
    "-a",
    "--hex-num",
    default=11,
)
@click.option(
    "-s",
    "--nside",
    default=64,
)
@click.option("-k", "--keep-ants", type=str, default="")
@click.option("--outriggers/--no-outriggers", default=False)
@add_common_options
def hera_profile(hex_num, nside, keep_ants, outriggers, **kwargs):
    """Run profiling of matvis with a HERA-like array."""
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
        beam = beam.to_uvbeam(
            freq_array=freqs,
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
