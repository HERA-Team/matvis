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
from astropy.coordinates import EarthLocation
from astropy.time import Time
from line_profiler import LineProfiler
from pathlib import Path
from pyuvdata import UVBeam
from pyuvsim import AnalyticBeam, simsetup
from typing import Literal

from matvis import DATA_PATH, HAVE_GPU, coordinates, cpu, simulate_vis

if HAVE_GPU:
    from matvis import gpu

simcpu = cpu.simulate

if HAVE_GPU:
    simgpu = gpu.simulate

beam_file = DATA_PATH / "NF_HERA_Dipole_small.fits"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matvis")


# These specify which line(s) in the code correspond to which algorithmic step.
STEPS = {
    "Coordinate Rotation": ("coords.rotate()",),
    "Beam Interpolation": ("bmfunc(",),
    "Compute exp(tau)": ("exptau =",),
    "Compute Z": ("zcalc.compute(",),
    "Compute V": ("matprod(z, c)",),
}

profiler = LineProfiler()

main = click.Group()


def get_label(**kwargs):
    """Get a label for the output profile files."""
    return "A{analytic_beam}_nf{nfreq}_nt{ntimes}_na{nants}_ns{nsource}_nb{nbeams}_g{gpu}_pr{2 if double_precision else 1}_{method}".format(
        **kwargs
    )


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
    method,
    pairs=None,
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
        lsts,
        cpu_beams,
        hera_lat,
        beam_idx,
    ) = get_standard_sim_params(analytic_beam, nfreq, ntimes, nants, nsource, nbeams)

    print("---------------------------------")
    print("Running matvis profile with:")
    print(f"  NANTS:            {nants:>7}")
    print(f"  NTIMES:           {ntimes:>7}")
    print(f"  NFREQ:            {nfreq:>7}")
    print(f"  NBEAMS:           {nbeams:>7}")
    print(f"  NSOURCE:          {nsource:>7}")
    print(f"  GPU:              {gpu:>7}")
    print(f"  DOUBLE-PRECISION: {double_precision:>7}")
    print(f"  ANALYTIC-BEAM:    {analytic_beam:>7}")
    print(f"  METHOD:           {method:>7}")
    print(f"  NPAIRS:           {len(pairs) if pairs is not None else nants**2:>7}")
    print("---------------------------------")

    if gpu:
        profiler.add_function(simgpu)
    else:
        profiler.add_function(simcpu)

    profiler.runcall(
        simulate_vis,
        ants=ants,
        fluxes=flux,
        ra=ra,
        dec=dec,
        freqs=freqs,
        lsts=lsts,
        beams=cpu_beams,
        polarized=True,
        precision=2 if double_precision else 1,
        latitude=hera_lat * np.pi / 180.0,
        use_gpu=gpu,
        beam_idx=beam_idx,
        matprod_method=f"{'GPU' if gpu else 'CPU'}{method}",
        antpairs=pairs,
    )

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
        method=method,
    )

    with open(f"{outdir}/full-stats-{str_id}.txt", "w") as fl:
        profiler.print_stats(stream=fl, stripzeros=True)

    if verbose:
        profiler.print_stats()

    line_stats, total_time = get_line_based_stats(profiler.get_stats())
    thing_stats = get_summary_stats(line_stats, total_time, STEPS)

    print()
    print("------------- Summary of timings -------------")
    for thing, (hits, time, time_per_hit, percent, nlines) in thing_stats.items():
        print(
            f"{thing:>19}: {hits:>4} hits, {time:.3e} seconds, {time_per_hit:.3e} sec/hit, {percent:3.2f}%, {nlines} lines"
        )
    print("----------------------------------------------")

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
        "--method",
        default="MatMul",
        type=click.Choice(["MatMul", "VectorDot"]),
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
    click.option("-o", "--outdir", default="."),
    click.option("--double-precision/--single-precision", default=True),
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
    d, total_time = get_stats_and_lines(fn, lineno, timings, time_unit)
    return d, total_time


def get_summary_stats(line_data, total_time, ids):
    """Convert a line-by-line set of stats into a summary of major components."""
    # specify contents of lines where important things happen
    thing_stats = {"total": (1, total_time, total_time / 1, 100, len(line_data))}
    for thing, lines in ids.items():
        init_line = None
        assoc_lines = []
        for dd in line_data:
            if lines[0] in dd:
                init_line = dd
                assoc_lines.append(init_line)

            if len(lines) > 1 and init_line:
                assoc_lines.append(dd)

                if lines[1] in dd:
                    break
            elif len(lines) == 1 and init_line:
                break

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
    for lineno, nhits, time in timings:
        total_time += time
        linenos.append(lineno)

    if not os.path.exists(filename):
        raise ValueError(f"Could not find file: {filename}")

    linecache.clearcache()
    all_lines = linecache.getlines(filename)
    sublines = inspect.getblock(all_lines[start_lineno - 1 :])
    all_linenos = list(range(start_lineno, start_lineno + len(sublines)))

    for lineno, nhits, time in timings:
        percent = 100 * time / total_time
        idx = all_linenos.index(lineno)

        d[sublines[idx].rstrip("\n").rstrip("\r")] = (
            nhits,
            time * time_unit,
            float(time) / nhits * time_unit,
            percent,
            lineno,
        )

    return d, total_time * time_unit


def get_standard_sim_params(
    use_analytic_beam: bool, nfreq, ntime, nants, nsource, nbeams
):
    """Create some standard random simulation parameters for use in profiling.

    Will create a sky with uniformly distributed point sources (half below the horizon).

    """
    hera_lat = -30.7215
    hera_lon = 21.4283
    hera_alt = 1073.0
    obstime = Time("2018-08-31T04:02:30.11", format="isot", scale="utc")

    # HERA location
    location = EarthLocation.from_geodetic(lat=hera_lat, lon=hera_lon, height=hera_alt)

    # Set the seed so that different runs take about the same time.
    np.random.seed(1)

    # Beam model
    if use_analytic_beam:
        beam = AnalyticBeam("gaussian", diameter=14.0)
    else:
        # This is a peak-normalized e-field beam file at 100 and 101 MHz,
        # downsampled to roughly 4 square-degree resolution.
        beam = UVBeam()
        beam.read_beamfits(beam_file)

    beams = [beam] * nbeams

    # Random antenna locations
    x = np.random.random(nants) * 400.0  # Up to 400 metres
    y = np.random.random(nants) * 400.0
    z = np.random.random(nants) * 0.0
    ants = {i: (x[i], y[i], z[i]) for i in range(nants)}

    # This will make the beam_idx like [0,1,2,3,3,3,3,3,3,3] where nbeams=4 and the
    # array is nants long.
    beam_idx = np.array(list(range(nbeams)) + [nbeams - 1] * (nants - nbeams))

    # Observing parameters in a UVData object
    uvdata = simsetup.initialize_uvdata_from_keywords(
        Nfreqs=nfreq,
        start_freq=100e6,
        channel_width=97.3e3,
        start_time=obstime.jd,
        integration_time=182.0,  # Just over 3 mins between time samples
        Ntimes=ntime,
        array_layout=ants,
        polarization_array=np.array(["XX", "YY", "XY", "YX"]),
        telescope_location=(hera_lat, hera_lon, hera_alt),
        telescope_name="test_array",
        phase_type="drift",
        vis_units="Jy",
        complete=True,
        write_files=False,
    )
    lsts = np.unique(uvdata.lst_array)

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

    # Source locations and frequencies
    freqs = np.unique(uvdata.freq_array)

    # Correct source locations so that matvis uses the right frame
    ra, dec = coordinates.equatorial_to_eci_coords(
        ra, dec, obstime, location, unit="rad", frame="icrs"
    )

    # Calculate source fluxes for matvis
    flux = ((freqs[:, np.newaxis] / freqs[0]) ** spec_indx.T * flux0.T).T

    return (
        ants,
        flux,
        ra,
        dec,
        freqs,
        lsts,
        beams,
        hera_lat,
        beam_idx,
    )
