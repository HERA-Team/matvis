"""CPU-based implementation of the matvis visibility simulator."""

from __future__ import annotations

import importlib
import logging
import time
import tracemalloc as tm
from collections.abc import Sequence
from typing import Literal

import numpy as np
import psutil
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AnalyticBeam
from pyuvdata.beam_interface import BeamInterface

from .._utils import get_desired_chunks, get_dtypes, log_progress, logdebug, memtrace
from ..core import _validate_inputs
from ..core import beams as _core_beams
from ..core.coords import CoordinateRotation
from ..core.getz import ZMatrixCalc
from ..core.tau import TauCalculator
from . import matprod as mp
from .beams import UVBeamInterpolator

importlib.import_module(
    ".coords", package=__package__
)  # need to import this to register the coordinate rotation methods

logger = logging.getLogger(__name__)

# Wall-clock timings of the most recent simulate() call, and one entry per call
# in call order, mirroring the GPU backend's dicts of the same names. Used by
# the profiling harness; not part of the public API.
LAST_RUN_STATS: dict = {}
ALL_RUN_STATS: list[dict] = []


def reset_run_stats():
    """Discard the stats of all previous simulate() calls."""
    LAST_RUN_STATS.clear()
    ALL_RUN_STATS.clear()


def simulate(
    *,
    antpos: np.ndarray,
    freq: float,
    times: Time,
    skycoords: SkyCoord,
    telescope_loc: EarthLocation,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam | AnalyticBeam | BeamInterface] | None,
    antpairs: np.ndarray | list[tuple[int, int]] | None = None,
    precision: int = 1,
    polarized: bool = False,
    beam_idx: np.ndarray | None = None,
    beam_spline_opts: dict | None = None,
    max_progress_reports: int = 100,
    matprod_method: Literal["CPUMatMul", "CPUVectorLoop"] = "CPUMatMul",
    coord_method: Literal[
        "CoordinateRotationAstropy", "CoordinateRotationERFA"
    ] = "CoordinateRotationAstropy",
    max_memory: int | float = np.inf,
    min_chunks: int = 1,
    source_buffer: float = 1.0,
    memory_buffer: float = 0.9,
    coord_method_params: dict | None = None,
):
    """
    Calculate visibility from an input intensity map and beam model.

    Parameters
    ----------
    antpos : array_like
        Antenna position array. Shape=(NANT, 3).
    freq : float
        Frequency to evaluate the visibilities at [GHz].
    I_sky : array_like
        Intensity distribution of sources/pixels on the sky, assuming intensity
        (Stokes I) only. The Stokes I intensity will be split equally between
        the two linear polarization channels, resulting in a factor of 0.5 from
        the value inputted here. This is done even if only one polarization
        channel is simulated.
        Shape=(NSRCS,).
    beam_list : list of UVBeam, optional
        If specified, evaluate primary beam values directly using UVBeam
        objects instead of using pixelized beam maps. Only one of ``bm_cube`` and
        ``beam_list`` should be provided.Note that if `polarized` is True,
        these beams must be efield beams, and conversely if `polarized` is False they
        must be power beams with a single polarization (either XX or YY).
    antpairs : array_like, optional
        Either a 2D array, shape ``(Npairs, 2)``, or list of 2-tuples of ints, with
        the list of antenna-pairs to return as visibilities (all feed-pairs are always
        calculated). If None, all feed-pairs are returned.
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:

        - 1: float32, complex64
        - 2: float64, complex128

    polarized : bool, optional
        Whether to simulate a full polarized response in terms of nn, ne, en,
        ee visibilities. See Eq. 6 of Kohn+ (arXiv:1802.04151) for notation.
        Default: False.
    beam_idx
        Optional length-NANT array specifying a beam index for each antenna.
        By default, either a single beam is assumed to apply to all antennas or
        each antenna gets its own beam.
    beam_spline_opts : dict, optional
        Dictionary of options to pass to the beam interpolation function.
    max_progress_reports : int, optional
        Maximum number of progress reports to print to the screen (if logging level
        allows). Default is 100.
    matprod_method : str, optional
        The method to use for the final matrix multiplication. Default is 'CPUMatMul',
        which simply uses `np.dot` over the two full matrices. Currently, the other
        option is `CPUVectorLoop`, which uses a loop over the antenna pairs,
        computing the sum over sources as a vector dot product.
        Whether to calculate visibilities for each antpair in antpairs as a vector
        dot-product instead of using a full matrix-matrix multiplication for all
        possible pairs. Default is False. Setting to True can be faster for large
        arrays where `antpairs` is small (possibly from high redundancy). You should
        run a performance test before using this.
    coord_method : str, optional
        The method to use to transform coordinates from the equatorial to horizontal
        frame. The default is to use Astropy coordinate transforms. A faster option,
        which is accurate to within 6 mas, is to use "CoordinateTransformERFA".
    max_memory : int, optional
        The maximum memory (in bytes) to use for the visibility calculation. This is
        not a hard-set limit, but rather a guideline for how much memory to use. If the
        expected memory usage is more than this, the calculation will be broken up into
        chunks.
    min_chunks : int, optional
        The minimum number of chunks to break the source axis into.
    source_buffer : float, optional
        The fraction of the total sources (per chunk) to pre-allocate memory for.
        Default is 1.0, which pre-allocates for all sources in each chunk. This
        avoids assuming that only a subset of sources will be above the horizon,
        but uses more memory. If you expect fewer or more sources to appear above
        the horizon at any time for a particular sky model, set this to a different
        value.
    memory_buffer : float, optional
        The fraction of free memory to use for the calculation. Default is 0.9,
        which leaves some buffer for other processes and overhead.
    coord_method_params
        Parameters particular to the coordinate rotation method of choice. For example,
        for the CoordinateRotationERFA (and GPU version of the same) method, there
        is the parameter ``update_bcrs_every``, which should be a time in seconds, for
        which larger values speed up the computation.

    Returns
    -------
    vis : array_like
        Simulated visibilities. If `polarized = True`, the output will have
        shape (NTIMES, NBLS, NFEED, NFEED), otherwise it will have
        shape (NTIMES, NBLS).

    """
    if not 0 < source_buffer <= 1:
        raise ValueError("source_buffer must satisfy 0 < source_buffer <= 1")
    if not 0 < memory_buffer <= 1:
        raise ValueError("memory_buffer must satisfy 0 < memory_buffer <= 1")

    init_time = time.time()

    # Host-side breakdown of setup, so the profiling harness can separate the
    # part of it that a multi-frequency restructure could hoist out of the
    # per-frequency loop from the part that is irreducibly per-frequency.
    setup_breakdown: dict[str, float] = {}
    _phase_t = init_time

    def _mark(name: str):
        nonlocal _phase_t
        now = time.time()
        setup_breakdown[name] = now - _phase_t
        _phase_t = now

    if not tm.is_tracing():
        tm.start()

    highest_peak = memtrace(0)

    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, times, I_sky
    )

    rtype, ctype = get_dtypes(precision)
    _mark("validate")

    current_memory = tm.get_traced_memory()[0]

    nchunks, npixc = get_desired_chunks(
        min(max_memory - current_memory, psutil.virtual_memory().available),
        min_chunks,
        beam_list,
        nax,
        nfeed,
        nant,
        len(I_sky),
        precision,
        source_buffer=source_buffer,
        memory_buffer=memory_buffer,
    )
    _mark("chunk_planning")

    coord_method = CoordinateRotation._methods[coord_method]

    coord_method_params = coord_method_params or {}
    coords = coord_method(
        flux=np.sqrt(0.5 * I_sky),
        times=times,
        telescope_loc=telescope_loc,
        skycoords=skycoords,
        chunk_size=npixc,
        precision=precision,
        source_buffer=source_buffer,
        **coord_method_params,
    )

    nsrc_alloc = coords.nsrc_alloc
    _mark("coord_construct")

    bmfunc = UVBeamInterpolator(
        beam_list=beam_list,
        beam_idx=beam_idx,
        polarized=polarized,
        nant=nant,
        freq=freq,
        spline_opts=beam_spline_opts,
        precision=precision,
        nsrc=nsrc_alloc,
    )
    _mark("beam_construct")
    # Inside beam_construct, how much was the per-frequency UVBeam.interp.
    setup_breakdown["beam_wrangle_freq_independent"] = (
        _core_beams.LAST_WRANGLE_TIMES.get("freq_independent", 0.0)
    )
    setup_breakdown["beam_wrangle_freq_dependent"] = _core_beams.LAST_WRANGLE_TIMES.get(
        "freq_dependent", 0.0
    )

    taucalc = TauCalculator(
        antpos=antpos, freq=freq, precision=precision, nsrc=nsrc_alloc
    )

    mpcls = getattr(mp, matprod_method)
    matprod = mpcls(nchunks, nfeed, nant, antpairs, precision=precision)
    zcalc = ZMatrixCalc(
        nsrc=nsrc_alloc,
        nfeed=nfeed,
        nant=nant,
        nax=nax,
        ctype=ctype,
    )

    vis = np.full((ntimes, matprod.npairs, nfeed, nfeed), 0.0, dtype=ctype)
    _mark("vis_alloc")

    bmfunc.setup()
    _mark("beam_setup")
    coords.setup()
    _mark("coord_setup")
    matprod.setup()
    _mark("matprod_setup")
    zcalc.setup()
    _mark("z_setup")
    taucalc.setup()
    _mark("tau_setup")

    logger.info(f"Visibility Array takes {vis.nbytes / 1024**2:.1f} MB")

    # Have up to 100 reports as it iterates through time.
    report_chunk = ntimes // max_progress_reports + 1
    pr = psutil.Process()
    tstart = time.time()
    mlast = pr.memory_info().rss
    plast = tstart

    highest_peak = memtrace(highest_peak)
    setup_time = time.time()

    logger.info(f"Setup Time: {setup_time - init_time:1.3e}")

    # Per-stage host timings. The CPU backend is synchronous, so unlike the GPU
    # backend's event timings these attribute work to the stage that does it
    # with no pipeline-stall ambiguity. `rotate` and `select_chunk` are the
    # frequency-independent stages.
    integration_times = []
    stage_samples = {
        "rotate": [],
        "select_chunk": [],
        "beam": [],
        "tau": [],
        "z": [],
        "matprod": [],
        "sum_chunks": [],
    }

    # Loop over time samples
    for t in range(ntimes):
        t_int_start = time.perf_counter()

        _t = time.perf_counter()
        coords.rotate(t)
        stage_samples["rotate"].append(time.perf_counter() - _t)

        for c in range(nchunks):
            _t = time.perf_counter()
            crd_top, flux_sqrt, nn = coords.select_chunk(c, t)
            stage_samples["select_chunk"].append(time.perf_counter() - _t)
            logdebug("crdtop", crd_top[:, :nn])
            logdebug("Isqrt", flux_sqrt[:nn])

            _t = time.perf_counter()
            A = bmfunc(crd_top[0], crd_top[1], check=t == 0)
            stage_samples["beam"].append(time.perf_counter() - _t)
            logdebug("beam", bmfunc.interpolated_beam[..., :nn])

            # Calculate delays, where tau = 2pi*nu*(b * s) / c
            _t = time.perf_counter()
            exptau = taucalc(crd_top)
            stage_samples["tau"].append(time.perf_counter() - _t)
            logdebug("exptau", exptau[:, :nn])

            _t = time.perf_counter()
            z = zcalc(flux_sqrt, A, exptau, bmfunc.beam_idx)
            stage_samples["z"].append(time.perf_counter() - _t)
            logdebug("Z", z[..., :nn])

            _t = time.perf_counter()
            matprod(z, c)
            stage_samples["matprod"].append(time.perf_counter() - _t)

            if not t % report_chunk and t != ntimes - 1 and c == nchunks - 1:
                plast, mlast = log_progress(tstart, plast, t + 1, ntimes, pr, mlast)
                highest_peak = memtrace(highest_peak)

        _t = time.perf_counter()
        matprod.sum_chunks(vis[t])
        stage_samples["sum_chunks"].append(time.perf_counter() - _t)
        logdebug("vis", vis[t])
        integration_times.append(time.perf_counter() - t_int_start)

    final_time = time.time()
    logger.info(f"Loop Time: {final_time - setup_time:1.3e}")

    # The first integration carries one-time costs (ERFA/IERS cache loads, BLAS
    # workspace allocation, first-touch page faults), so the steady-state
    # throughput is the median of the remaining ones.
    steady = integration_times[1:] if len(integration_times) > 1 else integration_times

    stats = {
        "freq": float(freq),
        "setup_time": setup_time - init_time,
        "loop_time": final_time - setup_time,
        "ntimes": ntimes,
        "nchunks": nchunks,
        "time_per_integration": (final_time - setup_time) / ntimes,
        "integration_times": integration_times,
        "steady_time_per_integration": float(np.median(steady)),
        "setup_breakdown": setup_breakdown,
        # Per-stage totals for the whole run, in seconds. Divide by ntimes for a
        # per-integration figure; `rotate` and `select_chunk` are the stages a
        # multi-frequency restructure could share across frequencies.
        "stage_totals": {k: float(np.sum(v)) for k, v in stage_samples.items()},
        "stage_median_ms": {
            k: float(np.median(v)) * 1e3 if v else 0.0 for k, v in stage_samples.items()
        },
    }

    logger.info(
        "CPU stage totals (s): %s",
        " ".join(f"{k}={v:.3e}" for k, v in stats["stage_totals"].items()),
    )

    LAST_RUN_STATS.clear()
    LAST_RUN_STATS.update(stats)
    ALL_RUN_STATS.append(stats)

    return vis if polarized else vis[:, :, 0, 0]
