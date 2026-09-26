"""GPU implementation of the simulator."""

from __future__ import annotations

import importlib
import logging
import time
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import psutil
from astropy.constants import c as speed_of_light
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from docstring_parser import combine_docstrings
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AnalyticBeam
from pyuvdata.beam_interface import BeamInterface

from .._nvtx import nvtx_range  # noqa: F401  (re-exported for back-compat)
from .._utils import get_desired_chunks, get_dtypes, log_progress, logdebug
from ..core import _validate_inputs
from ..core import beams as _core_beams
from ..core.coords import CoordinateRotation
from ..core.tau import TauCalculator
from ..cpu.cpu import simulate as simcpu

try:
    import cupy as cp

    from . import beams
    from . import matprod as mp
    from .getz import GPUZMatrixCalc

    importlib.import_module(
        ".coords", package=__package__
    )  # need to import this to register the coordinate rotation methods

    HAVE_CUDA = True

except ImportError:
    # if not installed, don't warn
    HAVE_CUDA = False
except Exception as e:  # pragma: no cover
    # if installed but having initialization issues
    # warn, but default back to non-gpu functionality
    warnings.warn(str(e), stacklevel=2)
    HAVE_CUDA = False

logger = logging.getLogger(__name__)


ONE_OVER_C = 1.0 / speed_of_light.value

# Wall-clock and (optional) CUDA-event timings of the most recent simulate()
# call, for profiling harnesses. Not part of the public API.
LAST_RUN_STATS: dict = {}

# The same, but one entry per simulate() call, appended in call order and never
# cleared implicitly. simulate() runs one frequency at a time, so a
# multi-frequency run produces nfreq entries; LAST_RUN_STATS would only ever
# show the last of them. Harnesses should call reset_run_stats() before a run
# and read this afterwards.
ALL_RUN_STATS: list[dict] = []


def available_device_memory() -> int:
    """Device memory cupy can allocate without the driver having to find more.

    ``Device().mem_info[0]`` is free memory as the *driver* sees it, but cupy
    does not hand freed blocks back to the driver -- it keeps them in its own
    pool. So after any previous allocation in the process, driver-visible free
    memory understates what is actually available by the size of the pool's
    free blocks, and source chunking sized from it is far too conservative.

    This matters most when ``simulate`` is called more than once in a process,
    which ``simulate_vis`` does for every frequency channel: without it, each
    channel after the first plans a smaller chunk size than the one before,
    and can reach the 100-chunk ceiling in ``get_required_chunks``.
    """
    pool = cp.get_default_memory_pool()
    return int(cp.cuda.Device().mem_info[0] + pool.total_bytes() - pool.used_bytes())


def reset_run_stats():
    """Discard the stats of all previous simulate() calls."""
    LAST_RUN_STATS.clear()
    ALL_RUN_STATS.clear()


@combine_docstrings(simcpu)
def simulate(  # noqa: C901
    *,
    antpos: np.ndarray,
    freq: float,
    times: Time,
    skycoords: SkyCoord,
    telescope_loc: EarthLocation,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam | AnalyticBeam | BeamInterface] | None,
    polarized: bool = False,
    antpairs: np.ndarray | list[tuple[int, int]] | None = None,
    beam_idx: np.ndarray | None = None,
    max_memory: int = np.inf,
    min_chunks: int = 1,
    precision: int = 1,
    beam_spline_opts: dict | None = None,
    coord_method: Literal[
        "CoordinateRotationAstropy",
        "CoordinateRotationERFA",
        "GPUCoordinateRotationERFA",
    ] = "CoordinateRotationAstropy",
    matprod_method: Literal["GPUMatMul", "GPUVectorDot"] = "GPUMatMul",
    source_buffer: float = 1.0,
    coord_method_params: dict | None = None,
    memory_buffer: float = 0.9,
    gpu_event_timing: bool = False,
) -> np.ndarray:
    """GPU implementation of the visibility simulator.

    Parameters
    ----------
    gpu_event_timing : bool, optional
        If True, collect per-chunk GPU event timings for beam interpolation,
        tau, Z construction, and matprod stages; log stage medians at INFO
        level at the end of the run, and expose median/mean/std/count per stage
        (plus per-integration wall times, a warmup-robust
        ``steady_time_per_integration``, and a warmup-robust
        ``steady_gpu_time_per_integration`` summed from the actual chunks of
        each integration) via ``LAST_RUN_STATS``. Default is False.

        ``sum_chunks`` runs once per integration rather than per chunk, so it
        is timed separately: the stream is drained before it starts, meaning
        ``steady_sum_chunks_per_integration`` is its own cost and not the
        queued chunk pipeline it would otherwise block on. The drain adds no
        work (the device-to-host copy blocks anyway), so per-integration wall
        times remain comparable with runs that have event timing off.

    """
    if not HAVE_CUDA:
        raise ImportError("You need to install the [gpu] extra to use this function!")

    if not 0 < source_buffer <= 1:
        raise ValueError("source_buffer must satisfy 0 < source_buffer <= 1")
    if not 0 < memory_buffer <= 1:
        raise ValueError("memory_buffer must satisfy 0 < memory_buffer <= 1")

    init_time = time.time()

    # Host-side breakdown of setup, so the profiling harness can separate the
    # part of it that a multi-frequency restructure could hoist out of the
    # per-frequency loop from the part that is irreducibly per-frequency.
    setup_breakdown: dict[str, float] = {}
    device_bytes: dict[str, int] = {}
    _phase_t = init_time

    def _mark(name: str):
        nonlocal _phase_t
        now = time.time()
        setup_breakdown[name] = now - _phase_t
        device_bytes[name] = int(cp.get_default_memory_pool().used_bytes())
        _phase_t = now

    pr = psutil.Process()
    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, times, I_sky
    )

    rtype, ctype = get_dtypes(precision)
    _mark("validate")

    nchunks, npixc = get_desired_chunks(
        min(max_memory, available_device_memory()),
        min_chunks,
        beam_list,
        nax,
        nfeed,
        nant,
        len(I_sky),
        precision,
        source_buffer=source_buffer,
        memory_buffer=memory_buffer,
        # The GPU matprods accumulate every chunk into one buffer, and keep a
        # second one holding the result in output ordering, rather than one
        # buffer per chunk.
        vis_buffers=2,
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
        gpu=True,
        **coord_method_params,
    )
    # Use the same buffer width as the coordinate rotator, which may ignore
    # source_buffer for small chunks (see CoordinateRotation.__init__).
    nsrc_alloc = coords.nsrc_alloc
    _mark("coord_construct")

    bmfunc = beams.GPUBeamInterpolator(
        beam_list=beam_list,
        beam_idx=beam_idx,
        polarized=polarized,
        nant=nant,
        freq=freq,
        nsrc=nsrc_alloc,
        precision=precision,
        spline_opts=beam_spline_opts,
    )
    _mark("beam_construct")
    # Inside beam_construct, how much was the per-frequency UVBeam.interp.
    setup_breakdown["beam_wrangle_freq_independent"] = (
        _core_beams.LAST_WRANGLE_TIMES.get("freq_independent", 0.0)
    )
    setup_breakdown["beam_wrangle_freq_dependent"] = _core_beams.LAST_WRANGLE_TIMES.get(
        "freq_dependent", 0.0
    )

    zcalc = GPUZMatrixCalc(
        nsrc=nsrc_alloc, nfeed=nfeed, nant=nant, nax=nax, ctype=ctype, gpu=True
    )
    taucalc = TauCalculator(
        antpos=antpos, freq=freq, precision=precision, nsrc=nsrc_alloc, gpu=True
    )

    mpcls = getattr(mp, matprod_method)
    matprod = mpcls(nchunks, nfeed, nant, antpairs, precision=precision)
    debug_enabled = logger.isEnabledFor(logging.DEBUG)

    logger.debug("Starting GPU allocations...")

    init_mem = cp.cuda.Device().mem_info[0]
    logger.debug(f"Before GPU allocations, GPU mem avail is: {init_mem / 1024**3} GB")

    # antpos here is imaginary and in wavelength units
    taucalc.setup()
    _mark("tau_setup")
    if debug_enabled:
        memnow = cp.cuda.Device().mem_info[0]
        logger.debug(f"After antpos, GPU mem avail is: {memnow / 1024**3} GB.")

    bmfunc.setup()
    _mark("beam_setup")
    if debug_enabled:
        memnow = cp.cuda.Device().mem_info[0]
        if bmfunc.use_interp:
            logger.debug(f"After bmfunc, GPU mem avail is: {memnow / 1024**3} GB.")

    coords.setup()
    _mark("coord_setup")
    if debug_enabled:
        memnow = cp.cuda.Device().mem_info[0]
        logger.debug(f"After coords, GPU mem avail is: {memnow / 1024**3} GB.")

    zcalc.setup()
    _mark("z_setup")
    if debug_enabled:
        memnow = cp.cuda.Device().mem_info[0]
        logger.debug(f"After zcalc, GPU mem avail is: {memnow / 1024**3} GB.")

    matprod.setup()
    _mark("matprod_setup")
    if debug_enabled:
        memnow = cp.cuda.Device().mem_info[0]
        logger.debug(f"After matprod, GPU mem avail is: {memnow / 1024**3} GB.")

    # A single in-order stream serializes the chunk pipeline on the device.
    # This is required for correctness (the stage objects share one set of
    # buffers across chunks) and lets the host queue many chunks ahead
    # without any device-side synchronization in the loop.
    stream = cp.cuda.Stream()
    stream.use()
    if gpu_event_timing:
        event_start = [cp.cuda.Event() for _ in range(nchunks)]
        event_eq2top = [cp.cuda.Event() for _ in range(nchunks)]
        event_beam = [cp.cuda.Event() for _ in range(nchunks)]
        event_tau = [cp.cuda.Event() for _ in range(nchunks)]
        event_z = [cp.cuda.Event() for _ in range(nchunks)]
        event_matprod = [cp.cuda.Event() for _ in range(nchunks)]
        event_end = [cp.cuda.Event() for _ in range(nchunks)]
        active_chunks = np.zeros(nchunks, dtype=bool)
        # Full per-chunk sample lists rather than running means: medians are
        # robust to the one-time costs (kernel compilation, cuBLAS workspace
        # allocation) that land in the first few samples.
        event_samples = {
            "chunk_total": [],
            "beam": [],
            "tau": [],
            "z": [],
            "matprod": [],
            "sum_chunks": [],
        }
        # sum_chunks runs once per integration rather than once per chunk, and
        # has a host-side component as well as a device one, so it gets its own
        # event pair plus a wall timer (see the `sum_chunks` block below).
        event_sum_start = [cp.cuda.Event() for _ in range(ntimes)]
        event_sum_end = [cp.cuda.Event() for _ in range(ntimes)]
        sum_chunks_wall = []

    vis = np.full((ntimes, matprod.npairs, nfeed, nfeed), 0.0, dtype=ctype)
    _mark("vis_alloc")

    logger.info(f"Running With {nchunks} chunks")

    report_chunk = ntimes // 100 + 1
    pr = psutil.Process()
    tstart = time.time()
    mlast = pr.memory_info().rss
    plast = tstart
    integration_times = []
    # cupy's pool bookkeeping is host-side, so sampling it costs no sync.
    peak_device_bytes = int(cp.get_default_memory_pool().used_bytes())

    for t in range(ntimes):
        t_int_start = time.time()
        with nvtx_range("rotate"):
            coords.rotate(t)
        if gpu_event_timing:
            active_chunks.fill(False)

        for c in range(nchunks):
            if gpu_event_timing:
                event_start[c].record(stream)

            with nvtx_range("select_chunk"):
                crdtop, Isqrt, nsrcs_up = coords.select_chunk(c, t)
            logdebug("crdtop", crdtop)
            logdebug("Isqrt", Isqrt)

            if nsrcs_up < 1:
                if gpu_event_timing:
                    event_end[c].record(stream)
                continue

            if gpu_event_timing:
                active_chunks[c] = True

            if gpu_event_timing:
                event_eq2top[c].record(stream)

            if debug_enabled:
                logger.debug(
                    f"After coords, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
                )

            # Get beam. Shape is (nax, nfeed, nbeam, nsrc_alloc)
            with nvtx_range("beam"):
                A = bmfunc(crdtop[0], crdtop[1], check=t == 0)
            if gpu_event_timing:
                event_beam[c].record(stream)
            logdebug("Beam", A)
            if debug_enabled:
                logger.debug(
                    f"After beam, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
                )

            # exptau has shape (nant, nsrc)
            with nvtx_range("tau"):
                exptau = taucalc(crdtop)
            logdebug("exptau", exptau)
            if debug_enabled:
                logger.debug(
                    f"After exptau, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
                )
            if gpu_event_timing:
                event_tau[c].record(stream)

            with nvtx_range("z"):
                z = zcalc(Isqrt, A, exptau, bmfunc.beam_idx)
            if gpu_event_timing:
                event_z[c].record(stream)
            logdebug("Z", z)
            if debug_enabled:
                logger.debug(
                    f"After Z, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
                )

            # compute vis = Z.Z^dagger
            with nvtx_range("matprod"):
                matprod(z, c)
            if debug_enabled:
                logger.debug(
                    f"After matprod, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
                )

            if gpu_event_timing:
                event_matprod[c].record(stream)
                event_end[c].record(stream)

        if gpu_event_timing:
            for c in range(nchunks):
                event_end[c].synchronize()
                event_samples["chunk_total"].append(
                    cp.cuda.get_elapsed_time(event_start[c], event_end[c])
                )

                if not active_chunks[c]:
                    continue

                event_samples["beam"].append(
                    cp.cuda.get_elapsed_time(event_eq2top[c], event_beam[c])
                )
                event_samples["tau"].append(
                    cp.cuda.get_elapsed_time(event_beam[c], event_tau[c])
                )
                event_samples["z"].append(
                    cp.cuda.get_elapsed_time(event_tau[c], event_z[c])
                )
                event_samples["matprod"].append(
                    cp.cuda.get_elapsed_time(event_z[c], event_matprod[c])
                )

        # No explicit synchronization needed: sum_chunks' device-to-host copy
        # is ordered on the same stream as all the compute above.
        if gpu_event_timing:
            # Drain the queued chunk pipeline first, so the wall timer below
            # measures sum_chunks' own cost rather than the backlog it would
            # otherwise block on. This only moves the (unavoidable) block
            # earlier -- it adds no work -- so per-integration wall times are
            # unaffected.
            stream.synchronize()
            event_sum_start[t].record(stream)
            t_sum_start = time.time()

        with nvtx_range("sum_chunks"):
            matprod.sum_chunks(vis[t])

        if gpu_event_timing:
            event_sum_end[t].record(stream)
            sum_chunks_wall.append(time.time() - t_sum_start)
        logdebug("vis", vis[t])

        integration_times.append(time.time() - t_int_start)
        peak_device_bytes = max(
            peak_device_bytes, int(cp.get_default_memory_pool().used_bytes())
        )

        if not t % report_chunk and t != ntimes - 1:
            plast, mlast = log_progress(tstart, plast, t + 1, ntimes, pr, mlast)

    final_time = time.time()

    # The first integration typically includes one-time costs (cupy kernel
    # compilation, cuBLAS workspace allocation, ERFA/IERS cache loads), so the
    # steady-state throughput is the median of the *remaining* integrations.
    steady = integration_times[1:] if len(integration_times) > 1 else integration_times

    stats = {
        "freq": float(freq),
        "setup_time": tstart - init_time,
        "loop_time": final_time - tstart,
        "ntimes": ntimes,
        "nchunks": nchunks,
        "time_per_integration": (final_time - tstart) / ntimes,
        "integration_times": integration_times,
        "steady_time_per_integration": float(np.median(steady)),
        "setup_breakdown": setup_breakdown,
        "setup_device_bytes": device_bytes,
        "peak_device_bytes": peak_device_bytes,
    }

    if gpu_event_timing and event_samples["chunk_total"]:
        for t in range(ntimes):
            event_sum_end[t].synchronize()
            event_samples["sum_chunks"].append(
                cp.cuda.get_elapsed_time(event_sum_start[t], event_sum_end[t])
            )

        stats["event_timing_ms"] = {
            stage: {
                "median": float(np.median(samples)) if samples else 0.0,
                "mean": float(np.mean(samples)) if samples else 0.0,
                "std": float(np.std(samples)) if samples else 0.0,
                "n": len(samples),
            }
            for stage, samples in event_samples.items()
        }

        per_integration_gpu_ms = (
            np.array(event_samples["chunk_total"]).reshape(ntimes, nchunks).sum(axis=1)
        )
        steady_gpu_ms = (
            per_integration_gpu_ms[1:]
            if len(per_integration_gpu_ms) > 1
            else per_integration_gpu_ms
        )
        stats["steady_gpu_time_per_integration"] = (
            float(np.median(steady_gpu_ms)) / 1000.0
        )

        # sum_chunks' wall cost (device reduction/copy plus any host-side
        # reshaping). Measured after a stream drain, so it excludes the time
        # spent waiting on the queued chunk pipeline -- unlike the line
        # profiler's or NVTX's view of the same call.
        steady_sum = (
            sum_chunks_wall[1:] if len(sum_chunks_wall) > 1 else sum_chunks_wall
        )
        stats["steady_sum_chunks_per_integration"] = float(np.median(steady_sum))

        logger.info(
            "GPU event timing, median (ms): chunk_total=%.3f beam=%.3f tau=%.3f "
            "z=%.3f matprod=%.3f sum_chunks=%.3f",
            *(
                stats["event_timing_ms"][stage]["median"]
                for stage in (
                    "chunk_total",
                    "beam",
                    "tau",
                    "z",
                    "matprod",
                    "sum_chunks",
                )
            ),
        )

    LAST_RUN_STATS.clear()
    LAST_RUN_STATS.update(stats)
    ALL_RUN_STATS.append(stats)

    return vis if polarized else vis[:, :, 0, 0]
