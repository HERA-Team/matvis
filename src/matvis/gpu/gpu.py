"""GPU implementation of the simulator."""

from __future__ import annotations

import importlib
import logging
import time
import warnings
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
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

from .._utils import get_desired_chunks, get_dtypes, log_progress, logdebug
from ..core import _validate_inputs
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

try:
    from cupy.cuda import nvtx as _nvtx

    @contextmanager
    def nvtx_range(name: str):
        """Annotate a block as an NVTX range (visible in nsys timelines)."""
        _nvtx.RangePush(name)
        try:
            yield
        finally:
            _nvtx.RangePop()

except ImportError:

    def nvtx_range(name: str):  # noqa: D103
        return nullcontext()


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
    """GPU implementation of the visibility simulator."""
    if not HAVE_CUDA:
        raise ImportError("You need to install the [gpu] extra to use this function!")

    if not 0 < source_buffer <= 1:
        raise ValueError("source_buffer must satisfy 0 < source_buffer <= 1")
    if not 0 < memory_buffer <= 1:
        raise ValueError("memory_buffer must satisfy 0 < memory_buffer <= 1")

    init_time = time.time()
    pr = psutil.Process()
    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, times, I_sky
    )

    rtype, ctype = get_dtypes(precision)

    nchunks, npixc = get_desired_chunks(
        min(max_memory, cp.cuda.Device().mem_info[0]),
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
    if debug_enabled:
        memnow = cp.cuda.Device().mem_info[0]
        logger.debug(f"After antpos, GPU mem avail is: {memnow / 1024**3} GB.")

    bmfunc.setup()
    if debug_enabled:
        memnow = cp.cuda.Device().mem_info[0]
        if bmfunc.use_interp:
            logger.debug(f"After bmfunc, GPU mem avail is: {memnow / 1024**3} GB.")

    coords.setup()
    if debug_enabled:
        memnow = cp.cuda.Device().mem_info[0]
        logger.debug(f"After coords, GPU mem avail is: {memnow / 1024**3} GB.")

    zcalc.setup()
    if debug_enabled:
        memnow = cp.cuda.Device().mem_info[0]
        logger.debug(f"After zcalc, GPU mem avail is: {memnow / 1024**3} GB.")

    matprod.setup()
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
        event_stats = {
            "chunk_total": 0.0,
            "beam": 0.0,
            "tau": 0.0,
            "z": 0.0,
            "matprod": 0.0,
            "chunk_samples": 0,
            "stage_samples": 0,
        }

    vis = np.full((ntimes, matprod.npairs, nfeed, nfeed), 0.0, dtype=ctype)

    logger.info(f"Running With {nchunks} chunks")

    report_chunk = ntimes // 100 + 1
    pr = psutil.Process()
    tstart = time.time()
    mlast = pr.memory_info().rss
    plast = tstart

    for t in range(ntimes):
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
                event_stats["chunk_total"] += cp.cuda.get_elapsed_time(
                    event_start[c], event_end[c]
                )
                event_stats["chunk_samples"] += 1

                if not active_chunks[c]:
                    continue

                event_stats["beam"] += cp.cuda.get_elapsed_time(
                    event_eq2top[c], event_beam[c]
                )
                event_stats["tau"] += cp.cuda.get_elapsed_time(
                    event_beam[c], event_tau[c]
                )
                event_stats["z"] += cp.cuda.get_elapsed_time(event_tau[c], event_z[c])
                event_stats["matprod"] += cp.cuda.get_elapsed_time(
                    event_z[c], event_matprod[c]
                )
                event_stats["stage_samples"] += 1

        # No explicit synchronization needed: sum_chunks' device-to-host copy
        # is ordered on the same stream as all the compute above.
        with nvtx_range("sum_chunks"):
            matprod.sum_chunks(vis[t])
        logdebug("vis", vis[t])

        if not t % report_chunk and t != ntimes - 1:
            plast, mlast = log_progress(tstart, plast, t + 1, ntimes, pr, mlast)

    final_time = time.time()
    LAST_RUN_STATS.clear()
    LAST_RUN_STATS.update(
        {
            "setup_time": tstart - init_time,
            "loop_time": final_time - tstart,
            "ntimes": ntimes,
            "nchunks": nchunks,
            "time_per_integration": (final_time - tstart) / ntimes,
        }
    )

    if gpu_event_timing and event_stats["chunk_samples"] > 0:
        chunk_samples = event_stats["chunk_samples"]
        stage_samples = max(event_stats["stage_samples"], 1)
        LAST_RUN_STATS["event_timing_ms"] = {
            "chunk_total": event_stats["chunk_total"] / chunk_samples,
            "beam": event_stats["beam"] / stage_samples,
            "tau": event_stats["tau"] / stage_samples,
            "z": event_stats["z"] / stage_samples,
            "matprod": event_stats["matprod"] / stage_samples,
        }
        logger.info(
            "GPU event timing (ms): chunk_total=%.3f beam=%.3f tau=%.3f z=%.3f matprod=%.3f",
            event_stats["chunk_total"] / chunk_samples,
            event_stats["beam"] / stage_samples,
            event_stats["tau"] / stage_samples,
            event_stats["z"] / stage_samples,
            event_stats["matprod"] / stage_samples,
        )

    return vis if polarized else vis[:, :, 0, 0]


simulate.__doc__ = (
    (simulate.__doc__ or "")
    + f"\n{simcpu.__doc__ or ''}"
    + """
    gpu_event_timing : bool, optional
        If True, collect per-chunk GPU event timings for beam interpolation,
        tau, Z construction, and matprod stages, and log stage averages at
        INFO level at the end of the run. Default is False.
    """
)
