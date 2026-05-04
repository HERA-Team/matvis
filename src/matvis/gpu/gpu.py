"""GPU implementation of the simulator."""

from __future__ import annotations

import logging
import numpy as np
import psutil
import time
import warnings
from astropy.constants import c as speed_of_light
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from collections.abc import Sequence
from docstring_parser import combine_docstrings
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AnalyticBeam
from pyuvdata.beam_interface import BeamInterface
from typing import Callable, Literal

from .._utils import get_desired_chunks, get_dtypes, log_progress, logdebug
from ..core import _validate_inputs
from ..core.coherency import (
    categorize_sources,
    check_sky_physicality,
    partition_and_negate,
    process_polarized_chunk,
    stokes_to_coherency,
)
from ..core.coords import CoordinateRotation
from ..core.getz import ZMatrixCalc
from ..core.tau import TauCalculator
from ..cpu.cpu import simulate as simcpu
from . import beams
from . import matprod as mp

try:
    import cupy as cp

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


@combine_docstrings(simcpu)
def simulate(
    *,
    antpos: np.ndarray,
    freq: float,
    times: Time,
    skycoords: SkyCoord,
    telescope_loc: EarthLocation,
    beam_list: Sequence[UVBeam | AnalyticBeam | BeamInterface] | None,
    I_sky: np.ndarray | None = None,
    polarized: bool | None = None,
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
    matprod_method: Literal["GPUMatMul", "GPUVectorLoop"] = "GPUMatMul",
    source_buffer: float = 1.0,
    coord_method_params: dict | None = None,
    stokes: np.ndarray | None = None,
    raise_on_negative_flux: bool | None = None,
) -> np.ndarray:
    """GPU implementation of the visibility simulator."""
    if not HAVE_CUDA:
        raise ImportError("You need to install the [gpu] extra to use this function!")

    if source_buffer > 1.0:
        raise ValueError("source_buffer must be less than 1.0")

    pr = psutil.Process()

    if polarized is None:
        polarized = stokes is not None
    elif not polarized and stokes is not None:
        raise ValueError(
            "polarized=False is incompatible with stokes=... — "
            "stokes input implies polarized=True. "
            "Either omit `polarized` or set polarized=True."
        )

    nax, nfeed, nant, ntimes, nsrc = _validate_inputs(
        precision, polarized, antpos, times, I_sky=I_sky, stokes=stokes
    )
    if raise_on_negative_flux is None:
        raise_on_negative_flux = stokes is None

    rtype, ctype = get_dtypes(precision)

    nchunks, npixc = get_desired_chunks(
        min(max_memory, cp.cuda.Device().mem_info[0]),
        min_chunks,
        beam_list,
        nax,
        nfeed,
        nant,
        nsrc,
        precision,
    )

    # Determine if we have a polarized sky model
    polarized_sky = stokes is not None and polarized

    use_sign_split = False
    use_partition = False
    n_P = n_N = 0
    if polarized_sky:
        I_s, Q_s, U_s, V_s = stokes
        use_sign_split = check_sky_physicality(
            I_s, Q_s, U_s, V_s, raise_on_negative=raise_on_negative_flux
        )
        if use_sign_split:
            idx_P, idx_N, idx_M = categorize_sources(I_s, Q_s, U_s, V_s)
            if len(idx_M) == 0:
                use_partition = True
                stokes, skycoords, I_sky, n_P, n_N = partition_and_negate(
                    stokes, skycoords, I_sky
                )
                I_s, Q_s, U_s, V_s = stokes

    if polarized_sky:
        coherency = stokes_to_coherency(I_s, Q_s, U_s, V_s)  # (2, 2, Nsrc)
        flux_for_coords = coherency.transpose(2, 0, 1)[
            :, np.newaxis, :, :
        ]  # (Nsrc, 1, 2, 2)
    else:
        flux_for_coords = np.sqrt(0.5 * I_sky)

    coord_method = CoordinateRotation._methods[coord_method]
    coord_method_params = coord_method_params or {}
    coords = coord_method(
        flux=flux_for_coords,
        times=times,
        telescope_loc=telescope_loc,
        skycoords=skycoords,
        chunk_size=npixc,
        precision=precision,
        source_buffer=source_buffer,
        gpu=True,
        **coord_method_params,
    )
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
    zcalc = ZMatrixCalc(
        nsrc=nsrc_alloc, nfeed=nfeed, nant=nant, nax=nax, ctype=ctype, gpu=True
    )
    taucalc = TauCalculator(
        antpos=antpos, freq=freq, precision=precision, nsrc=nsrc_alloc, gpu=True
    )

    mpcls = getattr(mp, matprod_method)
    matprod = mpcls(nchunks, nfeed, nant, antpairs, precision=precision)

    matprod_neg = None
    if use_sign_split:
        matprod_neg = mpcls(nchunks, nfeed, nant, antpairs, precision=precision)

    logger.debug("Starting GPU allocations...")

    init_mem = cp.cuda.Device().mem_info[0]
    logger.debug(f"Before GPU allocations, GPU mem avail is: {init_mem / 1024**3} GB")

    # antpos here is imaginary and in wavelength units
    taucalc.setup()
    memnow = cp.cuda.Device().mem_info[0]
    logger.debug(f"After antpos, GPU mem avail is: {memnow / 1024**3} GB.")

    bmfunc.setup()
    memnow = cp.cuda.Device().mem_info[0]
    if bmfunc.use_interp:
        logger.debug(f"After bmfunc, GPU mem avail is: {memnow / 1024**3} GB.")

    coords.setup()
    memnow = cp.cuda.Device().mem_info[0]
    logger.debug(f"After coords, GPU mem avail is: {memnow / 1024**3} GB.")

    zcalc.setup()
    memnow = cp.cuda.Device().mem_info[0]
    logger.debug(f"After zcalc, GPU mem avail is: {memnow / 1024**3} GB.")

    matprod.setup()
    if matprod_neg is not None:
        matprod_neg.setup()
    memnow = cp.cuda.Device().mem_info[0]
    logger.debug(f"After matprod, GPU mem avail is: {memnow / 1024**3} GB.")

    # output CPU buffers for downloading answers
    streams = [cp.cuda.Stream() for _ in range(nchunks)]
    event_order = [
        "start",
        "upload",
        "eq2top",
        "tau",
        "beam",
        "meas_eq",
        "vis",
        "end",
    ]

    vis = np.full((ntimes, matprod.npairs, nfeed, nfeed), 0.0, dtype=ctype)

    logger.info(f"Running With {nchunks} chunks")

    report_chunk = ntimes // 100 + 1
    pr = psutil.Process()
    tstart = time.time()
    mlast = pr.memory_info().rss
    plast = tstart

    for t in range(ntimes):
        coords.rotate(t)
        events = [{e: cp.cuda.Event() for e in event_order} for _ in range(nchunks)]

        for c, (stream, event) in enumerate(zip(streams, events)):
            stream.use()
            event["start"].record(stream)

            crdtop, Isqrt, nsrcs_up = coords.select_chunk(c, t)
            logdebug("crdtop", crdtop)
            logdebug("Isqrt", Isqrt)

            if nsrcs_up < 1:
                continue

            event["eq2top"].record(stream)
            logger.debug(
                f"After coords, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
            )

            # Get beam. Shape is (nax, nfeed, nbeam, nsrc_alloc)
            A = bmfunc(crdtop[0], crdtop[1], check=t == 0)
            event["beam"].record(stream)
            logdebug("Beam", A)
            logger.debug(
                f"After beam, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
            )

            # exptau has shape (nant, nsrc)
            exptau = taucalc(crdtop)
            logdebug("exptau", exptau)
            logger.debug(
                f"After exptau, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
            )
            event["tau"].record(stream)

            if polarized_sky:
                n_P_chunk = n_N_chunk = 0
                if use_partition:
                    chunk_start = c * npixc
                    p_local_end = min(max(n_P - chunk_start, 0), npixc)
                    n_local_end = min(max(n_P + n_N - chunk_start, 0), npixc)
                    above = coords.above_horizon
                    # searchsorted on device to avoid host sync.
                    counts = cp.searchsorted(
                        above, cp.asarray([p_local_end, n_local_end])
                    )
                    n_P_chunk = int(counts[0])
                    n_N_chunk = int(counts[1]) - n_P_chunk
                process_polarized_chunk(
                    Isqrt,
                    zcalc,
                    A,
                    exptau,
                    bmfunc.beam_idx,
                    matprod,
                    c,
                    use_sign_split=use_sign_split,
                    matprod_neg=matprod_neg,
                    use_partition=use_partition,
                    n_P_chunk=n_P_chunk,
                    n_N_chunk=n_N_chunk,
                    xp=cp,
                )
            else:
                z = zcalc(Isqrt, A, exptau, bmfunc.beam_idx)
                matprod(z, c)
                logdebug("Z", z)

            event["meas_eq"].record(stream)
            logger.debug(
                f"After Z, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
            )
            logger.debug(
                f"After matprod, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
            )

            event["vis"].record(stream)
            event["end"].record(stream)

        events[nchunks - 1]["end"].synchronize()
        matprod.sum_chunks(vis[t])
        if matprod_neg is not None:
            vis_neg = np.zeros_like(vis[t])
            matprod_neg.sum_chunks(vis_neg)
            vis[t] -= vis_neg
        logdebug("vis", vis[t])

        if not t % report_chunk and t != ntimes - 1:
            plast, mlast = log_progress(tstart, plast, t + 1, ntimes, pr, mlast)

    return vis if polarized else vis[:, :, 0, 0]


simulate.__doc__ += simcpu.__doc__
