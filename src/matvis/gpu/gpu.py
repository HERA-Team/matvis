"""GPU implementation of the simulator."""
from __future__ import annotations

import logging
import numpy as np
import psutil
import time
import warnings
from astropy.constants import c as speed_of_light
from collections.abc import Sequence
from pyuvdata import UVBeam
from typing import Callable

from .._utils import get_desired_chunks, get_dtypes, log_progress, logdebug
from ..core import _validate_inputs
from ..cpu.cpu import simulate as simcpu
from . import beams
from . import matprod as mp
from .coords import GPUCoordinateRotation
from .getz import GPUZMatrixCalc

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


def simulate(
    *,
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam | Callable] | None,
    polarized: bool = False,
    antpairs: np.ndarray | list[tuple[int, int]] | None = None,
    beam_idx: np.ndarray | None = None,
    max_memory: int = np.inf,
    min_chunks: int = 1,
    precision: int = 1,
    beam_spline_opts: dict | None = None,
    matprod_method: str = "GPUMatMul",
    source_buffer: float = 1.0,
) -> np.ndarray:
    """GPU implementation of the visibility simulator."""
    if not HAVE_CUDA:
        raise ImportError("You need to install the [gpu] extra to use this function!")

    if source_buffer > 1.0:
        raise ValueError("source_buffer must be less than 1.0")

    pr = psutil.Process()
    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, eq2tops, crd_eq, I_sky
    )

    if beam_spline_opts:
        warnings.warn(
            "You have passed beam_spline_opts, but these are not used in GPU.",
            stacklevel=1,
        )

    nsrc = len(I_sky)

    rtype, ctype = get_dtypes(precision)

    # ensure data types
    antpos = antpos.astype(rtype)

    nchunks, npixc = get_desired_chunks(
        min(max_memory, cp.cuda.Device().mem_info[0]),
        min_chunks,
        beam_list,
        nax,
        nfeed,
        nant,
        len(I_sky),
        precision,
    )

    nsrc_alloc = int(npixc * source_buffer)

    bmfunc = beams.GPUBeamInterpolator(
        beam_list=beam_list,
        beam_idx=beam_idx,
        polarized=polarized,
        nant=nant,
        freq=freq,
        nsrc=nsrc_alloc,
        precision=precision,
    )
    coords = GPUCoordinateRotation(
        flux=np.sqrt(0.5 * I_sky),
        crd_eq=crd_eq,
        eq2top=eq2tops,
        chunk_size=npixc,
        precision=precision,
        source_buffer=source_buffer,
    )
    zcalc = GPUZMatrixCalc(
        nsrc=nsrc_alloc,
        nfeed=nfeed,
        nant=nant,
        nax=nax,
        ctype=ctype,
    )
    mpcls = getattr(mp, matprod_method)
    matprod = mpcls(nchunks, nfeed, nant, antpairs, precision=precision)

    npixc = nsrc // nchunks

    logger.debug("Starting GPU allocations...")

    init_mem = cp.cuda.Device().mem_info[0]
    logger.debug(f"Before GPU allocations, GPU mem avail is: {init_mem / 1024**3} GB")

    # antpos here is imaginary and in wavelength units
    antpos = 2 * np.pi * freq * 1j * cp.asarray(antpos) * ONE_OVER_C
    memnow = cp.cuda.Device().mem_info[0]
    logger.debug(f"After antpos, GPU mem avail is: {memnow / 1024**3} GB.")

    bmfunc.setup()
    memnow = cp.cuda.Device().mem_info[0]
    if bmfunc.use_interp:
        logger.debug(f"After bmfunc, GPU mem avail is: {memnow / 1024**3} GB.")

    exptau = cp.zeros((nant, nsrc_alloc), dtype=ctype)

    coords.setup()
    memnow = cp.cuda.Device().mem_info[0]
    logger.debug(f"After coords, GPU mem avail is: {memnow / 1024**3} GB.")

    zcalc.setup()
    memnow = cp.cuda.Device().mem_info[0]
    logger.debug(f"After zcalc, GPU mem avail is: {memnow / 1024**3} GB.")

    matprod.setup()
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

            crdtop, Isqrt, nsrcs_up = coords.select_chunk(c)
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
            cp.exp(cp.matmul(antpos, crdtop, out=exptau), out=exptau)
            logdebug("exptau", exptau)
            logger.debug(
                f"After exptau, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
            )
            event["tau"].record(stream)

            zcalc.compute(Isqrt, A, exptau, bmfunc.beam_idx)
            event["meas_eq"].record(stream)
            logdebug("Z", zcalc.z)
            logger.debug(
                f"After Z, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
            )

            # compute vis = Z.Z^dagger
            matprod(zcalc.z, c)
            logger.debug(
                f"After matprod, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
            )

            event["vis"].record(stream)
            event["end"].record(stream)

        events[nchunks - 1]["end"].synchronize()
        matprod.sum_chunks(vis[t])
        logdebug("vis", vis[t])

        if not t % report_chunk and t != ntimes - 1:
            plast, mlast = log_progress(tstart, plast, t + 1, ntimes, pr, mlast)

    return vis if polarized else vis[:, :, 0, 0]


simulate.__doc__ += simcpu.__doc__
