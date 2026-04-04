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
    coherency_to_stokes,
    compute_m_matrix_eigen,
    compute_m_matrix_sign_split,
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
    matprod_method: Literal["GPUMatMul", "GPUVectorLoop"] = "GPUMatMul",
    source_buffer: float = 1.0,
    coord_method_params: dict | None = None,
    stokes: np.ndarray | None = None,
    negative_flux: Literal["raise", "split", "ignore"] = "raise",
) -> np.ndarray:
    """GPU implementation of the visibility simulator."""
    if not HAVE_CUDA:
        raise ImportError("You need to install the [gpu] extra to use this function!")

    if source_buffer > 1.0:
        raise ValueError("source_buffer must be less than 1.0")

    pr = psutil.Process()
    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, times, I_sky, stokes=stokes
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
    )

    nsrc_alloc = int(npixc * source_buffer)

    # Determine if we have a polarized sky model
    polarized_sky = stokes is not None and polarized

    if polarized_sky:
        I_s, Q_s, U_s, V_s = stokes
        coherency = stokes_to_coherency(I_s, Q_s, U_s, V_s)  # (2, 2, Nsrc)
        flux_for_coords = coherency.transpose(2, 0, 1)[
            :, np.newaxis, :, :
        ]  # (Nsrc, 1, 2, 2)
    else:
        flux_for_coords = np.sqrt(0.5 * I_sky)

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
    zcalc = ZMatrixCalc(
        nsrc=nsrc_alloc, nfeed=nfeed, nant=nant, nax=nax, ctype=ctype, gpu=True
    )
    taucalc = TauCalculator(
        antpos=antpos, freq=freq, precision=precision, nsrc=nsrc_alloc, gpu=True
    )

    mpcls = getattr(mp, matprod_method)
    matprod = mpcls(nchunks, nfeed, nant, antpairs, precision=precision)

    matprod_neg = None
    if polarized_sky and negative_flux == "split":
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
                # Isqrt is rotated coherency: shape (nsrc_alloc, 1, 2, 2)
                C_rot = Isqrt[:, 0]  # (nsrc_alloc, 2, 2)
                I_r, Q_r, U_r, V_r = coherency_to_stokes(
                    C_rot.transpose(1, 2, 0), xp=cp
                )

                if negative_flux == "split":
                    M_pos, M_neg, has_neg = compute_m_matrix_sign_split(
                        I_r, Q_r, U_r, V_r, xp=cp
                    )
                    z = zcalc(None, A, exptau, bmfunc.beam_idx, m_matrix=M_pos)
                    matprod(z, c)
                    if has_neg:
                        z = zcalc(None, A, exptau, bmfunc.beam_idx, m_matrix=M_neg)
                        matprod_neg(z, c)
                elif negative_flux == "ignore":
                    M_pos, _, _ = compute_m_matrix_sign_split(
                        I_r, Q_r, U_r, V_r, xp=cp
                    )
                    z = zcalc(None, A, exptau, bmfunc.beam_idx, m_matrix=M_pos)
                    matprod(z, c)
                elif negative_flux == "raise":
                    M = compute_m_matrix_eigen(I_r, Q_r, U_r, V_r, xp=cp)
                    z = zcalc(None, A, exptau, bmfunc.beam_idx, m_matrix=M)
                    matprod(z, c)
                else:
                    raise ValueError(
                        "negative_flux must be 'raise', 'split', or 'ignore'"
                    )
            else:
                z = zcalc(Isqrt, A, exptau, bmfunc.beam_idx)
                matprod(z, c)

            event["meas_eq"].record(stream)
            logdebug("Z", z)
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
        if polarized_sky and negative_flux == "split" and matprod_neg is not None:
            vis_neg = np.zeros_like(vis[t])
            matprod_neg.sum_chunks(vis_neg)
            vis[t] -= vis_neg
        logdebug("vis", vis[t])

        if not t % report_chunk and t != ntimes - 1:
            plast, mlast = log_progress(tstart, plast, t + 1, ntimes, pr, mlast)

    return vis if polarized else vis[:, :, 0, 0]


simulate.__doc__ += simcpu.__doc__
