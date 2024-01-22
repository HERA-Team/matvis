"""CPU-based implementation of the matvis visibility simulator."""
from __future__ import annotations

import logging
import numpy as np
import psutil
import time
import tracemalloc as tm
from astropy.constants import c as speed_of_light
from collections.abc import Sequence

# from pympler import tracker
from pyuvdata import UVBeam
from typing import Callable

from .._utils import get_desired_chunks, get_dtypes, log_progress, logdebug, memtrace
from ..core import _validate_inputs
from ..core.getz import ZMatrixCalc
from . import matprod as mp
from .beams import UVBeamInterpolator
from .coords import CPUCoordinateRotation

logger = logging.getLogger(__name__)


def simulate(
    *,
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam | Callable] | None,
    antpairs: np.ndarray | list[tuple[int, int]] | None = None,
    matsets: list[tuple[np.ndarray[int], np.ndarray[int]]] | None = None,
    precision: int = 1,
    polarized: bool = False,
    beam_idx: np.ndarray | None = None,
    beam_spline_opts: dict | None = None,
    max_progress_reports: int = 100,
    matprod_method: str = "CPUMatMul",
    max_memory: int | float = np.inf,
    min_chunks: int = 1,
    source_buffer: float = 1.0,
):
    """
    Calculate visibility from an input intensity map and beam model.

    Parameters
    ----------
    antpos : array_like
        Antenna position array. Shape=(NANT, 3).
    freq : float
        Frequency to evaluate the visibilities at [GHz].
    eq2tops : array_like
        Set of 3x3 transformation matrices to rotate the RA and Dec
        cosines in an ECI coordinate system (see `crd_eq`) to
        topocentric ENU (East-North-Up) unit vectors at each
        time/LST/hour angle in the dataset.
        Shape=(NTIMES, 3, 3).
    crd_eq : array_like
        Cartesian unit vectors of sources in an ECI (Earth Centered
        Inertial) system, which has the Earth's center of mass at
        the origin, and is fixed with respect to the distant stars.
        The components of the ECI vector for each source are:
        (cos(RA) cos(Dec), sin(RA) cos(Dec), sin(Dec)).
        Shape=(3, NSRCS).
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
    matsets  : array_like, optional
        A list of containing a set of 2-tuples of numpy arrays.  Each entry in the list
        corresponds to a different sub-matrix.  The first element of the ith tuple lists the rows of Z corresponding
        to the rows of the ith sub-matrix, and the second tuple element contains the list of columns.  In the case
        of vector-vector loop, the number of tuples is equal to Npairs and each element of the tuple contains only
        one int, such that the sub-matrices contain only one element each.  Outer dimension has to be a list because
        numpy doesn't like inhomogeneous arrays.
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
        which simply uses `np.dot` over the two full matrices. The second option is
        `CPUVectorLoop`, which uses a loop over the antenna pairs, computing the sum
        over sources as a vector dot product.  The third option is `CPUMatChunk`, which
        divides the product into several matrix multiplications that are optimized to have the
        highest possible density of non-redundant pairs without invoking the overhead of a large
        for loop over vector products.  For very large arrays, with very high redundancy, CPUVectorLoop
        is usually fastest, while for cases with low redundancy, `CPUMatMul` is fastest.  The `CPUMatChunk`
        can sometimes be fastest for intermediate levels of redundancy and intermediately sized arrays.  You should
        run a performance test before choosing one of these options.
    max_memory : int, optional
        The maximum memory (in bytes) to use for the visibility calculation. This is
        not a hard-set limit, but rather a guideline for how much memory to use. If the
        expected memory usage is more than this, the calculation will be broken up into
        chunks. Default is 512 MB.
    min_chunks : int, optional
        The minimum number of chunks to break the source axis into. Default is 1.
    source_buffer : float, optional
        The fraction of the total sources (per chunk) to pre-allocate memory for.
        Default is 0.55, which allows for 10% variance around half the sources
        (since half should be below the horizon). If you have a particular sky model in
        which you expect more or less sources to appear above the horizon at any time,
        set this to a different value.

    Returns
    -------
    vis : array_like
        Simulated visibilities. If `polarized = True`, the output will have
        shape (NTIMES, NBLS, NFEED, NFEED), otherwise it will have
        shape (NTIMES, NBLS).

    """
    if not tm.is_tracing() and logger.isEnabledFor(logging.INFO):
        tm.start()

    highest_peak = memtrace(0)

    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, eq2tops, crd_eq, I_sky
    )

    rtype, ctype = get_dtypes(precision)

    nchunks, npixc = get_desired_chunks(
        min(max_memory, psutil.virtual_memory().available),
        min_chunks,
        beam_list,
        nax,
        nfeed,
        nant,
        len(I_sky),
        precision,
        source_buffer=source_buffer,
    )
    nsrc_alloc = int(npixc * source_buffer)

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

    coords = CPUCoordinateRotation(
        flux=np.sqrt(0.5 * I_sky),
        crd_eq=crd_eq,
        eq2top=eq2tops,
        chunk_size=npixc,
        precision=precision,
        source_buffer=source_buffer,
    )
    mpcls = getattr(mp, matprod_method)
    matprod = mpcls(nchunks, nfeed, nant, antpairs, matsets, precision=precision)
    zcalc = ZMatrixCalc(
        nsrc=nsrc_alloc,
        nfeed=nfeed,
        nant=nant,
        nax=nax,
        ctype=ctype,
    )

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky. Factor of 0.5 accounts for splitting Stokes I between
    # polarization channels
    ang_freq = rtype(2.0 * np.pi * freq)
    antpos_u = antpos.astype(rtype) * ang_freq / speed_of_light.value

    vis = np.full((ntimes, matprod.npairs, nfeed, nfeed), 0.0, dtype=ctype)

    bmfunc.setup()
    coords.setup()
    matprod.setup()
    zcalc.setup()

    logger.info(f"Visibility Array takes {vis.nbytes/1024**2:.1f} MB")

    # Have up to 100 reports as it iterates through time.
    report_chunk = ntimes // max_progress_reports + 1
    pr = psutil.Process()
    tstart = time.time()
    mlast = pr.memory_info().rss
    plast = tstart

    highest_peak = memtrace(highest_peak)

    # Loop over time samples
    for t in range(ntimes):
        coords.rotate(t)

        for c in range(nchunks):
            crd_top, flux_sqrt, nn = coords.select_chunk(c)
            logdebug("crdtop", crd_top[:, :nn])
            logdebug("Isqrt", flux_sqrt[:nn])

            A = bmfunc(crd_top[0], crd_top[1], check=t == 0)
            logdebug("beam", bmfunc.interpolated_beam[..., :nn])

            # Calculate delays, where tau = 2pi*nu*(b * s) / c
            exptau = np.exp(1j * np.dot(antpos_u, crd_top))
            logdebug("exptau", exptau[:, :nn])

            zcalc.compute(flux_sqrt, A, exptau, bmfunc.beam_idx)
            logdebug("Z", zcalc.z[..., :nn])

            matprod(zcalc.z, c)

            if not (t % report_chunk or t == ntimes - 1):
                plast, mlast = log_progress(tstart, plast, t + 1, ntimes, pr, mlast)
                highest_peak = memtrace(highest_peak)

        matprod.sum_chunks(vis[t])
        logdebug("vis", vis[t])

    return vis if polarized else vis[:, :, 0, 0]
