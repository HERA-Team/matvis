"""CPU-based implementation of the matvis visibility simulator."""

from __future__ import annotations

import importlib
import logging
import time
import tracemalloc as tm
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
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
from ..redundancy import contiguity_order
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
    beam_list: Sequence[UVBeam | AnalyticBeam | BeamInterface] | None,
    I_sky: np.ndarray | None = None,
    antpairs: np.ndarray | list[tuple[int, int]] | None = None,
    antenna_blocks: list[tuple[np.ndarray, np.ndarray]] | None = None,
    precision: int = 1,
    polarized: bool | None = None,
    beam_idx: np.ndarray | None = None,
    beam_spline_opts: dict | None = None,
    max_progress_reports: int = 100,
    matprod_method: Literal["CPUMatMul", "CPUVectorDot", "CPUMatBlock"] = "CPUMatMul",
    coord_method: Literal[
        "CoordinateRotationAstropy", "CoordinateRotationERFA"
    ] = "CoordinateRotationERFA",
    max_memory: int | float = np.inf,
    min_chunks: int = 1,
    source_buffer: float = 1.0,
    memory_buffer: float = 0.9,
    coord_method_params: dict | None = None,
    stokes: np.ndarray | None = None,
    raise_on_negative_flux: bool | None = None,
):
    """
    Calculate visibility from an input sky model and beam model.

    Parameters
    ----------
    antpos : array_like
        Antenna position array. Shape=(NANT, 3).
    freq : float
        Frequency to evaluate the visibilities at [GHz].
    I_sky : array_like
        Per-source Stokes I values used when a scalar sky model is passed
        (no ``stokes`` argument). The intensity is split equally between
        the two linear polarization channels, introducing a factor of 0.5
        relative to the value given here; this applies even when only one
        polarization channel is simulated. Exactly one of ``I_sky`` or ``stokes`` must be provided.
        Shape=(NSRCS,).
    beam_list : list of UVBeam, optional
        If specified, evaluate primary beam values directly using UVBeam
        objects instead of using pixelized beam maps. Only one of ``bm_cube`` and
        ``beam_list`` should be provided.Note that if `polarized` is True,
        these beams must be efield beams, and conversely if `polarized` is False they
        must be power beams with a single polarization (either XX or YY).
    antpairs : array_like, optional
        Either a 2D array, shape ``(Npairs, 2)``, or list of 2-tuples of ints, with
        the antenna-index pairs (rows of ``antpos``) to return as visibilities (all
        feed-pairs are always calculated). If None, all ``NANT**2`` ordered pairs
        are returned.
    antenna_blocks : list, optional
        Advanced/optional. A list of ``(row_antenna_idx, col_antenna_idx)``
        integer-array tuples defining rectangular sub-matrix blocks to compute
        instead of the full antenna x antenna product; only used when
        ``matprod_method`` is ``CPUMatBlock``. See :mod:`matvis.redundancy`.
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:

        - 1: float32, complex64
        - 2: float64, complex128

    polarized : bool, optional
        Whether to simulate a full polarized response in terms of nn, ne, en,
        ee visibilities. See Eq. 6 of Kohn+ (arXiv:1802.04151) for notation.
        If left as ``None`` (default), inferred from ``stokes``: True when
        ``stokes`` is given, False otherwise. Passing ``polarized=False`` with
        ``stokes`` raises ``ValueError``.
    beam_idx
        Optional length-NANT array specifying a beam index for each antenna.
        By default, either a single beam is assumed to apply to all antennas or
        each antenna gets its own beam.
    beam_spline_opts : dict, optional
        Options for interpolating gridded (``UVBeam``) beams. Passed through to
        :func:`scipy.ndimage.map_coordinates` on the CPU backend, and to its GPU
        equivalent on the GPU backend. Keys left out fall back to
        :data:`~matvis.core.beams.DEFAULT_SPLINE_OPTS`, currently
        ``{"order": 3, "mode": "nearest"}``.

        On the GPU backend only ``order`` 1 (bilinear) and 3 (bicubic) have
        dedicated fused kernels. Any other order falls back to a
        per-(beam, feed, axis) ``map_coordinates`` loop that issues a separate
        kernel launch for every combination -- hundreds per source chunk at
        production scale -- and is *much* slower than either fused kernel. Orders
        0, 2, 4 and 5 are supported for completeness, not for production use.

        See :doc:`/beam_interpolation` for how to choose an order, and for the
        behaviour at the edges of the beam grid.
    max_progress_reports : int, optional
        Maximum number of progress reports to print to the screen (if logging level
        allows). Default is 100.
    matprod_method : str, optional
        The method to use for the final matrix multiplication. Default is 'CPUMatMul',
        which simply uses `np.dot` over the two full matrices. Currently, the other
        option is `CPUVectorDot`, which uses a loop over the antenna pairs,
        computing the sum over sources as a vector dot product.
        Whether to calculate visibilities for each antpair in antpairs as a vector
        dot-product instead of using a full matrix-matrix multiplication for all
        possible pairs. Default is False. Setting to True can be faster for large
        arrays where `antpairs` is small (possibly from high redundancy). You should
        run a performance test before using this.
    coord_method : str, optional
        The method to use to transform coordinates from the equatorial to horizontal
        frame. The default is "CoordinateRotationERFA". Use
        "CoordinateRotationAstropy" for direct Astropy coordinate transforms.
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
    stokes : array_like, optional
        Full Stokes parameters of shape (4, NSRCS) with [I, Q, U, V].
        Setting ``stokes`` automatically enables ``polarized=True`` and
        routes through the eigendecomposition of the coherency matrix;
        passing ``polarized=False`` alongside is an error. If ``None``
        (default), uses ``I_sky`` as Stokes I only (existing behavior).
        Exactly one of ``I_sky`` or ``stokes`` must be provided.
    raise_on_negative_flux : bool, optional
        How to handle negative eigenvalues in the coherency matrix.
        Defaults to False for Stokes input and True for scalar input.
        If True, raise ValueError if any coherency eigenvalue is negative.
        If False, use sign-split decomposition to handle negative eigenvalues
        (needed for EoR-like sky models with negative Stokes I).

    Returns
    -------
    vis : array_like
        Simulated visibilities. If `polarized = True`, the output will have
        shape (NTIMES, NPAIRS, NFEED, NFEED), otherwise it will have
        shape (NTIMES, NPAIRS). The pair order, feed order and visibility
        convention are those of :func:`matvis.simulate_vis`, for one frequency.

    Notes
    -----
    Three sky-model modes are supported:

    1. ``polarized=False`` — single-feed calculation using ``I_sky``.
    2. ``polarized=True, stokes=None`` — uses ``I_sky`` as Stokes I only,
       split 50/50 across the two feeds (legacy behavior).
    3. ``polarized=True, stokes.shape == (4, NSRCS)`` — full-Stokes
       visibility via eigendecomposition of the per-source coherency
       matrix ``C = 0.5 * [[I+Q, U+iV], [U-iV, I-Q]]``.

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
        nsrc,
        precision,
        source_buffer=source_buffer,
        memory_buffer=memory_buffer,
        polarized_sky=polarized_sky,
        sign_split=use_sign_split,
    )
    _mark("chunk_planning")

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

    # Relabelling the antenna axis is free in the Z construction but lets the
    # block-decomposed matprod slice most of its operands straight out of Z
    # instead of gathering them (issue #161). Both ends must agree on the
    # labelling, so the same array goes to the matprod and to the Z calculator.
    antenna_order = (
        contiguity_order(antenna_blocks, nant) if antenna_blocks is not None else None
    )

    mpcls = getattr(mp, matprod_method)
    matprod = mpcls(
        nchunks,
        nfeed,
        nant,
        antpairs,
        precision=precision,
        antenna_blocks=antenna_blocks,
        antenna_order=antenna_order,
    )
    zcalc = ZMatrixCalc(
        nsrc=nsrc_alloc,
        nfeed=nfeed,
        nant=nant,
        nax=nax,
        ctype=ctype,
        antenna_order=antenna_order,
    )

    # For sign-split, allocate a second matprod for negative eigenvalue contributions
    matprod_neg = None
    if use_sign_split:
        matprod_neg = mpcls(
            nchunks,
            nfeed,
            nant,
            antpairs,
            precision=precision,
            antenna_blocks=antenna_blocks,
            antenna_order=antenna_order,
        )

    vis = np.full((ntimes, matprod.npairs, nfeed, nfeed), 0.0, dtype=ctype)
    _mark("vis_alloc")

    bmfunc.setup()
    _mark("beam_setup")
    coords.setup()
    _mark("coord_setup")
    matprod.setup()
    if matprod_neg is not None:
        matprod_neg.setup()
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

    @contextmanager
    def time_stage(name: str) -> Iterator[None]:
        started = time.perf_counter()
        yield
        stage_samples[name].append(time.perf_counter() - started)

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

            if polarized_sky:
                n_P_chunk = n_N_chunk = 0
                if use_partition:
                    chunk_start = c * npixc
                    p_local_end = min(max(n_P - chunk_start, 0), npixc)
                    n_local_end = min(max(n_P + n_N - chunk_start, 0), npixc)
                    above = coords.above_horizon
                    n_P_chunk = int(np.searchsorted(above, p_local_end))
                    n_PN_chunk = int(np.searchsorted(above, n_local_end))
                    n_N_chunk = n_PN_chunk - n_P_chunk
                process_polarized_chunk(
                    flux_sqrt,
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
                    stage=time_stage,
                )
            else:
                with time_stage("z"):
                    z = zcalc(flux_sqrt, A, exptau, bmfunc.beam_idx)
                with time_stage("matprod"):
                    matprod(z, c)
                logdebug("Z", z[..., :nn])

            if not t % report_chunk and t != ntimes - 1 and c == nchunks - 1:
                plast, mlast = log_progress(tstart, plast, t + 1, ntimes, pr, mlast)
                highest_peak = memtrace(highest_peak)

        _t = time.perf_counter()
        matprod.sum_chunks(vis[t])
        if matprod_neg is not None:
            vis_neg = np.zeros_like(vis[t])
            matprod_neg.sum_chunks(vis_neg)
            vis[t] -= vis_neg
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
