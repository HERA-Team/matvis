"""CPU-based implementation of the visibility simulator."""
from __future__ import annotations

import gc
import linecache
import logging
import numpy as np
import psutil
import time
import tracemalloc as tm
from astropy.constants import c

# from pympler import tracker
from pyuvdata import UVBeam
from typing import Callable, Sequence

from . import conversions

# This enables us to put in profile decorators that will be no-ops if no profiling
# library is being used.
try:
    profile
except NameError:
    from ._utils import no_op as profile


logger = logging.getLogger(__name__)


def _wrangle_beams(
    beam_idx: np.ndarray,
    beam_list: list[UVBeam],
    polarized: bool,
    nant: int,
    freq: float,
) -> tuple[list[UVBeam], int, np.ndarray]:
    """Perform all the operations and checks on the input beams.

    Checks that the beam indices match the number of antennas, pre-interpolates to the
    given frequency, and checks that the beam type is appropriate for the given
    polarization

    Parameters
    ----------
    beam_idx
        Index of the beam to use for each antenna.
    beam_list
        List of unique beams.
    polarized
        Whether to use beam polarization
    nant
        Number of antennas
    freq
        Frequency to interpolate beam to.
    """
    # Get the number of unique beams
    nbeam = len(beam_list)

    # Check the beam indices
    if beam_idx is None:
        if nbeam == 1:
            beam_idx = np.zeros(nant, dtype=int)
        elif nbeam == nant:
            beam_idx = np.arange(nant, dtype=int)
        else:
            raise ValueError(
                "If number of beams provided is not 1 or nant, beam_idx must be provided."
            )
    else:
        assert beam_idx.shape == (nant,), "beam_idx must be length nant"
        assert all(
            0 <= i < nbeam for i in beam_idx
        ), "beam_idx contains indices greater than the number of beams"

    # make sure we interpolate to the right frequency first.
    beam_list = [
        bm.interp(freq_array=np.array([freq]), new_object=True, run_check=False)
        if isinstance(bm, UVBeam)
        else bm
        for bm in beam_list
    ]

    if polarized and any(b.beam_type != "efield" for b in beam_list):
        raise ValueError("beam type must be efield if using polarized=True")
    elif not polarized and any(
        (
            b.beam_type != "power"
            or getattr(b, "Npols", 1) > 1
            or b.polarization_array[0] not in [-5, -6]
        )
        for b in beam_list
    ):
        raise ValueError(
            "beam type must be power and have only one pol (either xx or yy) if polarized=False"
        )

    return beam_list, nbeam, beam_idx


def _evaluate_beam_cpu(
    A_s: np.ndarray,
    beam_list: list[UVBeam],
    tx: np.ndarray,
    ty: np.ndarray,
    polarized: bool,
    freq: float,
    check: bool = False,
    spline_opts: dict | None = None,
):
    """Evaluate the beam on the CPU.

    This function will either interpolate the beam to the given coordinates tx, ty,
    or evaluate the beam there if it is an analytic beam.

    Parameters
    ----------
    A_s
        Array of shape (nax, nfeed, nbeam, nsrcs_up) that will be filled with beam
        values.
    beam_list
        List of unique beams.
    tx, ty
        Coordinates to evaluate the beam at, in sin-projection.
    polarized
        Whether to use beam polarization.
    freq
        Frequency to interpolate beam to.
    check
        Whether to check that the beam has no inf/nan values. Set to False if you are
        sure that the beam is valid, as it will be faster.
    spline_opts
        Extra options to pass to the RectBivariateSpline class when interpolating.
    """
    # Primary beam pattern using direct interpolation of UVBeam object
    az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")
    for i, bm in enumerate(beam_list):
        kw = (
            {
                "reuse_spline": True,
                "check_azza_domain": False,
                "spline_opts": spline_opts,
            }
            if isinstance(bm, UVBeam)
            else {}
        )

        interp_beam = bm.interp(
            az_array=az,
            za_array=za,
            freq_array=np.atleast_1d(freq),
            **kw,
        )[0]

        if polarized:
            interp_beam = interp_beam[:, 0, :, 0, :]
        else:
            # Here we have already asserted that the beam is a power beam and
            # has only one polarization, so we just evaluate that one.
            interp_beam = np.sqrt(interp_beam[0, 0, 0, 0, :])

        A_s[:, :, i] = interp_beam

        # Check for invalid beam values
        if check:
            sm = np.sum(A_s)
            if np.isinf(sm) or np.isnan(sm):
                raise ValueError("Beam interpolation resulted in an invalid value")

    return A_s


def _validate_inputs(precision, polarized, antpos, eq2tops, crd_eq, I_sky):
    assert precision in {1, 2}

    # Specify number of polarizations (axes/feeds)
    if polarized:
        nax = nfeed = 2
    else:
        nax = nfeed = 1

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)."
    ncrd, nsrcs = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NSRCS)."
    assert (
        I_sky.ndim == 1 and I_sky.shape[0] == nsrcs
    ), "I_sky must have shape (NSRCS,)."

    return nax, nfeed, nant, ntimes


@profile
def vis_cpu(
    *,
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam | Callable] | None,
    precision: int = 1,
    polarized: bool = False,
    beam_idx: np.ndarray | None = None,
    beam_spline_opts: dict | None = None,
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

    Returns
    -------
    vis : array_like
        Simulated visibilities. If `polarized = True`, the output will have
        shape (NTIMES, NFEED, NFEED, NANTS, NANTS), otherwise it will have
        shape (NTIMES, NANTS, NANTS).
    """
    tm.start()

    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, eq2tops, crd_eq, I_sky
    )

    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    beam_list, nbeam, beam_idx = _wrangle_beams(
        beam_idx, beam_list, polarized, nant, freq
    )

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky. Factor of 0.5 accounts for splitting Stokes I between
    # polarization channels
    Isqrt = np.sqrt(0.5 * I_sky).astype(real_dtype)
    antpos = antpos.astype(real_dtype)

    ang_freq = real_dtype(2.0 * np.pi * freq)

    # Zero arrays: beam pattern, visibilities, delays, complex voltages
    vis = np.full((ntimes, nfeed * nant, nfeed * nant), 0.0, dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    # Have up to 100 reports as it iterates through time.
    report_chunk = ntimes // 100 + 1
    pr = psutil.Process()
    tstart = time.time()
    mlast = pr.memory_info().rss
    plast = tstart

    #    tr = tracker.SummaryTracker()

    snapshot1 = tm.take_snapshot()
    # Loop over time samples
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        # Dot product converts ECI cosines (i.e. from RA and Dec) into ENU
        # (topocentric) cosines, with (tx, ty, tz) = (e, n, u) components
        # relative to the center of the array
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)
        above_horizon = tz > 0
        tx = tx[above_horizon]
        ty = ty[above_horizon]
        nsrcs_up = len(tx)
        isqrt = Isqrt[above_horizon]

        A_s = np.full((nax, nfeed, nbeam, nsrcs_up), 0.0, dtype=complex_dtype)

        _evaluate_beam_cpu(
            A_s,
            beam_list,
            tx,
            ty,
            polarized,
            freq,
            check=t == 0,
            spline_opts=beam_spline_opts,
        )
        A_s = A_s.transpose((1, 2, 0, 3))  # Now (Nfeed, Nbeam, Nax, Nsrc)

        _log_array("beam", A_s)

        # Calculate delays, where tau = (b * s) / c
        tau = np.dot(antpos / c.value, crd_top[:, above_horizon])
        _log_array("tau", tau)

        v = _get_antenna_vis(
            A_s, ang_freq, tau, isqrt, beam_idx, nfeed, nant, nax, nsrcs_up
        )
        _log_array("vant", v)

        # Compute visibilities using product of complex voltages (upper triangle).
        vis[t] = v.conj().dot(v.T)
        _log_array("vis", vis[t])

        gc.collect()
        if not t % report_chunk or t == ntimes - 1:
            plast, mlast = _log_progress(tstart, plast, t + 1, ntimes, pr, mlast)

            # tr.print_diff()
            snapshot2 = tm.take_snapshot()
            display_top(snapshot1, snapshot2)
            snapshot1 = snapshot2

    vis.shape = (ntimes, nfeed, nant, nfeed, nant)

    # Return visibilities with or without multiple polarization channels
    return vis.transpose((0, 1, 3, 2, 4)) if polarized else vis[:, 0, :, 0, :]


def display_top(sn1, sn2, key_type="lineno", limit=10):
    """Display top N malloc differences between snapshots."""
    sn1 = sn1.filter_traces(
        (
            tm.Filter(False, "<frozen importlib._bootstrap>"),
            tm.Filter(False, "<unknown>"),
        )
    )
    sn2 = sn2.filter_traces(
        (
            tm.Filter(False, "<frozen importlib._bootstrap>"),
            tm.Filter(False, "<unknown>"),
        )
    )

    top_stats = sn2.compare_to(sn1, key_type)

    msg = "\n"

    def _get_size_unit(size):
        if size > 1024**3:
            size = size / 1024**3
            unit = "GB"
        elif size > 1024**2:
            size = size / 1024**2
            unit = "MB"
        elif size > 1024**1:
            size = size / 1024**1
            unit = "KB"
        else:
            size = size
            unit = "B"
        return size, unit

    msg += f"Top {limit} lines:\n"
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        last_frame = stat.traceback[-1]

        size, unit = _get_size_unit(stat.size)
        sized, dunit = _get_size_unit(stat.size_diff)
        msg += f"#{index:>02}: {frame.filename}:{frame.lineno}\n"
        msg += f"  -> {size:.1f} {unit}, {stat.count:>4} | {sized:+.1f} {dunit}, {stat.count_diff:+d}\n"

        if line := linecache.getline(frame.filename, frame.lineno).strip():
            msg += f"  ->           Line: {line}\n"

        if line := linecache.getline(last_frame.filename, last_frame.lineno).strip():
            msg += f"  -> Top-Level Line: {line}\n"

    msg += "\n"
    if other := top_stats[limit:]:
        size, unit = _get_size_unit(sum(stat.size for stat in other))
        sized, dunit = _get_size_unit(sum(stat.size_diff for stat in other))
        msg += f"> {len(other)} others: {size:.1f} {unit} | {sized:+.1f} {dunit}\n"

    size, unit = _get_size_unit(sum(stat.size for stat in top_stats))
    sized, dunit = _get_size_unit(sum(stat.size_diff for stat in top_stats))
    msg += f"Total allocated size: {size:.1f} {unit} | {sized:+.1f} {dunit}\n"
    logger.info(msg)


def _get_antenna_vis(
    A_s, ang_freq, tau, Isqrt, beam_idx, nfeed, nant, nax, nsrcs_up
) -> np.ndarray:
    """Compute the antenna-wise visibility integrand."""
    # Component of complex phase factor for one antenna
    # (actually, b = (antpos1 - antpos2) * crd_top / c; need dot product
    # below to build full phase factor for a given baseline)
    v = np.exp(1.0j * (ang_freq * tau)) * Isqrt

    # A_s has shape (Nfeed, Nbeams, Nax, Nsources)
    # v has shape (Nants, Nsources) and is sqrt(I)*exp(1j tau*nu)
    # Here we expand A_s to all ants (from its beams), then broadcast to v, so we
    # end up with shape (Nax, Nfeed, Nants, Nsources)
    v = A_s[:, beam_idx] * v[np.newaxis, :, np.newaxis, :]  # ^ but Nbeam -> Nant
    return v.reshape((nfeed * nant, nax * nsrcs_up))  # reform into matrix


def _log_array(name, x):
    """Debug logging of the value of an array."""
    if logger.getEffectiveLevel() <= logging.DEBUG:  # pragma: no cover
        logger.debug(
            f"CPU: {name}: {x.flatten() if x.size < 40 else x.flatten()[:40]} {x.shape}"
        )


def _log_progress(start_time, prev_time, iters, niters, pr, last_mem):
    """Logging of progress."""
    if logger.getEffectiveLevel() > logging.INFO:
        return prev_time, last_mem

    t = time.time()
    lapsed = t - prev_time
    if lapsed > 120:
        lapsed /= 60
        minutes = True
    else:
        minutes = False
    total = t - start_time
    if total > 3600:
        total /= 3600
        tunit = "hrs"
    elif total > 60:
        total /= 60
        tunit = "min"
    else:
        tunit = "sec"
    expected = total * niters / iters
    rss = pr.memory_info().rss
    if rss > 1024**3:
        mem = rss / 1024**3
        munit = "GB"
    elif rss > 1024**2:
        mem = rss / 1024**2
        munit = "MB"
    else:
        mem = rss
        munit = "KB"
    memdiff = rss - last_mem
    if memdiff > 1024**3:
        memdiff /= 1024**3
        mdunit = "GB"
    elif rss > 1024**2:
        memdiff /= 1024**2
        mdunit = "MB"
    else:
        mdunit = "KB"

    logger.info(
        f"""
        Progress Info   [{iters}/{niters} times ({100 * iters / niters:.1f}%)]
            -> Update Time:   {lapsed:.2f} {'min' if minutes else 'sec'}
            -> Total Time:    {total:.2f} {tunit}  [{total/iters:.2f} {tunit} per integration]
            -> Expected Time: {expected:.2f} {tunit}  [{expected - total:.2f} remaining]
            -> Memory Usage:  {mem:.2f} {munit}   [{memdiff:+.2f} {mdunit}]
        """
    )

    return t, rss
