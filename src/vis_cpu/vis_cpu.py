"""CPU-based implementation of the visibility simulator."""

import numpy as np
from astropy.constants import c
from scipy.interpolate import RectBivariateSpline
from typing import Optional, Sequence

from . import conversions


def vis_cpu(
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    bm_cube: Optional[np.ndarray] = None,
    beam_list: Optional[Sequence[np.ndarray]] = None,
    precision: int = 1,
):
    """
    Calculate visibility from an input intensity map and beam model.

    Provided as a standalone function.

    Parameters
    ----------
    antpos : array_like
        Antenna position array. Shape=(NANT, 3).
    freq : float
        Frequency to evaluate the visibilities at [GHz].
    eq2tops : array_like
        Set of 3x3 transformation matrices converting equatorial
        coordinates to topocentric at each
        hour angle (and declination) in the dataset.
        Shape=(NTIMES, 3, 3).
    crd_eq : array_like
        Equatorial coordinates of Healpix pixels, in Cartesian system.
        Shape=(3, NPIX).
    I_sky : array_like
        Intensity distribution on the sky,
        stored as array of Healpix pixels. Shape=(NPIX,).
    bm_cube : array_like, optional
        Pixelized beam maps for each antenna. Shape=(NANT, BM_PIX, BM_PIX).
    beam_list : list of UVBeam, optional
        If specified, evaluate primary beam values directly using UVBeam
        objects instead of using pixelized beam maps (`bm_cube` will be ignored
        if `beam_list` is not None).
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128

    Returns
    -------
    array_like
        Visibilities. Shape=(NTIMES, NANTS, NANTS).
    """
    assert precision in (1, 2)
    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    if bm_cube is None and beam_list is None:
        raise RuntimeError("One of bm_cube/beam_list must be specified")
    if bm_cube is not None and beam_list is not None:
        raise RuntimeError("Cannot specify both bm_cube and beam_list")

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)."
    ncrd, npix = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NPIX)."
    assert I_sky.ndim == 1 and I_sky.shape[0] == npix, "I_sky must have shape (NPIX,)."

    if beam_list is None:
        bm_pix = bm_cube.shape[-1]
        assert bm_cube.shape == (
            nant,
            bm_pix,
            bm_pix,
        ), "bm_cube must have shape (NANTS, BM_PIX, BM_PIX)."
    else:
        assert len(beam_list) == nant, "beam_list must have length nant"

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky.
    Isqrt = np.sqrt(I_sky).astype(real_dtype)
    antpos = antpos.astype(real_dtype)

    ang_freq = 2 * np.pi * freq

    # Empty arrays: beam pattern, visibilities, delays, complex voltages.
    A_s = np.empty((nant, npix), dtype=real_dtype)
    vis = np.empty((ntimes, nant, nant), dtype=complex_dtype)
    tau = np.empty((nant, npix), dtype=real_dtype)
    v = np.empty((nant, npix), dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    # Precompute splines is using pixelized beams
    if beam_list is None:
        bm_pix_x = np.linspace(-1, 1, bm_pix)
        bm_pix_y = np.linspace(-1, 1, bm_pix)

        splines = []
        for i in range(nant):
            # Linear interpolation of primary beam pattern.
            spl = RectBivariateSpline(bm_pix_y, bm_pix_x, bm_cube[i], kx=1, ky=1)
            splines.append(spl)

    # Loop over time samples
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)

        # Primary beam response
        if beam_list is None:
            # Primary beam pattern using pixelized primary beam
            for i in range(nant):
                A_s[i] = splines[i](ty, tx, grid=False)
                # TODO: Try using a log-space beam for accuracy!
        else:
            # Primary beam pattern using direct interpolation of UVBeam object
            az, za = conversions.lm_to_az_za(tx, ty)
            for i in range(nant):
                interp_beam = beam_list[i].interp(az, za, np.atleast_1d(freq))[0]
                A_s[i] = interp_beam[0, 0, 1]  # FIXME: assumes xx pol for now

        A_s = np.where(tz > 0, A_s, 0)

        # Calculate delays, where tau = (b * s) / c
        np.dot(antpos, crd_top, out=tau)
        tau /= c.value

        # Component of complex phase factor for one antenna
        # (actually, b = (antpos1 - antpos2) * crd_top / c; need dot product
        # below to build full phase factor for a given baseline)
        np.exp(1.0j * (ang_freq * tau), out=v)

        # Complex voltages.
        v *= A_s * Isqrt

        # Compute visibilities using product of complex voltages (upper triangle).
        for i in range(len(antpos)):
            np.dot(v[i : i + 1].conj(), v[i:].T, out=vis[t, i : i + 1, i:])

    return vis
