"""CPU-based implementation of the visibility simulator."""

import numpy as np
from astropy.constants import c
from scipy.interpolate import RectBivariateSpline


def vis_cpu(
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    sky_flux: np.ndarray,
    beams: np.ndarray,
    precision: int = 1,
) -> np.ndarray:
    """
    Calculate visibility from an input intensity map and beam model.

    Provided as a standalone function.

    Parameters
    ----------
    antpos
        Antenna position array. Shape=(NANT, 3).
    freq
        Frequency to evaluate the visibilities at [GHz].
    eq2tops
        Set of 3x3 transformation matrices converting equatorial
        coordinates to topocentric at each
        hour angle (and declination) in the dataset.
        Shape=(NTIMES, 3, 3).
    crd_eq
        Equatorial coordinates of Healpix pixels, in Cartesian system.
        Shape=(3, NPIX).
    sky_flux
        Intensity distribution on the sky,
        stored as array of Healpix pixels. Shape=(NPIX,).
    beams
        Beam maps for each antenna. Shape=(NANT, BM_PIX, BM_PIX).
    precision
        One for single-precision, two for double-precision.

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
    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)."
    ncrd, npix = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NPIX)."
    assert (
        sky_flux.ndim == 1 and sky_flux.shape[0] == npix
    ), "I_sky must have shape (NPIX,)."
    bm_pix = beams.shape[-1]
    assert beams.shape == (
        nant,
        bm_pix,
        bm_pix,
    ), "bm_cube must have shape (NANTS, BM_PIX, BM_PIX)."

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky.
    Isqrt = np.sqrt(sky_flux).astype(real_dtype)
    antpos = antpos.astype(real_dtype)

    ang_freq = 2 * np.pi * freq

    # Empty arrays: beam pattern, visibilities, delays, complex voltages.
    A_s = np.empty((nant, npix), dtype=real_dtype)
    vis = np.empty((ntimes, nant, nant), dtype=complex_dtype)
    tau = np.empty((nant, npix), dtype=real_dtype)
    v = np.empty((nant, npix), dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    bm_pix_x = np.linspace(-1, 1, bm_pix)
    bm_pix_y = np.linspace(-1, 1, bm_pix)

    # Loop over time samples.
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)

        for i in range(nant):
            # Linear interpolation of primary beam pattern.
            spline = RectBivariateSpline(bm_pix_y, bm_pix_x, beams[i], kx=1, ky=1)
            A_s[i] = spline(ty, tx, grid=False)

        A_s = np.where(tz > 0, A_s, 0)

        # Calculate delays, where TAU = (b * s) / c.
        np.dot(antpos, crd_top, out=tau)
        tau /= c.value

        np.exp(1.0j * (ang_freq * tau), out=v)

        # Complex voltages.
        v *= A_s * Isqrt

        # Compute visibilities (upper triangle only).
        for i in range(len(antpos)):
            np.dot(v[i : i + 1].conj(), v[i:].T, out=vis[t, i : i + 1, i:])

    return vis
