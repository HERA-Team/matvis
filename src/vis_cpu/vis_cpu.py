"""CPU-based implementation of the visibility simulator."""

import numpy as np
from astropy.constants import c
from pyuvdata import UVBeam
from scipy.interpolate import RectBivariateSpline
from typing import Optional, Sequence

from . import conversions

# This enables us to put in profile decorators that will be no-ops if no profiling
# library is being used.
try:
    profile
except NameError:

    def profile(fnc):
        """No-op profiling decorator."""
        return fnc


def construct_pixel_beam_spline(bm_cube):
    """Construct bivariate spline for pixelated beams for all antennas.

    Uses the ``scipy.interpolate.RectBivariateSpline`` function.

    Parameters
    ----------
    bm_cube : array_like
        Pixelized beam maps for each antenna (must be real-valued).
        Shape: (NANT, BEAM_PIX, BEAM_PIX) if unpolarized, or
        (NANT, NAXES, NFEEDS, BEAM_PIX, BEAM_PIX) if polarized.

    Returns
    -------
    splines : list of fn
        List of interpolation functions, one for each antenna, in order.
    """
    # Raise error if complex
    if np.any(np.iscomplex(bm_cube)):
        raise TypeError(
            "bm_cube cannot be complex. Interpolate real and "
            "imaginary components separately."
        )

    # Polarized beam
    nax, nfeed, nbeam, bm_pix, _ = bm_cube.shape

    # x and y coordinates of beam
    lm = conversions.bm_pix_to_lm(bm_pix)

    # Construct splines for each polarization (pol. vector axis + feed) and
    # antenna. The `splines` list has shape (Naxes, Nfeeds, Nants).
    splines = []
    for p1 in range(nax):
        spl_axes = []
        for p2 in range(nfeed):
            spl_feeds = []

            # Loop over beams/antennas
            for i in range(nbeam):
                # Linear interpolation of primary beam pattern.
                spl = RectBivariateSpline(lm, lm, bm_cube[p1, p2, i], kx=1, ky=1)
                spl_feeds.append(spl)
            spl_axes.append(spl_feeds)
        splines.append(spl_axes)
    return splines


@profile
def vis_cpu(
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    bm_cube: Optional[np.ndarray] = None,
    beam_list: Optional[Sequence[UVBeam]] = None,
    precision: int = 1,
    polarized: bool = False,
    beam_idx: Optional[np.ndarray] = None,
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
    bm_cube : array_like, optional
        Pixelized beam maps for each antenna. If ``polarized=False``,
        shape=``(NBEAMS, BM_PIX, BM_PIX)``, otherwise
        shape=``(NAX, NFEED, NBEAMS, BM_PIX, BM_PIX)``. Only one of ``bm_cube`` and
        ``beam_list`` should be provided. If NBEAMS != NANT, then `beam_idx` must be
        provided also. Note that the projected coordinates corresponding to the bm_cube
        MUST be equivalent to those returned by :func:`~conversions.bm_pix_to_lm`.
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
        shape (NAXES, NFEED, NTIMES, NANTS, NANTS), otherwise it will have
        shape (NTIMES, NANTS, NANTS).
    """
    assert precision in {1, 2}
    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    # Specify number of polarizations (axes/feeds)
    if polarized:
        nax = nfeed = 2
    else:
        nax = nfeed = 1

    if bm_cube is None and beam_list is None:
        raise RuntimeError("One of bm_cube/beam_list must be specified")
    if bm_cube is not None and beam_list is not None:
        raise RuntimeError("Cannot specify both bm_cube and beam_list")

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)."
    ncrd, nsrcs = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NSRCS)."
    assert (
        I_sky.ndim == 1 and I_sky.shape[0] == nsrcs
    ), "I_sky must have shape (NSRCS,)."

    # Get the number of unique beams
    if bm_cube is not None:
        nbeam = bm_cube.shape[2 if polarized else 0]
    else:
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

    if beam_list is None:
        bm_pix = bm_cube.shape[-1]
        complex_bm_cube = np.any(np.iscomplex(bm_cube))
        if polarized:
            assert bm_cube.shape == (nax, nfeed, nbeam, bm_pix, bm_pix), (
                "bm_cube must have shape (NAXES, NFEEDS, NBEAMS, BM_PIX, BM_PIX) if "
                f"polarized=True. Shape wanted: {(nax, nfeed, nant, bm_pix, bm_pix)}; "
                f"shape given: {bm_cube.shape}"
            )
        elif bm_cube.shape != (1, 1, nbeam, bm_pix, bm_pix):
            assert bm_cube.shape == (nbeam, bm_pix, bm_pix), (
                "bm_cube must have shape (NBEAMS, BM_PIX, BM_PIX) "
                "or (1, 1, nbeam, bm_pix, bm_pix) if polarized=False. "
                f"Shape wanted: {(nbeam, bm_pix, bm_pix)}; "
                f"shape given: {bm_cube.shape}"
            )
            bm_cube = bm_cube[np.newaxis, np.newaxis]
    elif polarized and any(b.beam_type != "efield" for b in beam_list):
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

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky. Factor of 0.5 accounts for splitting Stokes I between
    # polarization channels
    Isqrt = np.sqrt(0.5 * I_sky).astype(real_dtype)
    antpos = antpos.astype(real_dtype)

    ang_freq = 2.0 * np.pi * freq

    # Zero arrays: beam pattern, visibilities, delays, complex voltages
    vis = np.zeros((nfeed, nfeed, ntimes, nant, nant), dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    # Precompute splines using pixelized beams
    if beam_list is None:
        splines_re = construct_pixel_beam_spline(bm_cube.real)
        if complex_bm_cube:
            splines_im = construct_pixel_beam_spline(bm_cube.imag)

    im = 0

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

        A_s = np.zeros((nax, nfeed, nbeam, nsrcs_up), dtype=complex_dtype)
        tau = np.zeros((nant, nsrcs_up), dtype=real_dtype)
        v = np.zeros((nant, nsrcs_up), dtype=complex_dtype)

        # Primary beam response
        if beam_list is None:
            # Primary beam pattern using pixelized primary beam
            for i in range(nbeam):
                # Extract requested polarizations
                for p1 in range(nax):
                    for p2 in range(nfeed):
                        # The beam pixel grid has been reshaped in the order
                        # ty,tx, which implies m,l order

                        re = splines_re[p1][p2][i](ty, tx, grid=False)

                        if complex_bm_cube:
                            im = 1.0j * splines_im[p1][p2][i](ty, tx, grid=False)

                        A_s[p1, p2, i] = re + im
        else:

            # Primary beam pattern using direct interpolation of UVBeam object
            az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")
            for i, bm in enumerate(beam_list):
                kw = (
                    {"reuse_spline": True, "check_azza_domain": False}
                    if isinstance(bm, UVBeam)
                    else {}
                )

                interp_beam = bm.interp(
                    az_array=az, za_array=za, freq_array=np.atleast_1d(freq), **kw
                )[0]

                if polarized:
                    interp_beam = interp_beam[:, 0, :, 0, :]
                else:
                    # Here we have already asserted that the beam is a power beam and
                    # has only one polarization, so we just evaluate that one.
                    interp_beam = np.sqrt(interp_beam[0, 0, 0, 0, :])

                A_s[:, :, i] = interp_beam

        # Check for invalid beam values
        if np.any(np.isinf(A_s)) or np.any(np.isnan(A_s)):
            raise ValueError("Beam interpolation resulted in an invalid value")

        # Calculate delays, where tau = (b * s) / c
        np.dot(antpos, crd_top[:, above_horizon], out=tau)
        tau /= c.value

        # Component of complex phase factor for one antenna
        # (actually, b = (antpos1 - antpos2) * crd_top / c; need dot product
        # below to build full phase factor for a given baseline)
        np.exp(1.0j * (ang_freq * tau), out=v)

        # Complex voltages.
        v *= Isqrt[above_horizon]

        # Compute visibilities using product of complex voltages (upper triangle).
        # Input arrays have shape (Nax, Nfeed, [Nants], Nsrcs
        v = A_s[:, :, beam_idx] * v[np.newaxis, np.newaxis, :]

        for i in range(len(antpos)):
            # We want to take an outer product over feeds/antennas, contract over
            # E-field components, and integrate over the sky.
            vis[:, :, t, i : i + 1, i:] = np.einsum(
                "jiln,jkmn->iklm", v[:, :, i : i + 1].conj(), v[:, :, i:], optimize=True
            )

    # Return visibilities with or without multiple polarization channels
    return vis if polarized else vis[0, 0]
