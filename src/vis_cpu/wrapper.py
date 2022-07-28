"""Simple example wrapper for basic usage of vis_cpu."""
from __future__ import annotations

import logging
import numpy as np

from . import HAVE_GPU, conversions, vis_cpu

if HAVE_GPU:
    from . import vis_gpu

logger = logging.getLogger(__name__)


def simulate_vis(
    ants,
    fluxes,
    ra,
    dec,
    freqs,
    lsts,
    beams,
    polarized=False,
    precision=1,
    latitude=-30.7215 * np.pi / 180.0,
    use_feed="x",
    use_gpu: bool = False,
    beam_spline_opts: dict | None = None,
    beam_idx: np.ndarray | None = None,
    **backend_kwargs,
):
    """
    Run a basic simulation using ``vis_cpu``.

    This wrapper handles the necessary coordinate conversions etc.

    Parameters
    ----------
    ants : dict
        Dictionary of antenna positions. The keys are the antenna names
        (integers) and the values are the Cartesian x,y,z positions of the
        antennas (in meters) relative to the array center.
    fluxes : array_like
        2D array with the flux of each source as a function of frequency, of
        shape (NSRCS, NFREQS).
    ra, dec : array_like
        Arrays of source RA and Dec positions in radians. RA goes from [0, 2 pi]
        and Dec from [-pi, +pi].
    freqs : array_like
        Frequency channels for the simulation, in Hz.
    lsts : array_like
        Local sidereal times for the simulation, in radians. Range is [0, 2 pi].
    beams : list of ``UVBeam`` objects
        Beam objects to use for each antenna.
    pixel_beams : bool, optional
        If True, interpolate the beams onto a pixel grid. Otherwise, use the
        ``UVBeam`` interpolation method directly.
    beam_npix : int, optional
        If ``pixel_beam == True``, sets the pixel grid resolution along each
        dimension (corresponds to the ``n_pix_lm`` parameter of the
        `conversions.uvbeam_to_lm` function).
    polarized : bool, optional
        If True, use polarized beams and calculate all available linearly-
        polarized visibilities, e.g. V_nn, V_ne, V_en, V_ee.
        Default: False (only uses the 'ee' polarization).
    precision : int, optional
        Which precision setting to use for :func:`~vis_cpu`. If set to ``1``,
        uses the (``np.float32``, ``np.complex64``) dtypes. If set to ``2``,
        uses the (``np.float64``, ``np.complex128``) dtypes.
    latitude : float, optional
        The latitude of the center of the array, in radians. The default is the
        HERA latitude = -30.7215 * pi / 180.
    beam_spline_opts : dict, optional
        Options to be passed to :meth:`pyuvdata.uvbeam.UVBeam.interp` as `spline_opts`.

    Returns
    -------
    vis : array_like
        Complex array of shape (NFREQS, NTIMES, NFEED, NFEED, NANTS, NANTS)
        if ``polarized == True``, or (NFREQS, NTIMES, NANTS, NANTS) otherwise.
    """
    if use_gpu:
        if not HAVE_GPU:
            raise ImportError("You cannot use GPU without installing GPU-dependencies!")

        from pycuda import driver

        device = driver.Device(0)
        attrs = device.get_attributes()
        attrs = {str(k): v for k, v in attrs.items()}
        string = "\n\t".join(f"{k}: {v}" for k, v in attrs.items())
        logger.debug(
            f"""
            Your GPU has the following attributes:
            \t{string}
            """
        )

    fnc = vis_gpu if use_gpu else vis_cpu

    assert fluxes.shape == (
        ra.size,
        freqs.size,
    ), "The `fluxes` array must have shape (NSRCS, NFREQS)."

    # Determine precision
    complex_dtype = np.complex64 if precision == 1 else np.complex128

    # Get polarization information from beams
    if polarized:
        nfeeds = getattr(beams[0], "Nfeeds", 2)

    # Antenna x,y,z positions
    antpos = np.array([ants[k] for k in ants.keys()])
    nants = antpos.shape[0]

    # Source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Get coordinate transforms as a function of LST
    eq2tops = np.array([conversions.eci_to_enu_matrix(lst, latitude) for lst in lsts])

    # Create beam pixel models (if requested)
    beams = [
        conversions.prepare_beam(beam, polarized=polarized, use_feed=use_feed)
        for beam in beams
    ]

    if polarized:
        vis = np.zeros(
            (freqs.size, lsts.size, nfeeds, nfeeds, nants, nants), dtype=complex_dtype
        )
    else:
        vis = np.zeros((freqs.size, lsts.size, nants, nants), dtype=complex_dtype)

    # Loop over frequencies and call vis_cpu/gpu
    for i, freq in enumerate(freqs):
        vis[i] = fnc(
            antpos=antpos,
            freq=freq,
            eq2tops=eq2tops,
            crd_eq=crd_eq,
            I_sky=fluxes[:, i],
            beam_list=beams,
            precision=precision,
            polarized=polarized,
            beam_spline_opts=beam_spline_opts,
            beam_idx=beam_idx,
            **backend_kwargs,
        )
    return vis
