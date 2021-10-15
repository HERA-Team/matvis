"""Simple example wrapper for basic usage of vis_cpu."""
import numpy as np
from pyuvdata.uvbeam import UVBeam

from . import conversions, vis_cpu


def simulate_vis(
    ants,
    fluxes,
    ra,
    dec,
    freqs,
    lsts,
    beams,
    pixel_beams=False,
    beam_npix=63,
    polarized=False,
    precision=1,
    latitude=-30.7215 * np.pi / 180.0,
    use_feed="x",
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

    Returns
    -------
    vis : array_like
        Complex array of shape (NFREQS, NAXES, NFEED, NTIMES, NANTS, NANTS)
        if ``polarized == True``, or (NFREQS, NTIMES, NANTS, NANTS) otherwise.
    """
    assert len(ants) == len(
        beams
    ), "The `beams` list must have as many entries as the ``ants`` dict."

    assert fluxes.shape == (
        ra.size,
        freqs.size,
    ), "The `fluxes` array must have shape (NSRCS, NFREQS)."

    # Determine precision
    if precision == 1:
        complex_dtype = np.complex64
    else:
        complex_dtype = np.complex128

    # Get polarization information from beams
    if polarized:
        try:
            naxes = beams[0].Naxes_vec
            nfeeds = beams[0].Nfeeds
        except AttributeError:
            # If Naxes_vec and Nfeeds properties aren't set, assume all pol.
            naxes = nfeeds = 2

    # Antenna x,y,z positions
    antpos = np.array([ants[k] for k in ants.keys()])
    nants = antpos.shape[0]

    # Source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Get coordinate transforms as a function of LST
    eq2tops = np.array([conversions.eci_to_enu_matrix(lst, latitude) for lst in lsts])

    # Create beam pixel models (if requested)
    if pixel_beams:
        beam_pix = [
            conversions.uvbeam_to_lm(
                beam, freqs, n_pix_lm=beam_npix, polarized=polarized, use_feed=use_feed
            )
            for beam in beams
        ]
        beam_cube = np.array(beam_pix)
    else:
        beams = [
            conversions.prepare_beam(beam, polarized=polarized, use_feed=use_feed)
            for beam in beams
        ]

    # Run vis_cpu with pixel beams
    if polarized:
        vis = np.zeros(
            (naxes, nfeeds, freqs.size, lsts.size, nants, nants), dtype=complex_dtype
        )
    else:
        vis = np.zeros((freqs.size, lsts.size, nants, nants), dtype=complex_dtype)

    # Loop over frequencies and call vis_cpu for either UVBeam or pixel beams
    for i in range(freqs.size):

        if pixel_beams:

            # Get per-freq. pixel beam
            bm = beam_cube[:, :, :, i, :, :] if polarized else beam_cube[:, i, :, :]

            # Run vis_cpu
            v = vis_cpu(
                antpos,
                freqs[i],
                eq2tops,
                crd_eq,
                fluxes[:, i],
                bm_cube=bm,
                precision=precision,
                polarized=polarized,
            )
            if polarized:
                vis[:, :, i] = v  # v.shape: (nax, nfeed, ntimes, nant, nant)
            else:
                vis[i] = v  # v.shape: (ntimes, nant, nant)
        else:
            v = vis_cpu(
                antpos,
                freqs[i],
                eq2tops,
                crd_eq,
                fluxes[:, i],
                beam_list=beams,
                precision=precision,
                polarized=polarized,
            )
            if polarized:
                vis[:, :, i] = v  # v.shape: (nax, nfeed, ntimes, nant, nant)
            else:
                vis[i] = v  # v.shape: (ntimes, nant, nant)

    return vis
