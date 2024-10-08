"""Simple example wrapper for basic usage of matvis."""

from __future__ import annotations

import logging
import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord

from . import HAVE_GPU, cpu
from .core.beams import prepare_beam_unpolarized

if HAVE_GPU:
    from . import gpu

logger = logging.getLogger(__name__)


def simulate_vis(
    ants,
    fluxes,
    ra,
    dec,
    freqs,
    times,
    beams,
    telescope_loc: EarthLocation,
    polarized=False,
    precision=1,
    use_feed="x",
    use_gpu: bool = False,
    beam_spline_opts: dict | None = None,
    beam_idx: np.ndarray | None = None,
    antpairs: np.ndarray | list[tuple[int, int]] | None = None,
    source_buffer: float = 1.0,
    **backend_kwargs,
):
    """
    Run a basic simulation using ``matvis``.

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
        and Dec from [-pi/2, +pi/2].
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
        Which precision setting to use for :func:`~matvis`. If set to ``1``,
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
        Complex array of shape (NFREQS, NTIMES, NBLS, NFEED, NFEED)
        if ``polarized == True``, or (NFREQS, NTIMES, NBLS) otherwise.
    """
    if use_gpu:
        if not HAVE_GPU:
            raise ImportError("You cannot use GPU without installing GPU-dependencies!")

        import cupy as cp

        device = cp.cuda.Device()
        attrs = device.attributes
        attrs = {str(k): v for k, v in attrs.items()}
        string = "\n\t".join(f"{k}: {v}" for k, v in attrs.items())
        logger.debug(
            f"""
            Your GPU has the following attributes:
            \t{string}
            """
        )

    fnc = gpu.simulate if use_gpu else cpu.simulate

    assert fluxes.shape == (
        ra.size,
        freqs.size,
    ), "The `fluxes` array must have shape (NSRCS, NFREQS)."

    # Determine precision
    complex_dtype = np.complex64 if precision == 1 else np.complex128

    # Get polarization information from beams
    if polarized:
        nfeeds = getattr(beams[0], "Nfeeds", 2)
    else:
        beams = [prepare_beam_unpolarized(beam) for beam in beams]

    # Antenna x,y,z positions
    antpos = np.array([ants[k] for k in ants.keys()])
    nants = antpos.shape[0]

    skycoords = SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame="icrs")

    npairs = len(antpairs) if antpairs is not None else nants * nants
    if polarized:
        vis = np.zeros(
            (freqs.size, times.size, npairs, nfeeds, nfeeds), dtype=complex_dtype
        )
    else:
        vis = np.zeros((freqs.size, times.size, npairs), dtype=complex_dtype)

    # Loop over frequencies and call matvis_cpu/gpu
    for i, freq in enumerate(freqs):
        vis[i] = fnc(
            antpos=antpos,
            freq=freq,
            times=times,
            skycoords=skycoords,
            telescope_loc=telescope_loc,
            I_sky=fluxes[:, i],
            beam_list=beams,
            precision=precision,
            polarized=polarized,
            beam_spline_opts=beam_spline_opts,
            beam_idx=beam_idx,
            antpairs=antpairs,
            source_buffer=source_buffer,
            **backend_kwargs,
        )
    return vis
