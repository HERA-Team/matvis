"""Simple example wrapper for basic usage of matvis."""

from __future__ import annotations

import logging
import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AnalyticBeam
from pyuvdata.beam_interface import BeamInterface
from typing import Literal

from . import HAVE_GPU, cpu
from .core.beams import prepare_beam_unpolarized

if HAVE_GPU:
    from . import gpu

logger = logging.getLogger(__name__)


def simulate_vis(
    ants: dict[int, np.ndarray],
    fluxes: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    times: Time,
    beams: list[AnalyticBeam | UVBeam | BeamInterface],
    telescope_loc: EarthLocation,
    polarized: bool = False,
    precision: Literal[1, 2] = 1,
    use_feed: Literal["x", "y"] = "x",
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
    times : astropy.Time instance
        Times of the observation (can be an array of times).
    beams : list of ``UVBeam``, ``AnalyticBeam`` or ``BeamInterface`` objects
        Beam objects to use for each antenna.
    telescope_loc
        An EarthLocation object representing the center of the array.
    polarized : bool, optional
        If True, use polarized beams and calculate all available linearly-
        polarized visibilities, e.g. V_nn, V_ne, V_en, V_ee.
        Default: False (only uses the 'ee' polarization).
    precision : int, optional
        Which precision setting to use for :func:`~matvis`. If set to ``1``,
        uses the (``np.float32``, ``np.complex64``) dtypes. If set to ``2``,
        uses the (``np.float64``, ``np.complex128``) dtypes.
    use_feed
        Either 'x' or 'y'. Only used if polarized is False.
    use_gpu : bool, optional
        Whether to use the GPU for simulation.
    beam_spline_opts : dict, optional
        Options to be passed to :meth:`pyuvdata.uvbeam.UVBeam.interp` as `spline_opts`.
    beam_idx
        An array of integers, of the same length as ``ants``. Each entry is for an
        antenna of the same index, and its value should be the index of the beam in
        the beam list that corresponds to the antenna.
    antpairs
        A list of antpairs (in the form of 2-tuples of integers) to actually
        calculate visibility for. If None, all feed-pairs are calculated.
    source_buffer : float, optional
        The fraction of the total number of sources to use when allocating memory
        for the sources above horizon. For large numbers of sources, a fraction of
        ~0.55 should be sufficient.

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
