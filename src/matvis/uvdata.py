"""Convert :func:`matvis.simulate_vis` output into a :class:`pyuvdata.UVData` object.

:func:`matvis.simulate_vis` returns a bare array, whose layout and conventions are
described in its Returns section. :func:`matvis_to_uvdata` writes those conventions
down once, as code: which antenna pair, time, frequency and polarization each element
belongs to. Comparisons against other simulators (e.g. pyuvsim) can then go through
:meth:`pyuvdata.UVData.get_data` on both sides instead of indexing the matvis array by
hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pyuvdata import Telescope, UVBeam, UVData
from pyuvdata import utils as uvutils
from pyuvdata.analytic_beam import AnalyticBeam
from pyuvdata.beam_interface import BeamInterface

from . import __version__
from .core.beams import prepare_beam_unpolarized


def matvis_to_uvdata(
    vis: np.ndarray,
    *,
    ants: dict[int, np.ndarray],
    freqs: np.ndarray,
    times: Time,
    telescope_loc: EarthLocation,
    beams: Sequence[UVBeam | AnalyticBeam | BeamInterface],
    polarized: bool = False,
    antpairs: np.ndarray | None = None,
    channel_width: float | np.ndarray | None = None,
    integration_time: float | np.ndarray | None = None,
    vis_units: Literal["Jy", "K str", "uncalib"] = "Jy",
    telescope_name: str = "matvis",
) -> UVData:
    """Put the output of :func:`matvis.simulate_vis` into a ``UVData`` object.

    The keyword arguments are the ones given to :func:`~matvis.simulate_vis`, so
    the array is interpreted exactly as that function documents.

    Parameters
    ----------
    vis : np.ndarray
        Output of :func:`~matvis.simulate_vis`, of shape
        ``(Nfreqs, Ntimes, Npairs, Nfeeds, Nfeeds)`` if ``polarized`` is True and
        ``(Nfreqs, Ntimes, Npairs)`` otherwise.
    ants : dict
        Antenna positions given to :func:`~matvis.simulate_vis`: keys are integer
        antenna numbers, values are East-North-Up positions in meters relative to
        ``telescope_loc``. The key order defines the antenna indices.
    freqs : np.ndarray
        Frequencies in Hz.
    times : astropy.time.Time
        Observation times.
    telescope_loc : astropy.coordinates.EarthLocation
        Array center.
    beams : list of ``UVBeam``, ``AnalyticBeam`` or ``BeamInterface``
        Beams given to :func:`~matvis.simulate_vis`. The first beam sets the
        polarization labels: its ``feed_array`` order if ``polarized``, otherwise
        the single polarization matvis simulates for it.
    polarized : bool, optional
        Whether ``vis`` came from a polarized simulation.
    antpairs : np.ndarray, optional
        Antenna-index pairs given to :func:`~matvis.simulate_vis`, shape
        ``(Npairs, 2)``. If None, ``vis`` holds all ``Nants**2`` ordered pairs.
    channel_width, integration_time : float or np.ndarray, optional
        Metadata only: matvis evaluates each frequency and time exactly. If None,
        pyuvdata infers them from the spacing of ``freqs`` and ``times`` (and
        warns if there is only one of either).
    vis_units : str, optional
        Units of ``vis``, i.e. of the ``fluxes`` given to
        :func:`~matvis.simulate_vis`. Default is ``"Jy"``.
    telescope_name : str, optional
        Name to give the telescope (and instrument).

    Returns
    -------
    UVData
        One baseline per entry of ``antpairs`` (antenna numbers taken from
        ``ants``), with time-major baseline-time ordering. Polarizations are
        ``feed_array[p] + feed_array[q]`` for ``vis[..., p, q]``, and uvws are
        ``x_j - x_i`` (the matvis and pyuvsim convention). The data are a copy
        of ``vis``.

    Raises
    ------
    ValueError
        If the shape of ``vis`` does not match the other inputs.

    Notes
    -----
    A UVData object can hold both orders of a pair, and does so here if
    ``antpairs`` includes both (as the default, all ``Nants**2`` pairs, does).
    :meth:`~pyuvdata.UVData.get_data` on such a pair then returns the rows of
    both: those of the requested order, followed by the conjugate of the other
    order with its polarization swapped. Pass ``antpairs`` with a single order
    (e.g. ``i <= j``) for one row per time.
    """
    vis = np.asarray(vis)
    names = list(ants)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    jds = np.atleast_1d(Time(times).jd)
    if antpairs is None:
        antpairs = np.array(
            [(i, j) for i in range(len(names)) for j in range(len(names))]
        )
    antpairs = np.asarray(antpairs)

    beam = BeamInterface(beams[0])
    if polarized:
        feeds = [str(feed) for feed in beam.feed_array]
        polarizations = uvutils.polstr2num(
            [p + q for p in feeds for q in feeds],
            x_orientation=beam.beam.get_x_orientation_from_feeds(),
        )
        feed_shape = (len(feeds), len(feeds))
    else:
        polarizations = list(prepare_beam_unpolarized(beam).polarization_array[:1])
        feed_shape = ()

    expected = (freqs.size, jds.size, len(antpairs), *feed_shape)
    if vis.shape != expected:
        raise ValueError(
            f"vis has shape {vis.shape}, but the other inputs imply {expected}."
        )

    # (Nfreqs, Ntimes, Npairs, Npols) -> (Ntimes * Npairs, Nfreqs, Npols), with
    # pairs varying fastest to match do_blt_outer=True below. The reshape alone
    # would be a strided view of vis, so take a contiguous copy.
    data = vis.reshape(*expected[:3], len(polarizations)).transpose(1, 2, 0, 3)
    data = np.ascontiguousarray(
        data.reshape(jds.size * len(antpairs), freqs.size, len(polarizations))
    )

    # Telescope positions are ECEF, relative to the telescope location.
    enu = np.array([ants[name] for name in names], dtype=float)
    center = np.array([c.to_value("m") for c in telescope_loc.geocentric])
    telescope = Telescope.new(
        location=telescope_loc,
        name=telescope_name,
        instrument=telescope_name,
        antenna_positions=uvutils.ECEF_from_ENU(enu, center_loc=telescope_loc) - center,
        antenna_names=[str(name) for name in names],
        antenna_numbers=np.asarray(names),
    )

    return UVData.new(
        freq_array=freqs,
        polarization_array=polarizations,
        times=jds,
        telescope=telescope,
        antpairs=[(names[i], names[j]) for i, j in antpairs],
        do_blt_outer=True,
        time_axis_faster_than_bls=False,
        data_array=data,
        channel_width=channel_width,
        integration_time=integration_time,
        vis_units=vis_units,
        history=f"Converted from matvis {__version__} output by matvis_to_uvdata.",
    )
