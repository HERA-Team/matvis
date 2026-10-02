"""Tests of matvis.uvdata.matvis_to_uvdata."""

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pyuvdata import utils as uvutils
from pyuvdata.analytic_beam import GaussianBeam

from matvis import matvis_to_uvdata, simulate_vis

# Non-contiguous, unsorted antenna names, so index/name mix-ups show up.
ANTS = {10: (0.0, 0.0, 0.0), 3: (14.6, 0.0, 0.0), 7: (-7.3, 25.3, 0.5)}
NAMES = list(ANTS)
FREQS = np.array([100e6, 120e6, 140e6])
TIMES = Time(2459845.25 + np.arange(4) * 10.7 / 86400, format="jd")
LOC = EarthLocation.from_geodetic(lat=-30.7215, lon=21.4283, height=1051.7)
ALL_PAIRS = [(i, j) for i in range(len(ANTS)) for j in range(len(ANTS))]


def _encoded_vis(
    npairs: int, polarized: bool, nfreqs: int = FREQS.size, ntimes: int = TIMES.size
) -> np.ndarray:
    """Visibilities whose values are unique, so any misplacement is detected."""
    shape = (nfreqs, ntimes, npairs) + ((2, 2) if polarized else ())
    n = np.prod(shape)
    return (np.arange(n) + 1j * np.arange(n, 2 * n)).reshape(shape)


def _rows(uvd, i: int, j: int, pol: str) -> np.ndarray:
    """Data for exactly the ordered pair (i, j), shape (Ntimes, Nfreqs)."""
    inds = uvd.antpair2ind(NAMES[i], NAMES[j], ordered=True)
    pind = list(uvd.polarization_array).index(uvutils.polstr2num(pol))
    return uvd.data_array[inds, :, pind]


@pytest.mark.parametrize(
    "antpairs",
    [None, np.array([(2, 0), (1, 1), (0, 1)])],
    ids=["all_pairs", "subset_with_reversed"],
)
def test_polarized_layout(antpairs):
    """Every element of the matvis array lands at its baseline, time, freq and pol."""
    pairs = ALL_PAIRS if antpairs is None else [tuple(p) for p in antpairs]
    vis = _encoded_vis(len(pairs), polarized=True)
    uvd = matvis_to_uvdata(
        vis,
        ants=ANTS,
        freqs=FREQS,
        times=TIMES,
        telescope_loc=LOC,
        beams=[GaussianBeam(diameter=14.0)],
        polarized=True,
        antpairs=antpairs,
    )

    assert uvd.get_pols() == ["xx", "xy", "yx", "yy"]
    assert not np.shares_memory(uvd.data_array, vis)
    assert uvd.Nbls == len(pairs)
    assert uvd.Ntimes == TIMES.size
    assert uvd.vis_units == "Jy"
    np.testing.assert_array_equal(uvd.freq_array, FREQS)
    np.testing.assert_array_equal(uvd.telescope.antenna_numbers, NAMES)

    feeds = "xy"
    for k, (i, j) in enumerate(pairs):
        inds = uvd.antpair2ind(NAMES[i], NAMES[j], ordered=True)
        np.testing.assert_allclose(uvd.time_array[inds], TIMES.jd, rtol=0, atol=1e-9)
        # pyuvdata's uvw for (ant1, ant2) is x_ant2 - x_ant1, the matvis b_ij.
        np.testing.assert_allclose(
            uvd.uvw_array[inds],
            np.broadcast_to(
                np.subtract(ANTS[NAMES[j]], ANTS[NAMES[i]]), (TIMES.size, 3)
            ),
            rtol=0,
            atol=1e-6,
        )
        for p in range(2):
            for q in range(2):
                np.testing.assert_array_equal(
                    _rows(uvd, i, j, feeds[p] + feeds[q]), vis[:, :, k, p, q].T
                )


def test_unpolarized_layout_and_pol_label(uvbeam):
    """Unpolarized output is labelled with the feed matvis actually simulates."""
    vis = _encoded_vis(len(ALL_PAIRS), polarized=False)
    kw = {"ants": ANTS, "freqs": FREQS, "times": TIMES, "telescope_loc": LOC}

    # An efield beam is reduced to its x-feed power response.
    uvd = matvis_to_uvdata(vis, beams=[uvbeam], **kw)
    assert uvd.get_pols() == ["xx"]
    for k, (i, j) in enumerate(ALL_PAIRS):
        np.testing.assert_array_equal(_rows(uvd, i, j, "xx"), vis[:, :, k].T)

    # A single-polarization power beam is used as it is.
    yy = uvbeam.efield_to_power(calc_cross_pols=False, inplace=False)
    yy.select(polarizations=["yy"])
    assert matvis_to_uvdata(vis, beams=[yy], **kw).get_pols() == ["yy"]


def test_get_data_returns_both_orientations():
    """A pair stored in both orders comes back twice from get_data, as documented."""
    vis = _encoded_vis(len(ALL_PAIRS), polarized=True)
    uvd = matvis_to_uvdata(
        vis,
        ants=ANTS,
        freqs=FREQS,
        times=TIMES,
        telescope_loc=LOC,
        beams=[GaussianBeam(diameter=14.0)],
        polarized=True,
    )
    i, j = 0, 2
    k_ij, k_ji = ALL_PAIRS.index((i, j)), ALL_PAIRS.index((j, i))
    expected = np.concatenate([vis[:, :, k_ij, 0, 1].T, vis[:, :, k_ji, 1, 0].T.conj()])
    np.testing.assert_array_equal(uvd.get_data((NAMES[i], NAMES[j], "xy")), expected)


def test_matches_simulate_vis_output():
    """A real simulate_vis result converts with the same inputs, including precision."""
    kw = {
        "ants": ANTS,
        "freqs": FREQS,
        "times": TIMES,
        "telescope_loc": LOC,
        "beams": [GaussianBeam(diameter=14.0)],
        "polarized": True,
    }
    antpairs = np.array([(0, 1), (1, 0), (2, 2)])
    vis = simulate_vis(
        fluxes=np.ones((1, FREQS.size)),
        ra=np.array([0.3]),
        dec=np.array([np.deg2rad(-25.0)]),
        precision=1,
        antpairs=antpairs,
        **kw,
    )
    uvd = matvis_to_uvdata(vis, antpairs=antpairs, **kw)
    assert uvd.data_array.dtype == np.complex64
    np.testing.assert_array_equal(_rows(uvd, 1, 0, "yx"), vis[:, :, 1, 1, 0].T)


def test_metadata_overrides():
    """channel_width, integration_time and vis_units are passed through."""
    uvd = matvis_to_uvdata(
        _encoded_vis(len(ALL_PAIRS), polarized=False, nfreqs=1, ntimes=1),
        ants=ANTS,
        freqs=FREQS[:1],
        times=TIMES[:1],
        telescope_loc=LOC,
        beams=[GaussianBeam(diameter=14.0)],
        channel_width=97.0e3,
        integration_time=8.0,
        vis_units="K str",
    )
    np.testing.assert_array_equal(uvd.channel_width, [97.0e3])
    np.testing.assert_array_equal(uvd.integration_time, 8.0)
    assert uvd.vis_units == "K str"


def test_wrong_shape_raises():
    """The array has to match the inputs it claims to come from."""
    with pytest.raises(ValueError, match=r"imply \(3, 4, 9, 2, 2\)"):
        matvis_to_uvdata(
            _encoded_vis(len(ALL_PAIRS), polarized=False),
            ants=ANTS,
            freqs=FREQS,
            times=TIMES,
            telescope_loc=LOC,
            beams=[GaussianBeam(diameter=14.0)],
            polarized=True,
        )
