"""Test coordinate rotation modules."""

import numpy as np
import pytest
from astropy import units as un
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pyuvdata.telescopes import Telescope

from matvis import HAVE_GPU
from matvis._test_utils import get_standard_sim_params
from matvis.core.coords import CoordinateRotation
from matvis.cpu.coords import CoordinateRotationAstropy, CoordinateRotationERFA

if HAVE_GPU:
    import cupy as cp
    from cupy import get_array_module
else:
    cp = np

    def get_array_module(x):
        """Dummy function to return np."""
        return np


def get_angles(x, y):
    """Compute angles between arrays of 3-vectors."""
    xp = get_array_module(x)

    dot = xp.sum(x * y, axis=0)
    norms = xp.sqrt(xp.linalg.norm(x, axis=0) * xp.linalg.norm(y, axis=0))
    ratio = dot / norms
    ratio[ratio > 1.0] = 1.0
    ratio[ratio < -1.0] = -1.0
    return xp.arccos(ratio)


def test_complex_flux():
    """Test that using a complex flux works appropriately."""
    rng = np.random.default_rng(1234)
    n = 23
    location = Telescope.from_known_telescopes("hera").location
    skycoords = SkyCoord(
        ra=rng.uniform(0, 2 * np.pi, size=n) * un.rad,
        dec=rng.uniform(-np.pi / 2, np.pi / 2, size=n) * un.rad,
        frame="icrs",
    )

    coords = CoordinateRotationAstropy(
        flux=rng.normal(100, 2, size=n) + 1j * rng.normal(100, 2, size=n),
        times=Time(np.array([2459863.0]), format="jd", scale="utc"),
        telescope_loc=location,
        skycoords=skycoords,
        gpu=False,
        precision=2,
    )
    assert coords.sky_model_dtype == coords.ctype == np.complex128


_COORD_METHODS = [
    pytest.param(m, marks=pytest.mark.gpu) if m.requires_gpu else m
    for m in CoordinateRotation._methods.values()
]


def get_random_coordrot(n, method, gpu, seed, precision=2, setup: bool = True, **kw):
    """Get a random coordinate rotation object."""
    rng = np.random.default_rng(seed)
    location = Telescope.from_known_telescopes("hera").location
    skycoords = SkyCoord(
        ra=rng.uniform(0, 2 * np.pi, size=n) * un.rad,
        dec=rng.uniform(-np.pi / 2, np.pi / 2, size=n) * un.rad,
        frame="icrs",
    )
    coords = method(
        flux=rng.normal(100, 2, size=n),
        times=Time(np.array([2459863.0]), format="jd", scale="utc"),
        telescope_loc=location,
        skycoords=skycoords,
        gpu=gpu,
        precision=precision,
        **kw,
    )
    if setup:
        coords.setup()
    return coords


@pytest.mark.parametrize("method", _COORD_METHODS)
@pytest.mark.parametrize("gpu", [False, True] if HAVE_GPU else [False])
def test_repeat_stays_same(method, gpu):
    """This test just checks that repeating the .rotate() method multiple times works."""
    if not gpu and method.requires_gpu:
        pytest.skip()

    coords = get_random_coordrot(15, method, gpu, seed=35)

    coords.rotate(0)
    xx = coords.all_coords_topo.copy()
    xp = cp if gpu else np

    coords.rotate(0)
    assert xp.allclose(xx, coords.all_coords_topo)


@pytest.mark.parametrize("method", _COORD_METHODS)
@pytest.mark.parametrize("gpu", [False, True] if HAVE_GPU else [False])
@pytest.mark.parametrize("precision", [1, 2])
def test_accuracy_against_astropy(method, gpu, precision):
    """Test other methods against the benchmark Astropy method."""
    if not gpu and method.requires_gpu:
        pytest.skip()

    astr = get_random_coordrot(
        1000, CoordinateRotationAstropy, gpu, seed=42, precision=precision
    )
    coords = get_random_coordrot(1000, method, gpu, seed=42, precision=precision)

    coords.rotate(0)
    astr.rotate(0)

    # get anglular distance between each point in arcsec
    angles = (
        get_angles(coords.all_coords_topo, astr.all_coords_topo) * 180 / np.pi * 3600
    )
    assert len(angles) == 1000
    if gpu:
        angles = angles.get()

    if precision == 2:
        np.testing.assert_allclose(angles, 0, atol=0.01)  # 10 mas
    else:
        np.testing.assert_allclose(angles, 0, atol=150)  # 50 mas


def test_coord_rot_erfa_set_bcrs():
    """Test that setting bcrs before setup works as expected."""
    normal = get_random_coordrot(
        1000, CoordinateRotationERFA, gpu=False, seed=1, precision=1
    )
    bcrs = get_random_coordrot(
        1000, CoordinateRotationERFA, gpu=False, seed=1, precision=1, setup=False
    )
    bcrs._set_bcrs(0)
    bcrs.setup()

    normal.rotate(0)
    bcrs.rotate(0)

    np.testing.assert_allclose(normal.all_coords_topo, bcrs.all_coords_topo)


def test_larger_chunksize():
    """Test that using different chunk sizes results in the same output."""
    small = get_random_coordrot(
        10000, CoordinateRotationERFA, gpu=False, seed=1, precision=1, chunk_size=100
    )
    large = get_random_coordrot(
        10000, CoordinateRotationERFA, gpu=False, seed=1, precision=1, chunk_size=5000
    )
    default = get_random_coordrot(
        10000, CoordinateRotationERFA, gpu=False, seed=1, precision=1
    )
    small.select_chunk(0, 0)
    large.select_chunk(0, 0)
    default.select_chunk(0, 0)

    np.testing.assert_allclose(
        small.coords_above_horizon, large.coords_above_horizon[:, :100]
    )
    np.testing.assert_allclose(
        small.coords_above_horizon, default.coords_above_horizon[:, :100]
    )


@pytest.mark.parametrize("first_source_antizenith", [True, False])
def test_polarized_flux(first_source_antizenith):
    """Test that using a polarized flux works appropriately."""
    params, sky_model, *_ = get_standard_sim_params(
        use_analytic_beam=False,
        polarized=True,
        nsource=10,
        ntime=2,
        first_source_antizenith=first_source_antizenith,
        use_polarized_sky=True,
    )

    # calculate the frame coherency matrix
    sky_model.calc_frame_coherency()

    coord_mgr = CoordinateRotationAstropy(
        flux=sky_model.frame_coherency.T,
        times=params["times"],
        telescope_loc=params["telescope_loc"],
        skycoords=sky_model.skycoord,
        precision=2,
    )
    coord_mgr.setup()

    # Generate random point sources
    for ti, time in enumerate(params["times"]):
        sky_model.update_positions(
            time=time, telescope_location=params["telescope_loc"]
        )
        uvsim_coherency_matrix = sky_model.coherency_calc()
        n_above_horizon = uvsim_coherency_matrix.shape[-1]
        coord_mgr.rotate(ti)
        _, flux, _ = coord_mgr.select_chunk(0, ti)

        np.testing.assert_allclose(
            uvsim_coherency_matrix.value, flux[:n_above_horizon].T, rtol=1e-8, atol=1e-8
        )


@pytest.mark.gpu
@pytest.mark.parametrize("precision", [1, 2])
@pytest.mark.parametrize(
    ("nsrc", "chunk_size", "source_buffer"),
    [
        (20000, None, 1.0),
        (20000, 5000, 1.0),
        (20000, 5000, 0.7),
        (10003, 3334, 0.7),  # ragged last chunk
        (500, None, 1.0),  # fewer sources than one kernel block
        (7, None, 1.0),
    ],
)
def test_gpu_compaction_matches_where(precision, nsrc, chunk_size, source_buffer):
    """The device horizon cut must reproduce the cp.where one exactly."""
    coords = get_random_coordrot(
        nsrc,
        CoordinateRotationERFA,
        gpu=True,
        seed=7,
        precision=precision,
        chunk_size=chunk_size,
        source_buffer=source_buffer,
    )
    assert coords._use_gpu_compaction

    coords.rotate(0)
    # Reference: the numpy horizon cut applied to the same rotated coordinates.
    topo = cp.asnumpy(coords.all_coords_topo)
    flux = cp.asnumpy(coords.flux)

    for c in range(coords.nchunks):
        crd, flx, n = coords.select_chunk(c, 0)
        crd, flx = cp.asnumpy(crd), cp.asnumpy(flx)

        slc = slice(c * coords.chunk_size, (c + 1) * coords.chunk_size)
        keep = np.nonzero(topo[2, slc] > 0)[0]

        assert n == len(keep)
        # Same sources, in the same order, bit for bit.
        np.testing.assert_array_equal(crd[:, :n], topo[:, slc][:, keep])
        np.testing.assert_array_equal(flx[:n], flux[slc][keep])
        # The padded tail carries zero flux at the zenith.
        np.testing.assert_array_equal(flx[n:], 0)
        np.testing.assert_array_equal(crd[:2, n:], 0)
        np.testing.assert_array_equal(crd[2, n:], 1)


@pytest.mark.gpu
def test_gpu_compaction_overflow():
    """Too small a source_buffer must still raise, not silently drop sources."""
    coords = get_random_coordrot(
        20000,
        CoordinateRotationERFA,
        gpu=True,
        seed=7,
        precision=1,
        chunk_size=5000,
        source_buffer=0.2,
    )
    coords.rotate(0)
    with pytest.raises(ValueError, match="is too small for the number of sources"):
        coords.select_chunk(0, 0)


@pytest.mark.gpu
def test_gpu_compaction_empty_chunk():
    """A chunk with nothing above the horizon reports zero sources."""
    location = Telescope.from_known_telescopes("hera").location
    # All sources at the north celestial pole: never up from HERA.
    n = 16
    skycoords = SkyCoord(
        ra=np.zeros(n) * un.rad,
        dec=np.full(n, np.pi / 2 - 1e-6) * un.rad,
        frame="icrs",
    )
    coords = CoordinateRotationERFA(
        flux=np.ones(n),
        times=Time(np.array([2459863.0]), format="jd", scale="utc"),
        telescope_loc=location,
        skycoords=skycoords,
        gpu=True,
        precision=1,
    )
    coords.setup()
    coords.rotate(0)
    _, flux, nsrcs_up = coords.select_chunk(0, 0)
    assert nsrcs_up == 0
    assert cp.all(flux == 0)
