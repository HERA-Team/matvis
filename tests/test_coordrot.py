"""Test coordinate rotation modules."""

import numpy as np
import pytest
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pyuvdata.telescopes import Telescope

from matvis import HAVE_GPU
from matvis._test_utils import get_standard_sim_params
from matvis.core.coords import CoordinateRotation
from matvis.cpu import coords as cpu_coords
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


def unfused_bcrs(eci, astrom):
    """Evaluate BCRS coordinates with the original three-pass implementation."""
    coords = CoordinateRotationERFA.__new__(CoordinateRotationERFA)
    coords.xp = np
    result = eci.copy()
    coords._ld(result, astrom["eh"], astrom["em"], 1e-6)
    coords._ab(result, astrom["v"], astrom["em"], astrom["bm1"])
    coords._bpn(result, astrom)
    return result


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


@pytest.mark.parametrize("precision", [1, 2])
def test_coord_rot_erfa_set_bcrs(precision):
    """Test that setting bcrs before setup works as expected."""
    normal = get_random_coordrot(
        1000, CoordinateRotationERFA, gpu=False, seed=1, precision=precision
    )
    bcrs = get_random_coordrot(
        1000,
        CoordinateRotationERFA,
        gpu=False,
        seed=1,
        precision=precision,
        setup=False,
    )
    bcrs._set_bcrs(0)
    bcrs.setup()

    normal.rotate(0)
    bcrs.rotate(0)

    np.testing.assert_allclose(normal.all_coords_topo, bcrs.all_coords_topo)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_bcrs_matches_unfused_near_solar_limiter(dtype):
    """The fused CPU kernel preserves each correction near the solar limiter."""
    qdqpe = np.array([0.0, 0.5e-6, 1.0e-6, 2.0e-6])
    theta = np.arccos(1.0 - qdqpe)
    eci = np.array(
        [-np.cos(theta), np.sin(theta), np.zeros_like(theta)], dtype=dtype
    )
    assert eci.dtype == dtype
    astrom = {
        "eh": np.array([1.0, 0.0, 0.0]),
        "em": 0.983,
        "v": np.array([2.1e-5, -8.7e-5, 3.4e-5]),
        "bm1": 0.9999999954,
        "bpn": np.array(
            [
                [0.9999999, -3.0e-4, 2.0e-4],
                [3.0e-4, 0.99999995, -1.0e-4],
                [-2.0e-4, 1.0e-4, 0.99999997],
            ]
        ),
    }
    expected = unfused_bcrs(eci, astrom)
    actual = np.empty_like(eci)

    cpu_coords._fused_bcrs(
        eci,
        astrom["eh"],
        astrom["em"],
        astrom["v"],
        astrom["bm1"],
        astrom["bpn"],
        actual,
    )

    tolerance = 2e-7 if dtype == np.float32 else 3e-16
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("precision", [1, 2])
@pytest.mark.parametrize(
    ("date", "location"),
    [
        ("2000-01-01T12:00:00", EarthLocation.from_geodetic(0.0, 0.0, 0.0)),
        (
            "2025-06-21T03:15:00",
            EarthLocation.from_geodetic(21.4283, -30.7215, 1073.0),
        ),
    ],
)
def test_fused_bcrs_matches_unfused_dates_sites(precision, date, location):
    """The fused CPU path matches the previous arithmetic for dates and sites."""
    rng = np.random.default_rng(3829)
    skycoords = SkyCoord(
        ra=rng.uniform(0, 2 * np.pi, 64) * un.rad,
        dec=rng.uniform(-np.pi / 2, np.pi / 2, 64) * un.rad,
        frame="icrs",
    )
    coords = CoordinateRotationERFA(
        flux=np.ones(64),
        times=Time([date]),
        telescope_loc=location,
        skycoords=skycoords,
        precision=precision,
    )
    coords._eci = coords._eci.astype(coords.rtype)
    assert coords._eci.dtype == coords.rtype
    astrom = coords._apco(coords._get_obsf(coords.times[0], location))
    expected = unfused_bcrs(coords._eci, astrom)

    coords._set_bcrs(0, astrom)

    tolerance = 2e-7 if precision == 1 else 3e-16
    np.testing.assert_allclose(coords._bcrs, expected, rtol=0.0, atol=tolerance)


def test_set_bcrs_retains_refresh_cache():
    """The CPU fusion retains the requested BCRS refresh interval."""
    coords = get_random_coordrot(
        32,
        CoordinateRotationERFA,
        gpu=False,
        seed=901,
        precision=2,
        setup=False,
        update_bcrs_every=60.0,
    )
    coords.times = Time(
        [2459863.0, 2459863.0 + 30 / 86400, 2459863.0 + 90 / 86400],
        format="jd",
    )
    astrom = coords._apco(coords._get_obsf(coords.times[0], coords.telescope_loc))
    modified_astrom = astrom.copy()
    modified_astrom["bpn"] = -astrom["bpn"]

    coords._set_bcrs(0, astrom)
    first = coords._bcrs.copy()
    coords._set_bcrs(1, modified_astrom)
    np.testing.assert_array_equal(coords._bcrs, first)
    assert coords._time_of_last_evaluation == 0

    coords._set_bcrs(2, modified_astrom)
    np.testing.assert_allclose(coords._bcrs, -first, rtol=0.0, atol=5e-16)
    assert coords._time_of_last_evaluation == 2


def test_set_bcrs_gpu_retains_array_module_path(monkeypatch):
    """GPU-backed rotators continue to use their existing correction methods."""
    coords = get_random_coordrot(
        8, CoordinateRotationERFA, gpu=False, seed=176, precision=2, setup=False
    )
    coords.gpu = True
    astrom = coords._apco(coords._get_obsf(coords.times[0], coords.telescope_loc))
    calls = []

    def record(name):
        original = getattr(coords, name)

        def wrapped(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(coords, "_ld", record("_ld"))
    monkeypatch.setattr(coords, "_ab", record("_ab"))
    monkeypatch.setattr(coords, "_bpn", record("_bpn"))
    monkeypatch.setattr(
        cpu_coords,
        "_fused_bcrs",
        lambda *args, **kwargs: pytest.fail("GPU path called the CPU kernel"),
        raising=False,
    )

    coords._set_bcrs(0, astrom)

    assert calls == ["_ld", "_ab", "_bpn"]


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
