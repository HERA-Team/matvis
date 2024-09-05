"""Test coordinate rotation modules."""

import pytest

import cupy as cp
import numpy as np
from astropy import units as un
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pyuvdata.telescopes import get_telescope

from matvis.core.coords import CoordinateRotation
from matvis.cpu.coords import CoordinateRotationAstropy
from matvis.gpu.coords import GPUCoordinateRotationERFA


def get_angles(x, y):
    """Compute angles between arrays of 3-vectors."""
    xp = cp.get_array_module(x)

    dot = xp.sum(x * y, axis=0)
    norms = xp.sqrt(xp.linalg.norm(x, axis=0) * xp.linalg.norm(y, axis=0))
    ratio = dot / norms
    ratio[ratio > 1.0] = 1.0
    ratio[ratio < -1.0] = -1.0
    return xp.arccos(ratio)


def get_random_coordrot(n, method, gpu, seed):
    """Get a random coordinate rotation object."""
    rng = np.random.default_rng(seed)
    location = get_telescope("hera").location
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
        precision=2,
    )
    coords.setup()
    return coords


@pytest.mark.parametrize("method", list(CoordinateRotation._methods.values()))
@pytest.mark.parametrize("gpu", [False, True])
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


@pytest.mark.parametrize("method", list(CoordinateRotation._methods.values()))
@pytest.mark.parametrize("gpu", [False, True])
def test_accuracy_against_astropy(method, gpu):
    """Test other methods against the benchmark Astropy method."""
    if not gpu and method.requires_gpu:
        pytest.skip()

    astr = get_random_coordrot(1000, CoordinateRotationAstropy, gpu, seed=42)
    coords = get_random_coordrot(1000, method, gpu, seed=42)

    coords.rotate(0)
    astr.rotate(0)

    # get anglular distance between each point in arcsec
    angles = (
        get_angles(coords.all_coords_topo, astr.all_coords_topo) * 180 / np.pi * 3600
    )
    assert len(angles) == 1000
    if gpu:
        angles = angles.get()

    np.testing.assert_allclose(angles, 0, atol=0.01)  # 10 mas
