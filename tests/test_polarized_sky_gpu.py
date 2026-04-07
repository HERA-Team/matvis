"""GPU tests for polarized sky model with eigendecomposition.

These tests mirror tests/test_polarized_sky.py but run on the GPU backend,
comparing GPU results against CPU to verify correctness of the polarized
sky code paths in src/matvis/gpu/gpu.py.
"""

import pytest

pytest.importorskip("cupy")

import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pyuvdata.analytic_beam import GaussianBeam

from matvis import simulate_vis


def _make_sim_params(nsrc=10, nant=3, ntime=2, nfreq=1, precision=2):
    """Create minimal simulation parameters for testing."""
    rng = np.random.default_rng(42)

    # Antennas
    ants = {i: rng.uniform(0, 50, 3) * np.array([1, 1, 0]) for i in range(nant)}

    # HERA-like location
    telescope_loc = EarthLocation(
        lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0 * un.m
    )

    # Sources around zenith
    ra = rng.uniform(0, 2 * np.pi, nsrc)
    dec = rng.uniform(-0.6, -0.4, nsrc)  # near HERA latitude

    # Frequencies
    freqs = np.linspace(100e6, 120e6, nfreq)

    # Times
    times = Time(np.linspace(2459863.0, 2459863.01, ntime), format="jd", scale="utc")

    # Beam
    beams = [GaussianBeam(diameter=14.0)]

    return {
        "ants": ants,
        "ra": ra,
        "dec": dec,
        "freqs": freqs,
        "times": times,
        "beams": beams,
        "telescope_loc": telescope_loc,
        "precision": precision,
    }


def test_gpu_stokes_matches_cpu():
    """GPU eigendecomp path should match CPU for polarized sky (negative_flux='raise')."""
    params = _make_sim_params(nsrc=15, nant=3, ntime=2, nfreq=1, precision=2)
    rng = np.random.default_rng(99)

    nsrc = len(params["ra"])
    nfreq = len(params["freqs"])
    fluxes = rng.uniform(0.5, 5.0, (nsrc, nfreq))

    stokes = np.zeros((4, nsrc, nfreq))
    stokes[0] = fluxes
    stokes[1] = 0.2 * fluxes  # Q
    stokes[2] = 0.1 * fluxes  # U
    stokes[3] = 0.05 * fluxes  # V

    vis_cpu = simulate_vis(
        fluxes=fluxes,
        polarized=True,
        stokes=stokes,
        negative_flux="raise",
        use_gpu=False,
        beam_spline_opts={"order": 1},
        **params,
    )
    vis_gpu = simulate_vis(
        fluxes=fluxes,
        polarized=True,
        stokes=stokes,
        negative_flux="raise",
        use_gpu=True,
        **params,
    )

    np.testing.assert_allclose(vis_gpu.real, vis_cpu.real, rtol=2e-4, atol=5e-4)
    np.testing.assert_allclose(vis_gpu.imag, vis_cpu.imag, rtol=2e-4, atol=5e-4)


def test_gpu_sign_split_matches_cpu():
    """GPU sign-split path should match CPU for all-positive sky."""
    params = _make_sim_params(nsrc=10, nant=3, ntime=2, nfreq=1, precision=2)
    rng = np.random.default_rng(33)

    nsrc = len(params["ra"])
    nfreq = len(params["freqs"])
    fluxes = rng.uniform(1.0, 5.0, (nsrc, nfreq))

    stokes = np.zeros((4, nsrc, nfreq))
    stokes[0] = fluxes
    stokes[1] = 0.1 * fluxes
    stokes[2] = 0.05 * fluxes
    stokes[3] = 0.02 * fluxes

    vis_cpu = simulate_vis(
        fluxes=fluxes,
        polarized=True,
        stokes=stokes,
        negative_flux="split",
        use_gpu=False,
        beam_spline_opts={"order": 1},
        **params,
    )
    vis_gpu = simulate_vis(
        fluxes=fluxes,
        polarized=True,
        stokes=stokes,
        negative_flux="split",
        use_gpu=True,
        **params,
    )

    np.testing.assert_allclose(vis_gpu.real, vis_cpu.real, rtol=2e-4, atol=5e-4)
    np.testing.assert_allclose(vis_gpu.imag, vis_cpu.imag, rtol=2e-4, atol=5e-4)


def test_gpu_sign_split_negative_flux():
    """GPU sign-split with negative-I sources should match CPU."""
    params = _make_sim_params(nsrc=10, nant=3, ntime=2, nfreq=1, precision=2)
    rng = np.random.default_rng(44)

    nsrc = len(params["ra"])
    nfreq = len(params["freqs"])
    fluxes_abs = rng.uniform(0.5, 5.0, (nsrc, nfreq))

    # Make half the sources negative (EOR-like)
    signs = np.ones(nsrc)
    signs[: nsrc // 2] = -1

    stokes = np.zeros((4, nsrc, nfreq))
    stokes[0] = fluxes_abs * signs[:, np.newaxis]  # Some negative I
    stokes[1] = 0.1 * fluxes_abs
    stokes[2] = 0.05 * fluxes_abs
    stokes[3] = 0.02 * fluxes_abs

    vis_cpu = simulate_vis(
        fluxes=fluxes_abs,
        polarized=True,
        stokes=stokes,
        negative_flux="split",
        use_gpu=False,
        beam_spline_opts={"order": 1},
        **params,
    )
    vis_gpu = simulate_vis(
        fluxes=fluxes_abs,
        polarized=True,
        stokes=stokes,
        negative_flux="split",
        use_gpu=True,
        **params,
    )

    np.testing.assert_allclose(vis_gpu.real, vis_cpu.real, rtol=2e-4, atol=5e-4)
    np.testing.assert_allclose(vis_gpu.imag, vis_cpu.imag, rtol=2e-4, atol=5e-4)


def test_gpu_negative_flux_ignore():
    """GPU with negative_flux='ignore' should match CPU."""
    params = _make_sim_params(nsrc=10, nant=3, ntime=1, nfreq=1, precision=2)
    rng = np.random.default_rng(77)

    nsrc = len(params["ra"])
    nfreq = len(params["freqs"])
    fluxes = rng.uniform(0.5, 5.0, (nsrc, nfreq))

    stokes = np.zeros((4, nsrc, nfreq))
    stokes[0] = fluxes.copy()
    stokes[0, : nsrc // 2] *= -1  # Half negative

    vis_cpu = simulate_vis(
        fluxes=np.abs(fluxes),
        polarized=True,
        stokes=stokes,
        negative_flux="ignore",
        use_gpu=False,
        beam_spline_opts={"order": 1},
        **params,
    )
    vis_gpu = simulate_vis(
        fluxes=np.abs(fluxes),
        polarized=True,
        stokes=stokes,
        negative_flux="ignore",
        use_gpu=True,
        **params,
    )

    np.testing.assert_allclose(vis_gpu.real, vis_cpu.real, rtol=2e-4, atol=5e-4)
    np.testing.assert_allclose(vis_gpu.imag, vis_cpu.imag, rtol=2e-4, atol=5e-4)


def test_gpu_negative_flux_invalid():
    """Invalid negative_flux value should raise ValueError on GPU."""
    params = _make_sim_params(nsrc=5, nant=2, ntime=1, nfreq=1)

    nsrc = len(params["ra"])
    nfreq = len(params["freqs"])
    fluxes = np.ones((nsrc, nfreq))
    stokes = np.zeros((4, nsrc, nfreq))
    stokes[0] = fluxes

    with pytest.raises(ValueError, match="negative_flux must be"):
        simulate_vis(
            fluxes=fluxes,
            polarized=True,
            stokes=stokes,
            negative_flux="invalid",
            use_gpu=True,
            **params,
        )
