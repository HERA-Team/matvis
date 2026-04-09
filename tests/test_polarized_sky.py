"""Accuracy tests comparing old (sqrt I) vs new (eigendecomp/sign-split) approaches.

Both code paths coexist in the same simulate_vis function:
- Old path: simulate_vis(fluxes=...) with no stokes param
- New path: simulate_vis(stokes=...) with full Stokes params
"""

import pytest

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


class TestBackwardCompatibility:
    """Tests that stokes=[I,0,0,0] matches the old sqrt(I) path exactly."""

    @pytest.mark.parametrize("precision,atol", [(2, 1e-12), (1, 1e-4)])
    def test_unpolarized_stokes_matches_existing(self, precision, atol):
        """Critical test: stokes=[I,0,0,0] must match old path to machine precision."""
        params = _make_sim_params(
            nsrc=15, nant=3, ntime=3, nfreq=2, precision=precision
        )
        rng = np.random.default_rng(99)

        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = rng.uniform(0.5, 5.0, (nsrc, nfreq))

        # OLD PATH — no stokes, runs existing sqrt(0.5*I) code
        vis_old = simulate_vis(fluxes=fluxes, polarized=True, **params)

        # NEW PATH — stokes=[I,0,0,0], runs eigendecomp M matrix code
        stokes = np.zeros((4, nsrc, nfreq))
        stokes[0] = fluxes  # I = fluxes, Q=U=V=0
        vis_new = simulate_vis(fluxes=fluxes, polarized=True, stokes=stokes, **params)

        np.testing.assert_allclose(vis_new, vis_old, atol=atol)


class TestPolarizedSkySanity:
    """Sanity checks for polarized sky simulation."""

    def test_polarized_sky_no_nans(self):
        """Polarized sky with eigendecomp should produce no NaNs."""
        params = _make_sim_params(nsrc=10, nant=3, ntime=2, nfreq=1)
        rng = np.random.default_rng(55)

        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = rng.uniform(1.0, 5.0, (nsrc, nfreq))

        stokes = np.zeros((4, nsrc, nfreq))
        stokes[0] = fluxes  # I
        stokes[1] = 0.2 * fluxes  # Q = 0.2 * I
        stokes[2] = 0.1 * fluxes  # U = 0.1 * I
        stokes[3] = 0.05 * fluxes  # V = 0.05 * I

        vis = simulate_vis(fluxes=fluxes, polarized=True, stokes=stokes, **params)

        assert not np.any(np.isnan(vis))
        assert not np.any(np.isinf(vis))
        # Visibilities should be nonzero
        assert np.any(np.abs(vis) > 0)

    def test_stokes_only_no_fluxes(self):
        """Passing only stokes (no fluxes) should work via auto-derivation."""
        params = _make_sim_params(nsrc=10, nant=3, ntime=2, nfreq=1)
        rng = np.random.default_rng(55)

        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = rng.uniform(1.0, 5.0, (nsrc, nfreq))

        stokes = np.zeros((4, nsrc, nfreq))
        stokes[0] = fluxes
        stokes[1] = 0.2 * fluxes

        # Pass stokes without fluxes
        vis = simulate_vis(polarized=True, stokes=stokes, **params)

        assert not np.any(np.isnan(vis))
        assert np.any(np.abs(vis) > 0)


class TestSignSplit:
    """Test sign-split approach for negative flux handling."""

    def test_sign_split_vs_eigendecomp_positive_sky(self):
        """For all-positive eigenvalues, sign-split and eigendecomp must match."""
        params = _make_sim_params(nsrc=10, nant=3, ntime=2, nfreq=1)
        rng = np.random.default_rng(33)

        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = rng.uniform(1.0, 5.0, (nsrc, nfreq))

        stokes = np.zeros((4, nsrc, nfreq))
        stokes[0] = fluxes
        stokes[1] = 0.1 * fluxes
        stokes[2] = 0.05 * fluxes
        stokes[3] = 0.02 * fluxes

        # Eigendecomp (default: raise_on_negative_flux=True)
        vis_eigen = simulate_vis(fluxes=fluxes, polarized=True, stokes=stokes, **params)

        # Sign-split
        vis_split = simulate_vis(
            fluxes=fluxes,
            polarized=True,
            stokes=stokes,
            raise_on_negative_flux=False,
            **params,
        )

        np.testing.assert_allclose(vis_split, vis_eigen, atol=1e-10)

    def test_sign_split_negative_flux_no_nans(self):
        """Sign-split with negative I sources should produce no NaNs."""
        params = _make_sim_params(nsrc=10, nant=3, ntime=2, nfreq=1)
        rng = np.random.default_rng(44)

        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes_abs = rng.uniform(0.5, 5.0, (nsrc, nfreq))

        # Make half the sources negative (EOR-like)
        signs = np.ones(nsrc)
        signs[: nsrc // 2] = -1
        fluxes = fluxes_abs * signs[:, np.newaxis]

        stokes = np.zeros((4, nsrc, nfreq))
        stokes[0] = fluxes  # Some negative I

        vis = simulate_vis(
            polarized=True,
            stokes=stokes,
            raise_on_negative_flux=False,
            **params,
        )

        assert not np.any(np.isnan(vis))
        assert not np.any(np.isinf(vis))

    def test_raises_on_negative_flux(self):
        """Default (raise_on_negative_flux=True) should raise on negative eigenvalues."""
        params = _make_sim_params(nsrc=5, nant=2, ntime=1, nfreq=1)

        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = np.ones((nsrc, nfreq))

        stokes = np.zeros((4, nsrc, nfreq))
        stokes[0, 0, 0] = -5.0  # Negative I for first source

        with pytest.raises(ValueError, match="Negative eigenvalue"):
            simulate_vis(fluxes=fluxes, polarized=True, stokes=stokes, **params)


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_intensity(self):
        """All-zero Stokes should give zero visibilities."""
        params = _make_sim_params(nsrc=5, nant=2, ntime=1, nfreq=1)

        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        stokes = np.zeros((4, nsrc, nfreq))

        vis = simulate_vis(polarized=True, stokes=stokes, **params)

        np.testing.assert_allclose(vis, 0.0, atol=1e-15)
