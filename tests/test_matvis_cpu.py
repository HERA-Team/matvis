"""Tests of matvis_cpu."""

import pytest

import numpy as np
from astropy.time import Time
from pyuvdata.analytic_beam import GaussianBeam
from pyuvdata.telescopes import get_telescope

from matvis import simulate_vis

NTIMES = 10
NFREQ = 5
NPTSRC = 20

ants = {0: (0.0, 0.0, 0.0), 1: (20.0, 20.0, 0.0)}


@pytest.mark.parametrize(
    "polarized",
    [True, False],
)
def test_simulate_vis(polarized):
    """Test basic operation of simple wrapper around matvis, `simulate_vis`."""
    # Point source equatorial coords (radians)
    hera = get_telescope("hera")
    ra = np.linspace(0.0, 2.0 * np.pi, NPTSRC)
    dec = np.linspace(-0.5 * np.pi, 0.5 * np.pi, NPTSRC)

    # Antenna x,y,z positions and frequency array
    freq = np.linspace(100.0e6, 120.0e6, NFREQ)  # Hz

    # SED for each point source
    fluxes = np.ones(NPTSRC)
    I_sky = fluxes[:, np.newaxis] * (freq[np.newaxis, :] / 100.0e6) ** -2.7

    # Get coordinate transforms as a function of LST
    times = Time(np.linspace(2459863.0, 2459864.0, NTIMES), format="jd")

    # Create beam models
    beam = GaussianBeam(diameter=14.0)

    # Run matvis on CPUs with pixel beams
    vis = simulate_vis(
        ants,
        I_sky,
        ra,
        dec,
        freq,
        times,
        beams=[beam, beam],
        polarized=polarized,
        precision=1,
        telescope_loc=hera.location,
        max_progress_reports=2,
    )
    assert np.all(~np.isnan(vis))  # check that there are no NaN values
