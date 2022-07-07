"""Tests of vis_cpu."""
import pytest

import numpy as np
from pyuvsim.analyticbeam import AnalyticBeam

from vis_cpu import simulate_vis

np.random.seed(0)
NTIMES = 10
NFREQ = 5
NPTSRC = 20

ants = {0: (0.0, 0.0, 0.0), 1: (20.0, 20.0, 0.0)}


@pytest.mark.parametrize(
    "polarized",
    [True, False],
)
def test_simulate_vis(polarized):
    """Test basic operation of simple wrapper around vis_cpu, `simulate_vis`."""
    # Point source equatorial coords (radians)
    ra = np.linspace(0.0, 2.0 * np.pi, NPTSRC)
    dec = np.linspace(-0.5 * np.pi, 0.5 * np.pi, NPTSRC)

    # Antenna x,y,z positions and frequency array
    freq = np.linspace(100.0e6, 120.0e6, NFREQ)  # Hz

    # SED for each point source
    fluxes = np.ones(NPTSRC)
    I_sky = fluxes[:, np.newaxis] * (freq[np.newaxis, :] / 100.0e6) ** -2.7

    # Get coordinate transforms as a function of LST
    lsts = np.linspace(0.0, 2.0 * np.pi, NTIMES)

    # Create beam models
    beam = AnalyticBeam("gaussian", diameter=14.0)

    # Run vis_cpu with pixel beams
    vis = simulate_vis(
        ants,
        I_sky,
        ra,
        dec,
        freq,
        lsts,
        beams=[beam, beam],
        polarized=polarized,
        precision=1,
        latitude=-30.7215 * np.pi / 180.0,
    )
    assert np.all(~np.isnan(vis))  # check that there are no NaN values
