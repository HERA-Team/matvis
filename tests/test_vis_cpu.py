"""Tests of vis_cpu."""
import pytest

import numpy as np
from astropy.units import sday
from pyuvsim.analyticbeam import AnalyticBeam

from vis_cpu import conversions, simulate_vis, vis_cpu

np.random.seed(0)
NTIMES = 10
NFREQ = 5
NPTSRC = 20

ants = {0: (0, 0, 0), 1: (1, 1, 0)}


def test_vis_cpu():
    """Test basic operation of vis_cpu."""
    # Point source equatorial coords (radians)
    ra = np.linspace(0.0, 2.0 * np.pi, NPTSRC)
    dec = np.linspace(-0.5 * np.pi, 0.5 * np.pi, NPTSRC)

    # Antenna x,y,z positions and frequency array
    antpos = np.array([ants[k] for k in ants.keys()])
    freq = np.linspace(100.0e6, 120.0e6, NFREQ)  # Hz

    # SED for each point source
    fluxes = np.ones(NPTSRC)
    I_sky = fluxes[:, np.newaxis] * (freq[np.newaxis, :] / 100.0e6) ** -2.7

    # Point source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Get coordinate transforms as a function of LST
    hera_lat = -30.7215 * np.pi / 180.0
    lsts = np.linspace(0.0, 2.0 * np.pi, NTIMES)
    eq2tops = np.array(
        [conversions.eci_to_enu_matrix(lst, lat=hera_lat) for lst in lsts]
    )

    # Create beam models
    beam = AnalyticBeam(type="gaussian", diameter=14.0)
    beam_pix = conversions.uvbeam_to_lm(beam, freq, n_pix_lm=63, polarized=False)
    beam_cube = np.array([beam_pix, beam_pix])

    # Run vis_cpu with pixel beams
    for i in range(freq.size):
        _vis = vis_cpu(
            antpos,
            freq[i],
            eq2tops,
            crd_eq,
            I_sky[:, i],
            bm_cube=beam_cube[:, i, :, :],
            precision=1,
            polarized=False,
        )

        assert np.all(~np.isnan(_vis))  # check that there are no NaN values


def test_simulate_vis():
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
    beam = AnalyticBeam(type="gaussian", diameter=14.0)

    # Run vis_cpu with pixel beams
    vis = simulate_vis(
        ants,
        I_sky,
        ra,
        dec,
        freq,
        lsts,
        beams=[beam, beam],
        pixel_beams=True,
        beam_npix=63,
        polarized=False,
        precision=1,
        latitude=-30.7215 * np.pi / 180.0,
    )
    assert np.all(~np.isnan(vis))  # check that there are no NaN values

    # Run vis_cpu with UVBeam beams
    vis = simulate_vis(
        ants,
        I_sky,
        ra,
        dec,
        freq,
        lsts,
        beams=[beam, beam],
        pixel_beams=False,
        polarized=False,
        precision=1,
        latitude=-30.7215 * np.pi / 180.0,
    )
    assert np.all(~np.isnan(vis))  # check that there are no NaN values
