"""Tests of vis_cpu."""
import pytest

import numpy as np
from astropy.units import sday
from pyuvsim.analyticbeam import AnalyticBeam

from vis_cpu import conversions, simulate_vis, vis_cpu
from vis_cpu.vis_cpu import construct_pixel_beam_spline

np.random.seed(0)
NTIMES = 10
NFREQ = 5
NPTSRC = 20

ants = {0: (0.0, 0.0, 0.0), 1: (20.0, 20.0, 0.0)}


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

    # Check that a wrongly-sized beam cube raises an error
    # for both polarized and unpolarized sims
    nb_bm_pix = beam_cube.shape[-1]
    broken_beam_cube_for_unpol = np.empty((2, 2, 2, nb_bm_pix, nb_bm_pix))
    broken_beam_cube_for_pol = np.empty((1, 2, 2, nb_bm_pix, nb_bm_pix))
    with pytest.raises(AssertionError):
        vis_cpu(
            antpos,
            freq[0],
            eq2tops,
            crd_eq,
            I_sky[:, 0],
            bm_cube=broken_beam_cube_for_unpol,
            precision=1,
            polarized=False,
        )
    with pytest.raises(AssertionError):
        vis_cpu(
            antpos,
            freq[0],
            eq2tops,
            crd_eq,
            I_sky[:, 0],
            bm_cube=broken_beam_cube_for_pol,
            precision=1,
            polarized=True,
        )

    # Check that a complex pixel beam works
    for i in range(freq.size):
        _vis = vis_cpu(
            antpos,
            freq[i],
            eq2tops,
            crd_eq,
            I_sky[:, i],
            bm_cube=(1.0 * 0.1j) * beam_cube[:, i, :, :],
            precision=1,
            polarized=False,
        )
    assert np.all(~np.isnan(_vis))  # check that there are no NaN values

    # Check that errors are raised when beams are input incorrectly
    with pytest.raises(RuntimeError):
        vis_cpu(
            antpos,
            freq[0],
            eq2tops,
            crd_eq,
            I_sky[:, i],
            bm_cube=None,
            beam_list=None,
            precision=1,
            polarized=False,
        )

    with pytest.raises(RuntimeError):
        vis_cpu(
            antpos,
            freq[0],
            eq2tops,
            crd_eq,
            I_sky[:, i],
            bm_cube=beam_cube[:, i, :, :],
            beam_list=[beam, beam],
            precision=1,
            polarized=False,
        )


@pytest.mark.parametrize(
    "pixel_beams,polarized",
    [(True, False), (False, False), (False, True), (True, True)],
)
def test_simulate_vis(pixel_beams, polarized):
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
        pixel_beams=pixel_beams,
        beam_npix=63,
        polarized=polarized,
        precision=1,
        latitude=-30.7215 * np.pi / 180.0,
    )
    assert np.all(~np.isnan(vis))  # check that there are no NaN values


def test_construct_pixel_beam_spline():
    """Test pixel beam interpolation."""
    freqs = np.linspace(100.0e6, 120.0e6, NFREQ)  # Hz

    # Create pixel beam
    beam = AnalyticBeam(type="gaussian", diameter=14.0)
    beams = [beam, beam, beam]  # 3 ants

    beam_pix = [
        conversions.uvbeam_to_lm(bm, freqs, n_pix_lm=20, polarized=False)
        for bm in beams
    ]
    beam_cube = np.array(beam_pix)

    # Complex values should raise an error
    with pytest.raises(TypeError):
        construct_pixel_beam_spline(beam_cube[:, 0] + 1e-5j)  # only 1 freq

    # Mock-up a polarized beam
    bm_cube_pol = beam_cube[np.newaxis, np.newaxis, :, 0, :, :]
    beam_splines_pol = construct_pixel_beam_spline(bm_cube_pol)

    # Check for expected no. of dimensions
    assert len(beam_splines_pol) == 1
    assert len(beam_splines_pol[0]) == 1
    assert len(beam_splines_pol[0][0]) == len(beams)
