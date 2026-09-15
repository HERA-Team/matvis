"""Tests of matvis_cpu."""

import numpy as np
import pytest
from astropy.time import Time
from pyuvdata.analytic_beam import GaussianBeam
from pyuvdata.beam_interface import BeamInterface
from pyuvdata.telescopes import Telescope

from matvis import simulate_vis
from matvis._test_utils import get_standard_sim_params

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
    hera = Telescope.from_known_telescopes("hera")
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
        ants=ants,
        fluxes=I_sky,
        ra=ra,
        dec=dec,
        freqs=freq,
        times=times,
        beams=[beam, beam],
        polarized=polarized,
        precision=1,
        telescope_loc=hera.location,
        max_progress_reports=2,
    )
    assert np.all(~np.isnan(vis))  # check that there are no NaN values


def test_multibeam_matches_single_beam_cpu():
    """Ensure per-antenna identical beams match a single shared beam on CPU."""
    kw, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=False, nfreq=1, nsource=2, ntime=1
    )
    kw |= {"precision": 2, "use_gpu": False}

    beams = kw.pop("beams")
    beam_idx = np.zeros(len(kw["ants"]), dtype=int)

    vis_multi = simulate_vis(beams=beams * len(kw["ants"]), beam_idx=beam_idx, **kw)
    vis_single = simulate_vis(beams=beams, **kw)

    np.testing.assert_allclose(vis_multi, vis_single, atol=0, rtol=0)


def test_multibeam_permutation_invariant_cpu():
    """Ensure beam-list reordering with remapped beam_idx is invariant on CPU."""
    kw, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=False, nfreq=1, nsource=2, ntime=1
    )
    kw |= {"precision": 2, "use_gpu": False}
    kw.pop("beams")

    beam_a = BeamInterface(GaussianBeam(diameter=14.0), beam_type="power")
    beam_b = BeamInterface(GaussianBeam(diameter=8.0), beam_type="power")

    vis_ab = simulate_vis(
        beams=[beam_a, beam_b],
        beam_idx=np.array([0, 1, 0], dtype=int),
        **kw,
    )
    vis_ba = simulate_vis(
        beams=[beam_b, beam_a],
        beam_idx=np.array([1, 0, 1], dtype=int),
        **kw,
    )

    np.testing.assert_allclose(vis_ab, vis_ba, atol=0, rtol=0)


def test_multibeam_assignment_change_affects_output_cpu():
    """Ensure changing beam_idx assignment changes visibilities on CPU."""
    kw, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=False, nfreq=1, nsource=2, ntime=1
    )
    kw |= {"precision": 2, "use_gpu": False}
    kw.pop("beams")

    beam_a = BeamInterface(GaussianBeam(diameter=14.0), beam_type="power")
    beam_b = BeamInterface(GaussianBeam(diameter=8.0), beam_type="power")

    vis_010 = simulate_vis(
        beams=[beam_a, beam_b],
        beam_idx=np.array([0, 1, 0], dtype=int),
        **kw,
    )
    vis_101 = simulate_vis(
        beams=[beam_a, beam_b],
        beam_idx=np.array([1, 0, 1], dtype=int),
        **kw,
    )

    assert not np.allclose(vis_010, vis_101, rtol=0, atol=1e-12)
