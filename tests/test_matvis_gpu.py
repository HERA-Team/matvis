"""Tests of functionality of matvis_gpu."""

import pytest

pytest.importorskip("cupy")

import numpy as np
from pyuvdata.analytic_beam import GaussianBeam

from matvis import simulate_vis
from matvis._test_utils import get_standard_sim_params


def test_antizenith():
    """Ensure that a single source at anti-zenith produces zero visibilities."""
    kw, *_ = get_standard_sim_params(
        True, False, nsource=1, first_source_antizenith=True
    )

    vis = simulate_vis(
        precision=2, use_gpu=True, beam_spline_opts={"kx": 1, "ky": 1}, **kw
    )

    assert np.all(vis == 0)


def test_multibeam():
    """Ensure that running with multiple beams of the same kind gives the same answer as a single beam."""
    kw, *_ = get_standard_sim_params(
        False, False, nsource=1, first_source_antizenith=True
    )
    kw |= {
        "precision": 2,
        "use_gpu": True,
        "beam_spline_opts": {"kx": 1, "ky": 1},
        "beam_idx": np.zeros(len(kw["ants"]), dtype=int),
    }
    beams = kw.pop("beams")

    vis1 = simulate_vis(beams=beams * len(kw["ants"]), **kw)
    vis2 = simulate_vis(beams=beams, **kw)

    assert np.all(vis1 == vis2)


def test_mixed_beams(uvbeam):
    """Test that error is raised when using a mixed beam list."""
    kw, *_ = get_standard_sim_params(use_analytic_beam=True, polarized=False)

    anl = GaussianBeam(diameter=14.0)
    cpu_beams = [uvbeam, anl, anl]
    del kw["beams"]

    with pytest.raises(
        ValueError, match="GPUBeamInterpolator only supports beam_lists with either"
    ):
        simulate_vis(beams=cpu_beams, use_gpu=True, **kw)


def test_single_precision():
    """Test that using single precision on gpu works."""
    polarized = True
    kw, *_ = get_standard_sim_params(
        polarized=polarized, use_analytic_beam=False, nfreq=1, nsource=2, ntime=1
    )

    kw |= {"use_gpu": True}

    vis_2 = simulate_vis(precision=2, **kw)
    vis_1 = simulate_vis(precision=1, **kw)

    np.testing.assert_allclose(vis_1, vis_2, atol=1e-5, rtol=0)
