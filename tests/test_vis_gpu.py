"""Tests of functionality of vis_gpu."""
import pytest

pytest.importorskip("pycuda")

import numpy as np
from pyuvsim.analyticbeam import AnalyticBeam

from vis_cpu import simulate_vis

from . import get_standard_sim_params


def test_gpu_with_spline_opts():
    """Compare vis_cpu and pyuvsim simulated visibilities."""
    (
        sky_model,
        ants,
        flux,
        ra,
        dec,
        freqs,
        lsts,
        cpu_beams,
        uvsim_beams,
        beam_dict,
        hera_lat,
        uvdata,
    ) = get_standard_sim_params(True, False)

    with pytest.warns(
        UserWarning,
        match="You have passed beam_spline_opts, but these are not used in GPU",
    ):
        simulate_vis(
            ants=ants,
            fluxes=flux,
            ra=ra,
            dec=dec,
            freqs=freqs,
            lsts=lsts,
            beams=cpu_beams,
            polarized=False,
            precision=2,
            latitude=hera_lat * np.pi / 180.0,
            use_gpu=True,
            beam_spline_opts={"kx": 1, "ky": 1},
            beam_idx=np.zeros(len(ants), dtype=int),
        )


def test_mixed_beams(uvbeam):
    """Test that error is raised when using a mixed beam list."""
    (
        sky_model,
        ants,
        flux,
        ra,
        dec,
        freqs,
        lsts,
        cpu_beams,
        uvsim_beams,
        beam_dict,
        hera_lat,
        uvdata,
    ) = get_standard_sim_params(True, False)

    anl = AnalyticBeam("gaussian", diameter=14.0)
    cpu_beams = [uvbeam, anl, anl]

    with pytest.raises(ValueError, match="vis_gpu only support beam_lists with either"):
        simulate_vis(
            ants=ants,
            fluxes=flux,
            ra=ra,
            dec=dec,
            freqs=freqs,
            lsts=lsts,
            beams=cpu_beams,
            polarized=False,
            precision=2,
            latitude=hera_lat * np.pi / 180.0,
            use_gpu=True,
            beam_spline_opts={"kx": 1, "ky": 1},
            beam_idx=np.zeros(len(ants), dtype=int),
        )


def test_single_precision():
    """Test that using single precision on gpu works."""
    (
        sky_model,
        ants,
        flux,
        ra,
        dec,
        freqs,
        lsts,
        cpu_beams,
        uvsim_beams,
        beam_dict,
        hera_lat,
        uvdata,
    ) = get_standard_sim_params(
        polarized=True, use_analytic_beam=False, nfreq=1, nsource=2, ntime=1
    )

    vis_1 = simulate_vis(
        ants=ants,
        fluxes=flux,
        ra=ra,
        dec=dec,
        freqs=freqs,
        lsts=lsts,
        beams=cpu_beams,
        polarized=True,
        precision=1,
        latitude=hera_lat * np.pi / 180.0,
        use_gpu=True,
    )

    vis_2 = simulate_vis(
        ants=ants,
        fluxes=flux,
        ra=ra,
        dec=dec,
        freqs=freqs,
        lsts=lsts,
        beams=cpu_beams,
        polarized=True,
        precision=2,
        latitude=hera_lat * np.pi / 180.0,
        use_gpu=True,
    )

    np.testing.assert_allclose(vis_1, vis_2, atol=1e-5, rtol=0)
