"""Compare matvis CPU and GPU visibilities."""
import pytest

pytest.importorskip("pycuda")

import numpy as np

from matvis import simulate_vis

from . import get_standard_sim_params, nants


@pytest.mark.parametrize("polarized", (True, False))
@pytest.mark.parametrize("use_analytic_beam", (True, False))
@pytest.mark.parametrize("precision", (2,))
def test_cpu_vs_gpu(polarized, use_analytic_beam, precision):
    """Compare matvis CPU and GPU isibilities."""
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
    ) = get_standard_sim_params(use_analytic_beam, polarized)
    print("Polarized=", polarized, "Analytic Beam =", use_analytic_beam)

    # ---------------------------------------------------------------------------
    # (1) Run matvis
    # ---------------------------------------------------------------------------
    vis_cpu = simulate_vis(
        ants=ants,
        fluxes=flux,
        ra=ra,
        dec=dec,
        freqs=freqs,
        lsts=lsts,
        beams=cpu_beams,
        polarized=polarized,
        precision=precision,
        latitude=hera_lat * np.pi / 180.0,
        use_gpu=False,
        beam_spline_opts={"kx": 1, "ky": 1},
        beam_idx=np.zeros(len(ants), dtype=int),
    )

    # ---------------------------------------------------------------------------
    # (2) Run pyuvsim
    # ---------------------------------------------------------------------------
    vis_gpu = simulate_vis(
        ants=ants,
        fluxes=flux,
        ra=ra,
        dec=dec,
        freqs=freqs,
        lsts=lsts,
        beams=cpu_beams,
        polarized=polarized,
        precision=precision,
        latitude=hera_lat * np.pi / 180.0,
        use_gpu=True,
        beam_idx=np.zeros(len(ants), dtype=int),
    )

    # ---------------------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------------------
    rtol = 2e-4 if use_analytic_beam else 0.01
    atol = 5e-4
    np.testing.assert_allclose(vis_gpu.real, vis_cpu.real, rtol=rtol, atol=atol)
    np.testing.assert_allclose(vis_gpu.imag, vis_cpu.imag, rtol=rtol, atol=atol)
