"""Compare vis_cpu with pyuvsim visibilities."""
import pytest

pytest.importorskip("pycuda")

import numpy as np

from vis_cpu import simulate_vis

from . import get_standard_sim_params, nants


@pytest.mark.parametrize("polarized", (True, False))
@pytest.mark.parametrize("use_analytic_beam", (True, False))
@pytest.mark.parametrize("precision", (2,))
def test_cpu_vs_gpu(polarized, use_analytic_beam, precision):
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
    ) = get_standard_sim_params(use_analytic_beam, polarized)
    print("Polarized=", polarized, "Analytic Beam =", use_analytic_beam)

    # ---------------------------------------------------------------------------
    # (1) Run vis_cpu
    # ---------------------------------------------------------------------------
    vis_vc = simulate_vis(
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
    vis_vg = simulate_vis(
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
    np.testing.assert_allclose(vis_vg.real, vis_vc.real, rtol=rtol, atol=atol)
    np.testing.assert_allclose(vis_vg.imag, vis_vc.imag, rtol=rtol, atol=atol)
