"""Compare vis_cpu with pyuvsim visibilities."""
import pytest

import numpy as np

from vis_cpu import simulate_vis

from . import get_standard_sim_params, nants


@pytest.mark.parametrize("use_analytic_beam", (True, False))
@pytest.mark.parametrize("polarized", (True, False))
def test_compare_pyuvsim(polarized, use_analytic_beam):
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
        precision=2,
        latitude=hera_lat * np.pi / 180.0,
        use_gpu=False,
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
        precision=2,
        latitude=hera_lat * np.pi / 180.0,
        use_gpu=True,
    )

    # ---------------------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------------------
    # Loop over baselines and compare
    diff_re = 0.0
    diff_im = 0.0
    rtol = 2e-4 if use_analytic_beam else 0.01
    atol = 5e-4
    for i in range(nants):
        for j in range(i, nants):
            d_visgpu = vis_vg[:, :, 0, 0, i, j] if polarized else vis_vg[:, :, i, j]
            d_viscpu = vis_vc[:, :, 0, 0, i, j] if polarized else vis_vc[:, :, i, j]

            # Keep track of maximum difference
            delta = d_visgpu - d_viscpu
            if np.max(np.abs(delta.real)) > diff_re:
                diff_re = np.max(np.abs(delta.real))
            if np.max(np.abs(delta.imag)) > diff_im:
                diff_im = np.abs(np.max(delta.imag))

            err = f"Max diff: {diff_re:10.10e} + 1j*{diff_im:10.10e}\n"
            err += f"Baseline: ({i},{j})\n"
            err += f"Avg. diff: {delta.mean():10.10e}\n"
            err += f"Max values: \n    uvsim={d_visgpu.max():10.10e}"
            err += f"\n    viscpu={d_viscpu.max():10.10e}"
            assert np.allclose(d_visgpu, d_viscpu, rtol=rtol, atol=atol), err
