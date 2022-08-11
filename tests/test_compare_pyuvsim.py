"""Compare vis_cpu with pyuvsim visibilities."""
import pytest

import numpy as np
from pathlib import Path
from pyuvsim import simsetup, uvsim

from vis_cpu import simulate_vis

from . import get_standard_sim_params, nants


@pytest.mark.parametrize("use_analytic_beam", (True, False))
@pytest.mark.parametrize("polarized", (True, False))
def test_compare_pyuvsim(polarized, use_analytic_beam):
    """Compare vis_cpu and pyuvsim simulated visibilities."""
    print("Polarized=", polarized, "Analytic Beam =", use_analytic_beam)
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
    )

    # ---------------------------------------------------------------------------
    # (2) Run pyuvsim
    # ---------------------------------------------------------------------------
    uvd_uvsim = uvsim.run_uvdata_uvsim(
        uvdata,
        uvsim_beams,
        beam_dict=beam_dict,
        catalog=simsetup.SkyModelData(sky_model),
    )

    # ---------------------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------------------
    # Loop over baselines and compare
    diff_re = 0.0
    diff_im = 0.0
    rtol = 2e-4 if use_analytic_beam else 0.01
    atol = 5e-4

    # If it passes this test, but fails the following tests, then its probably an
    # ordering issue.
    for i in range(nants):
        for j in range(i, nants):
            for if1, feed1 in enumerate(("X", "Y") if polarized else ("X",)):
                for if2, feed2 in enumerate(("X", "Y") if polarized else ("X",)):

                    d_uvsim = uvd_uvsim.get_data(
                        (i, j, feed1 + feed2)
                    ).T  # pyuvsim visibility
                    d_viscpu = (
                        vis_vc[:, :, if1, if2, i, j]
                        if polarized
                        else vis_vc[:, :, i, j]
                    )

                    # Keep track of maximum difference
                    delta = d_uvsim - d_viscpu
                    if np.max(np.abs(delta.real)) > diff_re:
                        diff_re = np.max(np.abs(delta.real))
                    if np.max(np.abs(delta.imag)) > diff_im:
                        diff_im = np.abs(np.max(delta.imag))

                    err = f"\nMax diff: {diff_re:10.10e} + 1j*{diff_im:10.10e}\n"
                    err += f"Baseline: ({i},{j},{feed1}{feed2})\n"
                    err += f"Avg. diff: {delta.mean():10.10e}\n"
                    err += f"Max values: \n    uvsim={d_uvsim.max():10.10e}"
                    err += f"\n    viscpu={d_viscpu.max():10.10e}"
                    assert np.allclose(
                        d_uvsim.real,
                        d_viscpu.real,
                        rtol=rtol if feed1 == feed2 else rtol * 100,
                        atol=atol,
                    ), err
                    assert np.allclose(
                        d_uvsim.imag,
                        d_viscpu.imag,
                        rtol=rtol if feed1 == feed2 else rtol * 100,
                        atol=atol,
                    ), err
