"""Compare matvis with pyuvsim visibilities."""

import pytest

import numpy as np
from pyuvdata import UVData
from pyuvsim import simsetup, uvsim

from matvis import simulate_vis
from matvis._test_utils import get_standard_sim_params, nants


@pytest.fixture(scope="function")
def default_uvsim() -> UVData:
    """Pyuvsim output for interpolated polarized beam."""
    _, sky_model, beams, beam_dict, uvdata = get_standard_sim_params(
        use_analytic_beam=False, polarized=True, nsource=250
    )

    return uvsim.run_uvdata_uvsim(
        uvdata,
        beams,
        beam_dict=beam_dict,
        catalog=simsetup.SkyModelData(sky_model),
    )


@pytest.mark.parametrize(
    "use_analytic_beam", (True, False), ids=["analytic_beam", "uvbeam"]
)
@pytest.mark.parametrize("polarized", (True, False), ids=["polarized", "unpolarized"])
def test_compare_pyuvsim(polarized, use_analytic_beam):
    """Compare matvis and pyuvsim simulated visibilities."""
    print("Polarized=", polarized, "Analytic Beam =", use_analytic_beam)
    kw, sky_model, uvbeams, bmdict, uvdata = get_standard_sim_params(
        use_analytic_beam, polarized, nsource=250
    )

    vis_matvis = simulate_vis(precision=2, **kw)
    uvd_uvsim = uvsim.run_uvdata_uvsim(
        uvdata,
        uvbeams,
        beam_dict=bmdict,
        catalog=simsetup.SkyModelData(sky_model),
    )

    # ---------------------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------------------
    rtol = 2e-4 if use_analytic_beam else 0.01

    compare_sims(uvd_uvsim, vis_matvis, nants, polarized, rtol)


@pytest.mark.parametrize("min_chunks", (1, 2, 3))
@pytest.mark.parametrize("source_buffer", (1.0, 0.75))
def test_compare_pyuvsim_chunking(min_chunks, source_buffer, default_uvsim):
    """Test chunking and source buffer against pyuvsim."""
    kw, *_ = get_standard_sim_params(
        use_analytic_beam=False, polarized=True, nsource=250
    )

    vis_matvis = simulate_vis(
        precision=2, min_chunks=min_chunks, source_buffer=source_buffer, **kw
    )

    compare_sims(default_uvsim, vis_matvis, nants, polarized=True, rtol=0.01)


def compare_sims(uvd_uvsim, vis_matvis, nants, polarized, rtol):
    """Run the test of comparing matvis and pyuvsim visibilities."""
    # If it passes this test, but fails the following tests, then its probably an
    # ordering issue.
    diff_re = 0.0
    diff_im = 0.0
    atol = 5e-4

    # Loop over baselines and compare
    for i in range(nants):
        for j in range(i, nants):
            for if1, feed1 in enumerate(("X", "Y") if polarized else ("X",)):
                for if2, feed2 in enumerate(("X", "Y") if polarized else ("X",)):
                    d_uvsim = uvd_uvsim.get_data(
                        (i, j, feed1 + feed2)
                    ).T  # pyuvsim visibility
                    d_matvis = (
                        vis_matvis[:, :, i * nants + j, if1, if2]
                        if polarized
                        else vis_matvis[:, :, i * nants + j]
                    )

                    # Keep track of maximum difference
                    delta = d_uvsim - d_matvis
                    if np.max(np.abs(delta.real)) > diff_re:
                        diff_re = np.max(np.abs(delta.real))
                    if np.max(np.abs(delta.imag)) > diff_im:
                        diff_im = np.abs(np.max(delta.imag))

                    err = f"\nMax diff: {diff_re:10.10e} + 1j*{diff_im:10.10e}\n"
                    err += f"Baseline: ({i},{j},{feed1}{feed2})\n"
                    err += f"Avg. diff: {delta.mean():10.10e}\n"
                    err += f"Max values: \n    uvsim={d_uvsim.max():10.10e}"
                    err += f"\n    matvis={d_matvis.max():10.10e}"
                    assert np.allclose(
                        d_uvsim.real,
                        d_matvis.real,
                        rtol=rtol if feed1 == feed2 else rtol * 100,
                        atol=atol,
                    ), err
                    assert np.allclose(
                        d_uvsim.imag,
                        d_matvis.imag,
                        rtol=rtol if feed1 == feed2 else rtol * 100,
                        atol=atol,
                    ), err
