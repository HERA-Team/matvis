"""Compare matvis CPU and GPU visibilities."""

import pytest

pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

import numpy as np

from matvis import simulate_vis
from matvis._test_utils import get_standard_sim_params, nants, perturbed_beam
from matvis.redundancy import antpairs_to_blocks


@pytest.mark.parametrize("polarized", (True, False))
@pytest.mark.parametrize("use_analytic_beam", (True, False))
@pytest.mark.parametrize("precision", (1, 2))
@pytest.mark.parametrize("min_chunks", (1, 2))
@pytest.mark.parametrize("source_buffer", (1.0, 0.75))
def test_cpu_vs_gpu(polarized, use_analytic_beam, precision, min_chunks, source_buffer):
    """Compare matvis CPU and GPU visibilities."""
    kw, *_ = get_standard_sim_params(use_analytic_beam, polarized, nsource=250)
    # (
    #     _,
    #     ants,
    #     flux,
    #     ra,
    #     dec,
    #     freqs,
    #     lsts,
    #     beams,
    #     _,
    #     _,
    #     lat,
    #     _,
    # ) = get_standard_sim_params(use_analytic_beam, polarized, nsource=250)
    print("Polarized=", polarized, "Analytic Beam =", use_analytic_beam)

    kw |= {
        "precision": precision,
        "min_chunks": min_chunks,
        "source_buffer": source_buffer,
    }
    # No beam_spline_opts on either side: both backends take order and mode from
    # matvis.core.beams.DEFAULT_SPLINE_OPTS, so this also guards them agreeing.
    # (This test used to pass order=1 to the CPU only, and passed because the
    # GPU's default happened to be 1 as well.)
    vis_cpu = simulate_vis(use_gpu=False, **kw)
    vis_gpu = simulate_vis(use_gpu=True, **kw)

    # ---------------------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------------------
    rtol = 2e-4 if use_analytic_beam else 0.01
    atol = 5e-4
    np.testing.assert_allclose(vis_gpu.real, vis_cpu.real, rtol=rtol, atol=atol)
    np.testing.assert_allclose(vis_gpu.imag, vis_cpu.imag, rtol=rtol, atol=atol)


@pytest.mark.parametrize("perturbation", ["feed_phase", "rotated_feed"])
@pytest.mark.parametrize("matprod_method", ["GPUMatMul", "GPUVectorDot", "GPUMatBlock"])
def test_cpu_vs_gpu_per_antenna_beams(perturbation, matprod_method):
    """GPU matches CPU when antennas have different beams, for every ordered pair.

    The CPU result is checked against pyuvsim in test_compare_pyuvsim.py.
    """
    kw, *_ = get_standard_sim_params(use_analytic_beam=False, polarized=True)
    beam0 = kw["beams"][0]
    kw["beams"] = [beam0, beam0.clone(beam=perturbed_beam(beam0.beam, perturbation))]
    kw |= {"precision": 2, "beam_idx": np.arange(nants) % 2}

    extra = {}
    if matprod_method == "GPUMatBlock":
        # Blocks hold only i <= j, so the reversed pairs come from the
        # Hermitian-conjugate path.
        extra["antenna_blocks"] = antpairs_to_blocks(
            [(i, j) for i in range(nants) for j in range(i, nants)]
        )

    vis_cpu = simulate_vis(use_gpu=False, matprod_method="CPUMatMul", **kw)
    vis_gpu = simulate_vis(use_gpu=True, matprod_method=matprod_method, **extra, **kw)

    rtol, atol = 0.01, 5e-4
    np.testing.assert_allclose(vis_gpu.real, vis_cpu.real, rtol=rtol, atol=atol)
    np.testing.assert_allclose(vis_gpu.imag, vis_cpu.imag, rtol=rtol, atol=atol)
