"""Compare matvis CPU and GPU visibilities."""
import pytest

pytest.importorskip("pycuda")

import numpy as np

from matvis import simulate_vis

from . import get_standard_sim_params, nants


@pytest.mark.parametrize("polarized", (True, False))
@pytest.mark.parametrize("use_analytic_beam", (True, False))
@pytest.mark.parametrize("precision", (2,))
@pytest.mark.parametrize("min_chunks", (1, 2))
@pytest.mark.parametrize("source_buffer", (1.0, 0.75))
def test_cpu_vs_gpu(polarized, use_analytic_beam, precision, min_chunks, source_buffer):
    """Compare matvis CPU and GPU visibilities."""
    (
        _,
        ants,
        flux,
        ra,
        dec,
        freqs,
        lsts,
        beams,
        _,
        _,
        lat,
        _,
    ) = get_standard_sim_params(use_analytic_beam, polarized, nsource=250)
    print("Polarized=", polarized, "Analytic Beam =", use_analytic_beam)

    kwargs = {
        "ants": ants,
        "fluxes": flux,
        "ra": ra,
        "dec": dec,
        "freqs": freqs,
        "lsts": lsts,
        "beams": beams,
        "polarized": polarized,
        "precision": precision,
        "latitude": lat * np.pi / 180.0,
        "beam_idx": np.zeros(len(ants), dtype=int),
        "min_chunks": min_chunks,
        "source_buffer": source_buffer,
    }
    # ---------------------------------------------------------------------------
    # (1) Run matvis
    # ---------------------------------------------------------------------------
    vis_cpu = simulate_vis(use_gpu=False, beam_spline_opts={"kx": 1, "ky": 1}, **kwargs)

    # ---------------------------------------------------------------------------
    # (2) Run pyuvsim
    # ---------------------------------------------------------------------------
    vis_gpu = simulate_vis(use_gpu=True, **kwargs)

    # ---------------------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------------------
    rtol = 2e-4 if use_analytic_beam else 0.01
    atol = 5e-4
    np.testing.assert_allclose(vis_gpu.real, vis_cpu.real, rtol=rtol, atol=atol)
    np.testing.assert_allclose(vis_gpu.imag, vis_cpu.imag, rtol=rtol, atol=atol)
