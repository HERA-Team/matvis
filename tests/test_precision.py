"""Validate single-precision results against double precision.

Single precision is the intended production mode (2x faster GEMM on datacenter
GPUs, half the memory), so this test acts as the accuracy gate for it: the two
precisions must agree to a tolerance consistent with float32 round-off
accumulated over the source sum.
"""

import numpy as np
import pytest

from matvis import HAVE_GPU, simulate_vis
from matvis._test_utils import get_standard_sim_params


@pytest.mark.parametrize("use_gpu", [False, True] if HAVE_GPU else [False])
@pytest.mark.parametrize("polarized", (True, False))
@pytest.mark.parametrize("use_analytic_beam", (True, False))
def test_single_vs_double_precision(use_gpu, polarized, use_analytic_beam):
    """Single-precision visibilities must match double precision within fp32 error."""
    kw, *_ = get_standard_sim_params(use_analytic_beam, polarized, nsource=250)
    if use_gpu:
        kw |= {"min_chunks": 2, "source_buffer": 0.75}

    vis_double = simulate_vis(use_gpu=use_gpu, precision=2, **kw)
    vis_single = simulate_vis(use_gpu=use_gpu, precision=1, **kw)

    assert vis_single.dtype == np.complex64
    assert vis_double.dtype == np.complex128

    # The error budget is fp32 round-off accumulated over the coherent source
    # sum: relative to the total flux scale (~|V| at zero spacing), not to each
    # individual (possibly near-zero) visibility.
    scale = np.abs(vis_double).max()
    np.testing.assert_allclose(vis_single, vis_double, atol=1e-5 * scale, rtol=0)
