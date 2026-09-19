"""Tests of the wrapper function, `simulate_vis`.

These are mainly for testing how the wrapping works,
not for testing actual simulations.
"""

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pyuvdata.analytic_beam import GaussianBeam

from matvis import simulate_vis
from matvis._test_utils import get_standard_sim_params
from matvis.gpu.gpu import HAVE_CUDA


def test_passing_matprod_method_with_prefix():
    """Test that passing different coordinate methods to `simulate_vis` works."""
    vis = simulate_vis(
        ants={0: (0.0, 0.0, 0.0)},
        fluxes=np.array([[1.0]]),
        ra=np.array([0.0]),
        dec=np.array([0.0]),
        freqs=np.array([100.0e6]),
        times=Time([2459863.5], format="jd"),
        beams=[GaussianBeam(diameter=14.0)],
        polarized=False,
        precision=1,
        telescope_loc=EarthLocation.from_geodetic(0.0, 0.0, 0.0),
        matprod_method="CPUMatMul",
        use_gpu=False,
    )
    assert np.iscomplexobj(
        vis
    )  # check that the output is complex, as expected for CPUMatMul


def test_simulate_vis_with_matblock_matches_default():
    """A full-covering ``tile_antennas`` block set must reproduce the CPUMatMul result.

    Run through the public ``simulate_vis`` API, with ``antenna_blocks`` as an
    explicit, typed parameter.
    """
    from matvis.redundancy import tile_antennas

    kw, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=False, nsource=15
    )
    nant = len(kw["ants"])

    vis_default = simulate_vis(
        precision=1, matprod_method="CPUMatMul", use_gpu=False, **kw
    )
    vis_block = simulate_vis(
        precision=1,
        matprod_method="CPUMatBlock",
        antenna_blocks=tile_antennas(nant, chunk_size=2),
        use_gpu=False,
        **kw,
    )
    np.testing.assert_allclose(vis_block, vis_default, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "use_gpu",
    [
        pytest.param(False, id="cpu"),
        pytest.param(
            True,
            id="gpu",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not HAVE_CUDA, reason="GPU is not available"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("polarized", [False, True])
def test_matblock_with_scattered_blocks_matches_default(use_gpu, polarized):
    """MatBlock must be exact even when its blocks force an antenna relabelling.

    The drivers reorder the antenna axis of ``Z`` so that scattered blocks become
    sliceable (issue #161). That reordering has to be undone consistently on the
    output side, and has to compose with the beam indexing -- so drive it through
    the public API and compare against the plain full product.
    """
    from matvis.redundancy import contiguity_order

    kw, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=polarized, nsource=15
    )
    nant = len(kw["ants"])

    # Interleaved blocks: nothing is a contiguous run in the natural order, so
    # the driver must actually permute to get any benefit.
    ev, od = np.arange(0, nant, 2), np.arange(1, nant, 2)
    blocks = [(ev, ev), (ev, od), (od, ev), (od, od)]
    assert not np.array_equal(contiguity_order(blocks, nant), np.arange(nant))

    vis_default = simulate_vis(
        precision=1,
        matprod_method="GPUMatMul" if use_gpu else "CPUMatMul",
        use_gpu=use_gpu,
        **kw,
    )
    vis_block = simulate_vis(
        precision=1,
        matprod_method="GPUMatBlock" if use_gpu else "CPUMatBlock",
        antenna_blocks=blocks,
        use_gpu=use_gpu,
        **kw,
    )
    np.testing.assert_allclose(vis_block, vis_default, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "use_gpu",
    [
        pytest.param(False, id="cpu"),
        pytest.param(
            True,
            id="gpu",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not HAVE_CUDA, reason="GPU is not available"),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("kwarg", "value", "errmsg"),
    [
        (
            "source_buffer",
            0.0,
            "source_buffer must satisfy 0 < source_buffer <= 1",
        ),
        (
            "memory_buffer",
            1.1,
            "memory_buffer must satisfy 0 < memory_buffer <= 1",
        ),
    ],
)
def test_buffer_validation(use_gpu, kwarg, value, errmsg):
    """Ensure buffer validation is enforced for both CPU and GPU wrapper calls."""
    kw, *_ = get_standard_sim_params(False, False)

    with pytest.raises(ValueError, match=errmsg):
        simulate_vis(use_gpu=use_gpu, **{kwarg: value}, **kw)
