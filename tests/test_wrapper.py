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


@pytest.mark.parametrize(
    "use_gpu",
    [
        pytest.param(False, id="cpu"),
        pytest.param(
            True,
            id="gpu",
            marks=pytest.mark.skipif(not HAVE_CUDA, reason="GPU is not available"),
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
