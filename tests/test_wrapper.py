"""Tests of the wrapper function, `simulate_vis`.

These are mainly for testing how the wrapping works,
not for testing actual simulations.
"""

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pyuvdata.analytic_beam import GaussianBeam

from matvis import simulate_vis


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
