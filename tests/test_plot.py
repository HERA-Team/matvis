"""Compare vis_cpu with pyuvsim visibilities."""
import numpy as np
from pyuvsim.analyticbeam import AnalyticBeam

from vis_cpu import conversions, plot

nsource = 10


def test_source_az_za_beam():
    """Test function that calculates the Az and ZA positions of sources."""
    # Observation latitude and LST
    hera_lat = -30.7215
    lst = 0.78

    # Add random sources
    ra = np.random.uniform(low=0.0, high=360.0, size=nsource - 1)
    dec = -30.72 + np.random.random(nsource - 1) * 10.0
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    # Point source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Beam model
    beam = AnalyticBeam(type="gaussian", diameter=14.0)

    # Calculate source locations and positions
    az, za, beamval = plot._source_az_za_beam(
        lst, crd_eq, beam, ref_freq=100.0e6, latitude=np.deg2rad(hera_lat)
    )
    assert np.all(np.isfinite(az))
    assert np.all(np.isfinite(za))
    # (Values of beamval should be NaN below the horizon)


def test_animate_source_map():
    """Test function that animates source positions vs LST."""
    # Observation latitude and LSTs
    hera_lat = -30.7215
    lsts = np.linspace(0.0, 2.0 * np.pi, 5)

    # Add random sources
    ra = np.random.uniform(low=0.0, high=360.0, size=nsource - 1)
    dec = -30.72 + np.random.random(nsource - 1) * 10.0
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    # Beam model
    beam = AnalyticBeam(type="gaussian", diameter=14.0)

    # Generate animation
    anim = plot.animate_source_map(
        ra,
        dec,
        lsts,
        beam,
        interval=200,
        ref_freq=100.0e6,
        latitude=np.deg2rad(hera_lat),
    )
    assert anim is not None
