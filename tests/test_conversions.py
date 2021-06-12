"""Tests of coordinate conversions."""
import pytest

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.coordinates.builtin_frames import AltAz
from astropy.time import Time

from vis_cpu import conversions

np.random.seed(0)
NTIMES = 4
NFREQ = 5
NPTSRC = 20


def test_equatorial_to_cosines():
    """Test conversions from RA, Dec to direction cosines.

    This tests the conversion of RA and Dec angles to a Cartesian x,y,z unit
    vector in an Earth-Centered Inertial (ECI) coordinate system aligned with
    the celestial pole, i.e. for (x,y,z)
    (RA=0  deg, Dec=0  deg) = (1, 0, 0)
    (RA=90 deg, Dec=0  deg) = (0, 1, 0)
    (RA=0  deg, Dec=90 deg) = (0, 0, 1)
    """
    # Point source equatorial coords in assumed ECI system (deg)
    ra = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 360.0]) * np.pi / 180.0

    # ---------------------------------------------------------------------------
    # (1) Source at Dec = +90 degrees (North Celestial Pole)
    # ---------------------------------------------------------------------------
    # Get ECI direction cosines for RA, Dec1 (source at Dec=90)
    dec1 = 0.5 * np.pi * np.ones(ra.size)  # Dec = 90 deg
    crd_eq1 = conversions.point_source_crd_eq(ra, dec1)

    # The x,y directions should be zero, and the z direction should be 1
    assert np.allclose(crd_eq1[0], 0.0 * np.ones(ra.size))
    assert np.allclose(crd_eq1[1], 0.0 * np.ones(ra.size))
    assert np.allclose(crd_eq1[2], 1.0 * np.ones(ra.size))

    # ---------------------------------------------------------------------------
    # (2) Source at Dec = -90 degrees (South Celestial Pole)
    # ---------------------------------------------------------------------------
    # Get ECI direction cosines for RA, -Dec1 (source at Dec=-90)
    crd_eq1 = conversions.point_source_crd_eq(ra, -dec1)

    # The x,y directions should be zero, and the z direction should be 1
    assert np.allclose(crd_eq1[0], 0.0 * np.ones(ra.size))
    assert np.allclose(crd_eq1[1], 0.0 * np.ones(ra.size))
    assert np.allclose(crd_eq1[2], -1.0 * np.ones(ra.size))

    # ---------------------------------------------------------------------------
    # (3) Source at Dec = 0 degrees (Celestial Equator)
    # ---------------------------------------------------------------------------
    # Get ECI direction cosines for RA, 0 (source at Dec=0)
    crd_eq1 = conversions.point_source_crd_eq(ra, 0.0 * dec1)

    # The z direction should be 0; in plane perpendicular to Celestial poles
    assert np.isclose(crd_eq1[0, 0], 1.0)  # RA=0, should be in x direction
    assert np.isclose(crd_eq1[1, 0], 0.0)
    assert np.isclose(crd_eq1[0, 2], 0.0)  # RA=90, should be in y direction
    assert np.isclose(crd_eq1[1, 2], 1.0)
    assert np.allclose(crd_eq1[2], 0.0 * np.ones(ra.size))  # Perp. to poles


def test_equatorial_to_enu():
    """Test conversions from RA, Dec to ENU and Az/ZA coordinates."""
    # HERA latitude
    hera_lat = -30.7215 * np.pi / 180.0
    lsts = np.linspace(0.0, 2.0 * np.pi, NTIMES)

    # Point source equatorial coords in assumed ECI system (deg)
    # RA is at zenith crossing, i.e. hour angle HA = LST - RA = 0 => RA = LST
    # So, one of these sources should be found exactly at the zenith at each LST
    ra = lsts
    dec = hera_lat * np.ones(ra.size)  # Dec = HERA latitude

    # Get Cartesian coords for these sources in our ECI system
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Loop over LSTs
    for i, lst in enumerate(lsts):

        # Rotation matrices from ECI <-> ENU
        mat_eci_to_enu = conversions.eci_to_enu_matrix(lst, lat=hera_lat)
        mat_enu_to_eci = conversions.enu_to_eci_matrix(lst, lat=hera_lat)

        # Convert to local direction cosines l,m,n
        tx, ty, tz = crd_top = np.dot(mat_eci_to_enu, crd_eq)

        # Ensure that the source with RA = lst is exactly overhead
        assert np.isclose(tx[i], 0.0)
        assert np.isclose(ty[i], 0.0)
        assert np.isclose(tz[i], 1.0)

        # Ensure that inverse-transformed coords match for all sources
        crd_eq2 = np.dot(mat_enu_to_eci, crd_top)
        assert np.allclose(crd_eq, crd_eq2)

        # Check that zenith angle is zero when source with RA = lst is overhead
        az, za = conversions.enu_to_az_za(tx, ty, orientation="astropy")
        az_uvb, za_uvb = conversions.enu_to_az_za(tx, ty, orientation="uvbeam")
        assert np.isclose(za[i], 0.0)
        assert np.isclose(za_uvb[i], 0.0)

        # Check that E, N, U are aligned with Az/ZA as expected
        vec_e = np.array([1.0, 0.0, 0.0])
        vec_n = np.array([0.0, 1.0, 0.0])
        vec_u = np.array([0.0, 0.0, 1.0])

        # Pointing East (astropy)
        _az, _za = conversions.enu_to_az_za(vec_e[0], vec_e[1], orientation="astropy")
        assert np.isclose(_az, 0.5 * np.pi)
        assert np.isclose(_za, 0.5 * np.pi)

        # Pointing North (astropy)
        _az, _za = conversions.enu_to_az_za(vec_n[0], vec_n[1], orientation="astropy")
        assert np.isclose(_az, 0.0)
        assert np.isclose(_za, 0.5 * np.pi)

        # Pointing Up (astropy)
        _az, _za = conversions.enu_to_az_za(vec_u[0], vec_u[1], orientation="astropy")
        assert np.isclose(_az, 0.0)
        assert np.isclose(_za, 0.0)

        # Pointing East (UVBeam)
        _az, _za = conversions.enu_to_az_za(vec_e[0], vec_e[1], orientation="uvbeam")
        assert np.isclose(_az, 0.0)
        assert np.isclose(_za, 0.5 * np.pi)

        # Pointing North (UVBeam)
        _az, _za = conversions.enu_to_az_za(vec_n[0], vec_n[1], orientation="uvbeam")
        assert np.isclose(_az, 0.5 * np.pi)
        assert np.isclose(_za, 0.5 * np.pi)

        # Pointing Up (UVBeam)
        _az, _za = conversions.enu_to_az_za(vec_u[0], vec_u[1], orientation="uvbeam")
        assert np.isclose(_za, 0.0)

        # Do inverse transform and check
        eq_e = np.dot(mat_enu_to_eci, vec_e)
        _tx, _ty, _tz = np.dot(mat_eci_to_enu, eq_e)
        assert np.isclose(_tx, 1.0)
        assert np.isclose(_ty, 0.0)
        assert np.isclose(_tz, 0.0)
