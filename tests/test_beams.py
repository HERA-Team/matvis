"""Test that pixel and analytic beams are properly aligned."""
import pytest

import numpy as np
import pyuvdata.utils as uvutils
from pathlib import Path
from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH
from pyuvsim import AnalyticBeam
from typing import List

from vis_cpu import conversions, simulate_vis, vis_cpu

np.random.seed(0)
NTIMES = 3
NFREQ = 2
NPTSRC = 4000
ants = {0: (0, 0, 0), 1: (1, 1, 0)}

cst_file = Path(DATA_PATH) / "NicCSTbeams" / "HERA_NicCST_150MHz.txt"


class EllipticalBeam(object):
    """Add ellipticity/shearing to an existing UVBeam/AnalyticBeam object."""

    def __init__(self, base_beam, xstretch=1.0, ystretch=1.0, rotation=0.0):
        """
        Take an existing UVBeam/AnalyticBeam and apply stretching/rotation.

        Parameters
        ----------
        base_beam : UVBeam or AnalyticBeam object
            Existing beam object that will be sheared/stretched/rotated.

        xstretch, ystretch : float, optional
            Stretching factors to apply to the beam in the x and y directions,
            which introduces beam ellipticity, as well as an overall
            stretching/shrinking. Default: 1.0 (no ellipticity or stretching).

        rotation : float, optional
            Rotation of the beam in the x-y plane, in degrees. Only has an
            effect if xstretch != ystretch. Default: 0.0.
        """
        self.base_beam = base_beam
        self.xstretch = xstretch
        self.ystretch = ystretch
        self.rotation = rotation

    @property
    def beam_type(self) -> str:
        """Whether the beam is `power` or `efield`."""
        return self.base_beam.beam_type

    def efield_to_power(self, **kwargs):
        """Convert from efield to power beam."""
        self.base_beam.efield_to_power(**kwargs)

    @property
    def polarization_array(self):
        """The polarization array of the base beam."""
        return self.base_beam.polarization_array

    def interp(self, az_array, za_array, freq_array):
        """Evaluate the beam after applying shearing, stretching, or rotation.

        Parameters
        ----------
        az_array : array_like
            Azimuth values in radians (same length as za_array). The azimuth
            here has the UVBeam convention: North of East(East=0, North=pi/2)

        za_array : array_like
            Zenith angle values in radians (same length as az_array).

        freq_array : array_like
            Frequency values to evaluate at.

        Returns
        -------
        interp_data : array_like
            Array of beam values, shape (Naxes_vec, Nspws, Nfeeds or Npols,
            Nfreqs or freq_array.size if freq_array is passed,
            Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)

        interp_basis_vector : array_like
            Array of interpolated basis vectors (or self.basis_vector_array
            if az/za_arrays are not passed), shape: (Naxes_vec, Ncomponents_vec,
            Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)
        """
        # Apply shearing, stretching, or rotation
        if self.xstretch != 1.0 or self.ystretch != 1.0:
            # Convert sheared Cartesian coords to circular polar coords
            # mX stretches in x direction, mY in y direction, a is angle
            # Notation: phi = az, theta = za. Subscript 's' are transformed coords
            a = self.rotation * np.pi / 180.0
            X = za_array * np.cos(az_array)
            Y = za_array * np.sin(az_array)
            Xs = (X * np.cos(a) - Y * np.sin(a)) / self.xstretch
            Ys = (X * np.sin(a) + Y * np.cos(a)) / self.ystretch

            # Updated polar coordinates
            theta_s = np.sqrt(Xs ** 2.0 + Ys ** 2.0)
            phi_s = np.arccos(Xs / theta_s)
            phi_s[Ys < 0.0] *= -1.0

            # Fix coordinates below the horizon of the unstretched beam
            theta_s[np.where(theta_s < 0.0)] = 0.5 * np.pi
            theta_s[np.where(theta_s >= np.pi / 2.0)] = 0.5 * np.pi

            # Update za_array and az_array
            az_array, za_array = phi_s, theta_s

        # Call interp() method on BaseBeam
        interp_data, interp_basis_vector = self.base_beam.interp(
            az_array=az_array, za_array=za_array, freq_array=freq_array
        )

        return interp_data, interp_basis_vector


def make_cst_beam(beam_type):
    """Make the default CST testing beam."""
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        beam_type=beam_type,
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    return beam


@pytest.fixture(scope="module")
def freq() -> np.ndarray:
    """Frequencies for tests."""
    return np.linspace(100.0e6, 120.0e6, NFREQ)  # Hz


@pytest.fixture(scope="function")
def beam_list_unpol() -> List[EllipticalBeam]:
    """Get Gaussian beam and transform into an elliptical version."""
    base_beam = AnalyticBeam("gaussian", diameter=14.0)
    beam_analytic = EllipticalBeam(base_beam, xstretch=2.2, ystretch=1.0, rotation=40.0)
    beam_analytic = conversions.prepare_beam(
        beam_analytic, polarized=False, use_feed="x"
    )

    return [beam_analytic, beam_analytic]


@pytest.fixture(scope="function")
def beam_list_pol() -> List[EllipticalBeam]:
    """Get Gaussian beam and transform into an elliptical version with polarization."""
    base_beam = AnalyticBeam("gaussian", diameter=14.0)
    beam_analytic = EllipticalBeam(base_beam, xstretch=2.2, ystretch=1.0, rotation=40.0)
    beam_analytic = conversions.prepare_beam(
        beam_analytic, polarized=True, use_feed="x"
    )

    return [beam_analytic, beam_analytic]


@pytest.fixture(scope="function")
def beam_cube(beam_list_unpol, freq) -> np.ndarray:
    """Construct pixel beam from analytic beam."""
    beam_pix = conversions.uvbeam_to_lm(
        beam_list_unpol[0], freq, n_pix_lm=1000, polarized=False
    )
    return np.array([beam_pix, beam_pix])


@pytest.fixture(scope="module")
def point_source_pos():
    """Some simple point source positions."""
    ra = np.linspace(0.0, 2.0 * np.pi, NPTSRC)
    dec = np.linspace(-0.5 * np.pi, 0.5 * np.pi, NPTSRC)

    return ra, dec


@pytest.fixture(scope="module")
def sky_flux(freq):
    """Array of sky intensity."""
    fluxes = np.ones(NPTSRC)
    return fluxes[:, np.newaxis] * (freq[np.newaxis, :] / 100.0e6) ** -2.7


@pytest.fixture(scope="module")
def crd_eq(point_source_pos):
    """Equatorial coordinates for the point sources."""
    ra, dec = point_source_pos
    return conversions.point_source_crd_eq(ra, dec)


@pytest.fixture(scope="module")
def eq2tops():
    """Get coordinate transforms as a function of LST."""
    hera_lat = -30.7215 * np.pi / 180.0
    lsts = np.linspace(0.0, 2.0 * np.pi, NTIMES)
    return np.array([conversions.eci_to_enu_matrix(lst, lat=hera_lat) for lst in lsts])


@pytest.fixture(scope="module")
def antpos():
    """Antenna positions in the test array."""
    return np.array([ants[k] for k in ants.keys()])


def test_beam_interpolation(
    beam_list_unpol, beam_cube, crd_eq, eq2tops, sky_flux, freq, antpos
):
    """Test that interpolated beams and UVBeam agree on coordinates."""
    # Run vis_cpu with pixel beams and analytic beams (uses precision=2)
    # This test is useful for checking a particular line in vis_cpu() that
    # evaluates the pixel beam splines, i.e. `splines[p1][p2][i](ty, tx, ...)`
    # This test should fail if the order of the arguments (ty, tx) is wrong
    for i in range(freq.size):
        # Pixel beams
        vis_pix = vis_cpu(
            antpos,
            freq[i],
            eq2tops,
            crd_eq,
            sky_flux[:, i],
            bm_cube=beam_cube[:, i, :, :],
            precision=2,
            polarized=False,
        )

        # Analytic beams
        vis_analytic = vis_cpu(
            antpos,
            freq[i],
            eq2tops,
            crd_eq,
            sky_flux[:, i],
            beam_list=beam_list_unpol,
            precision=2,
            polarized=False,
        )

        assert np.all(~np.isnan(vis_pix))  # check that there are no NaN values
        assert np.all(~np.isnan(vis_analytic))

        # Check that results are close (they should be for 1000^2 pixel-beams
        # if the elliptical beams are both oriented the same way)
        np.testing.assert_allclose(vis_pix, vis_analytic, rtol=1e-5, atol=1e-5)


def test_beam_interpolation_pol():
    """Test beam interpolation for polarized beams."""
    # Frequency array
    freq = np.linspace(100.0e6, 120.0e6, NFREQ)  # Hz

    # Get Gaussian beam and transform into an elliptical version
    base_beam = AnalyticBeam("gaussian", diameter=14.0)
    beam_analytic = EllipticalBeam(base_beam, xstretch=2.2, ystretch=1.0, rotation=40.0)

    # Construct pixel beam from analytic beam
    beam_pix = conversions.uvbeam_to_lm(
        beam_analytic, freq, n_pix_lm=20, polarized=True
    )
    assert np.all(~np.isnan(beam_pix))
    assert np.all(~np.isinf(beam_pix))

    # Check that unpolarized beam pixelization has 2 fewer dimensions
    beam_pix_unpol = conversions.uvbeam_to_lm(
        beam_analytic, freq, n_pix_lm=200, polarized=False
    )
    assert len(beam_pix.shape) == len(beam_pix_unpol.shape) + 2


def test_polarized_not_efield(beam_list_unpol, crd_eq, eq2tops, sky_flux, freq, antpos):
    """Test that when doing polarized sim, error is raised if beams aren't efield."""
    with pytest.raises(ValueError, match="beam type must be efield"):
        vis_cpu(
            antpos,
            freq[0],
            eq2tops,
            crd_eq,
            sky_flux[:, 0],
            beam_list=beam_list_unpol,
            precision=2,
            polarized=True,
        )


def test_unpolarized_efield(beam_list_pol, crd_eq, eq2tops, sky_flux, freq, antpos):
    """Test that when doing unpolarized sim, error is raised if beams aren't power."""
    with pytest.raises(ValueError, match="beam type must be power"):
        vis_cpu(
            antpos,
            freq[0],
            eq2tops,
            crd_eq,
            sky_flux[:, 0],
            beam_list=beam_list_pol,
            precision=2,
            polarized=False,
        )


def test_prepare_beams_wrong_feed():
    """Test that error is raised feed not in 'xy'."""
    base_beam = AnalyticBeam("gaussian", diameter=14.0)
    beam_analytic = EllipticalBeam(base_beam, xstretch=2.2, ystretch=1.0, rotation=40.0)
    with pytest.raises(ValueError, match="use_feed must be"):
        conversions.prepare_beam(beam_analytic, polarized=False, use_feed="z")


def test_prepare_beams_pol_power():
    """Test that error is raised if power beam passed to polarized sim."""
    base_beam = AnalyticBeam("gaussian", diameter=14.0)
    beam_analytic = EllipticalBeam(base_beam, xstretch=2.2, ystretch=1.0, rotation=40.0)
    beam_analytic.efield_to_power()

    with pytest.raises(ValueError, match="Beam type must be efield"):
        conversions.prepare_beam(beam_analytic, polarized=True, use_feed="x")


def test_prepare_beam_unpol_uvbeam():
    """Test that prepare_beam correctly handles an efield beam input to unpol sim."""
    beam = make_cst_beam("efield")
    new_beam = conversions.prepare_beam(beam, polarized=False, use_feed="x")

    assert new_beam.beam_type == "power"
    assert len(new_beam.polarization_array) == 1
    assert uvutils.polnum2str(new_beam.polarization_array[0]).lower() == "xx"

    assert beam.beam_type == "efield"


def test_prepare_beam_unpol_uvbeam_npols():
    """Test that prepare_beam correctly handles multiple pols to unpol simulation."""
    beam = make_cst_beam("power")
    new_beam = conversions.prepare_beam(beam, polarized=False, use_feed="x")

    assert new_beam.beam_type == "power"
    assert len(new_beam.polarization_array) == 1
    assert uvutils.polnum2str(new_beam.polarization_array[0]).lower() == "xx"

    assert len(beam.polarization_array) > 1


def test_prepare_beam_unpol_uvbeam_pol_no_exist():
    """Test that error is raised if desired polarization doesn't exist."""
    beam = make_cst_beam("efield")
    beam.efield_to_power()

    beam.select(polarizations=[uvutils.polstr2num("yy"), uvutils.polstr2num("xy")])

    with pytest.raises(
        ValueError, match="You want to use x feed, but it does not exist in the UVBeam"
    ):
        conversions.prepare_beam(beam, polarized=False, use_feed="x")
