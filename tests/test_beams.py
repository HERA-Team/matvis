"""Test that pixel and analytic beams are properly aligned."""
import numpy as np
from pyuvsim import AnalyticBeam

from vis_cpu import conversions, simulate_vis, vis_cpu

np.random.seed(0)
NTIMES = 3
NFREQ = 2
NPTSRC = 400
ants = {0: (0, 0, 0), 1: (1, 1, 0)}


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


def test_beam_interpolation():
    """Test that interpolated beams and UVBeam agree on coordinates."""
    # Point source equatorial coords (radians)
    ra = np.linspace(0.0, 2.0 * np.pi, NPTSRC)
    dec = np.linspace(-0.5 * np.pi, 0.5 * np.pi, NPTSRC)

    # Antenna x,y,z positions and frequency array
    antpos = np.array([ants[k] for k in ants.keys()])
    freq = np.linspace(100.0e6, 120.0e6, NFREQ)  # Hz

    # SED for each point source
    fluxes = np.ones(NPTSRC)
    I_sky = fluxes[:, np.newaxis] * (freq[np.newaxis, :] / 100.0e6) ** -2.7

    # Get Gaussian beam and transform into an elliptical version
    base_beam = AnalyticBeam("gaussian", diameter=14.0)
    beam_analytic = EllipticalBeam(base_beam, xstretch=2.2, ystretch=1.0, rotation=40.0)
    beam_list = [beam_analytic, beam_analytic]

    # Construct pixel beam from analytic beam
    beam_pix = conversions.uvbeam_to_lm(
        beam_analytic, freq, n_pix_lm=1000, polarized=False
    )
    beam_cube = np.array([beam_pix, beam_pix])

    # Point source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Get coordinate transforms as a function of LST
    hera_lat = -30.7215 * np.pi / 180.0
    lsts = np.linspace(0.0, 2.0 * np.pi, NTIMES)
    eq2tops = np.array(
        [conversions.eci_to_enu_matrix(lst, lat=hera_lat) for lst in lsts]
    )

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
            I_sky[:, i],
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
            I_sky[:, i],
            beam_list=beam_list,
            precision=2,
            polarized=False,
        )

        assert np.all(~np.isnan(vis_pix))  # check that there are no NaN values
        assert np.all(~np.isnan(vis_analytic))

        # Check that results are close (they should be for 1000^2 pixel-beams
        # if the elliptical beams are both oriented the same way)
        assert np.allclose(vis_pix, vis_analytic, rtol=1e-5, atol=1e-5)


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
