"""Functions for converting coordinates."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pyuvdata.utils as uvutils
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.coordinates.builtin_frames import AltAz
from astropy.time import Time
from copy import deepcopy
from numpy import typing as npt
from pyuvdata.uvbeam import UVBeam
from scipy.linalg import orthogonal_procrustes as ortho_procr
from typing import Literal

from . import HAVE_GPU

if HAVE_GPU:
    import cupy as cp

    get_array_module = cp.get_array_module
else:

    def get_array_module(*x):
        """Return numpy as the array module."""
        return np


def enu_to_az_za(enu_e, enu_n, orientation="astropy", periodic_azimuth=True):
    """Convert angle cosines in ENU coordinates into azimuth and zenith angle.

    For a pointing vector in East-North-Up (ENU) coordinates vec{p}, the input
    arguments are ``enu_e = vec{p}.hat{e}`` and ``enu_n = vec{p}.hat{n}`, where
    ``hat{e}`` is a unit vector in ENU coordinates etc.

    For a drift-scan telescope pointing at the zenith, the ``hat{e}`` direction
    is aligned with the ``U`` direction (in the UVW plane), which means that we
    can identify the direction cosines ``l = enu_e`` and ``m = enu_n``.

    Azimuth is oriented East of North, i.e. Az(N) = 0 deg, Az(E) = +90 deg in
    the astropy convention, and North of East, i.e. Az(N) = +90 deg, and
    Az(E) = 0 deg in the UVBeam convention.

    Parameters
    ----------
    enu_e, enu_n : array_like
        Normalized angle cosine coordinates on the interval (-1, +1).

    orientation : str, optional
        Orientation convention used for the azimuth angle. The default is
        ``'astropy'``, which uses an East of North convention (Az(N) = 0 deg,
        Az(E) = +90 deg). Alternatively, the ``'uvbeam'`` convention uses
        North of East (Az(N) = +90 deg, Az(E) = 0 deg).

    periodic_azimuth : bool, optional
        if True, constrain az to be betwee 0 and 2 * pi
        This avoids the issue that arctan2 outputs angles between -pi and pi
        while most CST beam formats store azimuths between 0 and 2pi which leads
        interpolation domain mismatches.

    Returns
    -------
    az, za : array_like
        Corresponding azimuth and zenith angles (in radians).
    """
    xp = get_array_module(enu_e, enu_n)

    assert orientation in [
        "astropy",
        "uvbeam",
    ], "orientation must be either 'astropy' or 'uvbeam'"

    lsqr = enu_n**2.0 + enu_e**2.0
    mask = lsqr < 1
    zeta = xp.zeros_like(lsqr)
    zeta[mask] = xp.sqrt(1 - lsqr[mask])

    az = xp.arctan2(enu_e, enu_n)
    za = 0.5 * np.pi - xp.arcsin(zeta)

    # Flip and rotate azimuth coordinate if uvbeam convention is used
    if orientation == "uvbeam":
        az = 0.5 * np.pi - az
    if periodic_azimuth:
        az = xp.mod(az, 2 * np.pi)
    return az, za


def eci_to_enu_matrix(ha, lat):
    """3x3 transformation matrix to rotate ECI to ENU coordinates.

    Transformation matrix to project Earth-Centered Inertial (ECI) coordinates
    to local observer-centric East-North-Up (ENU) coordinates at a given time
    and location.

    The ECI coordinates are aligned with the celestial pole, i.e. for (x,y,z)
    (RA=0  deg, Dec=0  deg) = (1, 0, 0)
    (RA=90 deg, Dec=0  deg) = (0, 1, 0)
    (RA=0  deg, Dec=90 deg) = (0, 0, 1)

    Note: This is a stripped-down version of the ``eq2top_m`` function.

    Parameters
    ----------
    ha : float
        Hour angle, in radians, where HA = LST - RA.

    lat : float
        Latitude of the observer, in radians.

    Returns
    -------
    m : array_like
        3x3 array containing the rotation matrix for a given time.
    """
    # Reference: https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    return np.array(
        [
            [-np.sin(ha), np.cos(ha), 0.0 * ha],
            [-np.sin(lat) * np.cos(ha), -np.sin(lat) * np.sin(ha), np.cos(lat)],
            [np.cos(lat) * np.cos(ha), np.cos(lat) * np.sin(ha), np.sin(lat)],
        ]
    )


def enu_to_eci_matrix(ha, lat):
    """3x3 transformation matrix to rotate ENU to ECI coordinates.

    3x3 transformation matrix to project local observer-centric East-North-Up
    (ENU) coordinates at a given time and location to Earth-Centered Inertial
    (ECI) coordinates.

    The ECI coordinates are aligned with the celestial pole, i.e. for (x,y,z)
    (RA=0  deg, Dec=0  deg) = (1, 0, 0)
    (RA=90 deg, Dec=0  deg) = (0, 1, 0)
    (RA=0  deg, Dec=90 deg) = (0, 0, 1)

    Parameters
    ----------
    ha : float
        Hour angle, in radians, where HA = LST - RA.

    lat : float
        Latitude of the observer, in radians.

    Returns
    -------
    m : array_like
        3x3 array containing the rotation matrix for a given time.
    """
    # Reference: https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    return np.array(
        [
            [-np.sin(ha), -np.cos(ha) * np.sin(lat), np.cos(ha) * np.cos(lat)],
            [np.cos(ha), -np.sin(ha) * np.sin(lat), np.sin(ha) * np.cos(lat)],
            [0.0 * ha, np.cos(lat), np.sin(lat)],
        ]
    )


def altaz_to_enu(el, az):
    """Convert alt/az coordinates as given by Astropy, into ENU coordinates.

    Astropy has Az oriented East of North, i.e. Az(N) = 0 deg, Az(E) = +90 deg.
    See: https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    """
    xp = get_array_module(el)

    return xp.array([xp.cos(el) * xp.sin(az), xp.cos(el) * xp.cos(az), xp.sin(el)])


def point_source_crd_eq(ra, dec):
    """Coordinate transform of source locations from equatorial to Cartesian.

    This converts RA and Dec angles to a Cartesian x,y,z unit vector in an
    Earth-Centered Inertial (ECI) coordinate system aligned with the celestial
    pole, i.e. for (x,y,z)
    (RA=0  deg, Dec=0  deg) = (1, 0, 0)
    (RA=90 deg, Dec=0  deg) = (0, 1, 0)
    (RA=0  deg, Dec=90 deg) = (0, 0, 1)

    The RA and Dec are assumed to be in a particular reference frame that may
    not match-up with standard frames like ICRS/J2000. To convert coordinates
    from a standard system into the relevant frame, see
    :func:`~equatorial_to_eci_coords`.

    Parameters
    ----------
    ra, dec : array_like
        1D arrays of source positions in equatorial coordinates (radians).

    Returns
    -------
    array_like
        Equatorial coordinates of sources, in Cartesian system. Shape=(3, NSRCS).
    """
    xp = get_array_module(ra, dec)
    return xp.asarray([xp.cos(ra) * xp.cos(dec), xp.cos(dec) * xp.sin(ra), xp.sin(dec)])


def equatorial_to_eci_coords(ra, dec, obstime, location, unit="rad", frame="icrs"):
    """Convert RA and Dec coordinates into the ECI system used by  matvis.

    Convert RA and Dec coordinates into the ECI (Earth-Centered Inertial)
    system used by  matvis. This ECI system is aligned with the celestial pole,
    not the Earth's axis.

    To ensure that all corrections are properly taken into account, this
    function uses Astropy to find the Alt/Az positions of the coordinates at a
    specified reference time and location. These are then transformed back to
    ENU (East-North-Up) coordinates, and then to Cartesian ECI coordinates,
    using the inverses of the same transforms that are used to do the forward
    transforms when running `` matvis``. The ECI coordinates are then finally
    converted back into adjusted RA and Dec coordinates in the ECI system.

    While the time-dependent corrections are determined for the reference time,
    the adjusted coordinates are expected to yield a good approximation to the
    true Azimuth and Zenith angle of the sources at other times (but the same
    location) when passed into the `` matvis`` function via the ``crd_eq``
    array and then converted using the standard conversions within `` matvis``.

    The following example code shows how to set up the Astropy ``Time`` and
    ``EarthLocation`` objects that are required by this function::

        # HERA location
        location = EarthLocation.from_geodetic(lat=-30.7215,
                                               lon=21.4283,
                                               height=1073.)
        # Observation time
        obstime = Time('2018-08-31T04:02:30.11', format='isot', scale='utc')


    Parameters
    ----------
    ra, dec : array_like
        Input RA and Dec positions. The units and reference frame of these
        positions can be set using the ``unit`` and ``frame`` kwargs.

    obstime : astropy.Time object
        ``Time`` object specifying the time of the reference observation.

    location : astropy.EarthLocation
        ``EarthLocation`` object specifying the location of the reference
        observation.

    unit : str, optional
        Which units the input RA and Dec values are in, using names intelligible
        to ``astropy.SkyCoord``. Default: 'rad'.

    frame : str, optional
        Which frame that input RA and Dec positions are specified in. Any
        system recognized by ``astropy.SkyCoord`` can be used. Default: 'icrs'.

    Returns
    -------
    eci_ra, eci_dec : array_like
        Arrays of RA and Dec coordinates with respect to the ECI system used
        by  matvis.
    """
    if not isinstance(obstime, Time):
        raise TypeError("obstime must be an astropy.Time object")
    if not isinstance(location, EarthLocation):
        raise TypeError("location must be an astropy.EarthLocation object")

    # Local sidereal time at this obstime and location
    lst = obstime.sidereal_time("apparent", longitude=location.lon).rad

    # Create Astropy SkyCoord object
    skycoords = SkyCoord(ra, dec, unit=unit, frame=frame)

    # Rotation matrix from Cartesian ENU to Cartesian ECI
    m = enu_to_eci_matrix(lst, location.lat.rad)  # Evaluated at HA (= LST - RA) = LST

    # Get AltAz and ENU coords of sources at reference time and location. Ref:
    # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    alt_az = skycoords.transform_to(AltAz(obstime=obstime, location=location))
    astropy_enu = altaz_to_enu(alt_az.alt.rad, alt_az.az.rad)
    # Convert to ECI coordinates using ENU->ECI transform
    astropy_eci = np.dot(m, astropy_enu)

    # Infer RA and Dec coords from astropy ECI
    px, py, pz = astropy_eci
    pdec = np.arcsin(pz)
    pra = np.arctan2(py, px)
    return pra, pdec


def calc_coherency_rotation(ra, dec, alt, az, time, location):
    """
    Compute the rotation matrix needed for time-dependent coherency calculation.

    This function computes the rotation matrix needed to rotate a source's coherency
    from equatorial (RA/Dec) frame into the local alt/az frame. Adopted
    from the pyradiosky coherency calculation, but modified for better vectorization.
    Coordinate rotations should be performed before running this function.

    Parameters
    ----------
    alt : array_like
        Altitude angle(s) of source(s), in radians.
    az : array_like
        Azimuth angle(s) of source(s), in radians. Same shape as `alt`.
    time : astropy.time.Time
        Observation time, used to compute the ICRS→altaz rotation.

    Returns
    -------
    coherency_rot_matrix : array_like
        2x2 rotation matrix that carries the coherency from the
        equatorial frame to the alt/az frame. Shape=(2, 2, N), where
        N is the number of sources.
    """
    xp = get_array_module(ra, dec, alt, az)

    # compute the bulk rotation from equatorial→altaz
    basis_rotation_matrix = _calc_rotation_matrix(
        ra=ra, dec=dec, alt=alt, az=az, location=location, time=time
    )

    # compute the rotation matrix that carries the coherency from
    # equatorial→altaz
    coherency_rot_matrix = spherical_basis_vector_rotation_matrix(
        theta=xp.pi / 2.0 - dec,
        phi=ra,
        rotation_matrix=basis_rotation_matrix,
        beta=xp.pi / 2.0 - alt,
        alpha=az,
    )
    return coherency_rot_matrix


def _calc_rotation_matrix(ra, dec, alt, az, time, location):
    """
    Build the full 3×3 rotation matrix between (RA, Dec) and (alt, az).

    This function builds the rotation matrix that carries unit vectors
    at (RA, Dec) in ICRS into unit vectors at (alt,az) in the local altaz frame.
    Adopted from pyradiosky.

    Parameters
    ----------
    ra, dec : array_like
        Equatorial coordinates of the source(s) in radians.
    alt, az : array_like
        Local horizontal coordinates of the same source(s).
    time : astropy.time.Time
        Observation time for Earth rotation.
    location : astropy.coordinates.EarthLocation
        Observatory location.

    Returns
    -------
    R_exact : ndarray
        3×3[×N] rotation matrix(s) from ICRS→altaz.

    Steps
    -----
    1. Compute the point-source unit vector in ICRS (frame_vec).
    2. Compute the target unit vector in altaz (altaz_vec).
    3. Get the average ICRS→altaz rotation at this time/location (R_avg).
    4. Compute a small perturbation rotation R_perturb so that R_perturb · frame_vec = altaz_vec.
    5. Compose R_exact = R_perturb ⋅ R_avg.
    """
    xp = get_array_module(ra, dec, alt, az)

    # 1) vector in ICRS from (ra, dec)
    frame_vec = point_source_crd_eq(ra, dec)

    # 2) vector in ICRS from (az, alt) — same routine, swapping args
    altaz_vec = point_source_crd_eq(az, alt)

    # 3) base rotation from ICRS → altaz at this time/location
    R_avg = _calc_average_rotation_matrix(time, location)
    R_avg = xp.asarray(R_avg)  # ensure the array matches either numpy or cupy

    # apply R_avg to the equatorial vector
    intermediate_vec = xp.matmul(R_avg, frame_vec)

    # 4) find the rotation that carries intermediate_vec → altaz_vec
    R_perturb = vecs2rot(r1=intermediate_vec, r2=altaz_vec)

    # 5) full exact rotation
    R_exact = xp.einsum("ab...,bc->ac...", R_perturb, R_avg)
    return R_exact


def vecs2rot(r1, r2):
    """
    Construct an axis-angle rotation matrix R that carries vector r1 to r2.

    This function has been adopted from pyradiosky.

    Parameters
    ----------
    r1, r2 : ndarray, shape (3, N)
        Sets of 3D vectors.

    Returns
    -------
    R : ndarray, shape (3, 3, N)
        Rotation matrices such that R[..., i] @ r1[:, i] = r2[:, i].
    """
    xp = get_array_module(r1, r2)

    # axis of rotation ∝ cross product
    norm = xp.cross(r1, r2, axis=0)
    sinPsi = xp.linalg.norm(norm, axis=0)
    n_hat = norm / sinPsi  # unit rotation axis
    cosPsi = xp.sum(r1 * r2, axis=0)
    Psi = xp.arctan2(sinPsi, cosPsi)
    return axis_angle_rotation_matrix(n_hat, Psi)


def axis_angle_rotation_matrix(axis, angle):
    """
    Build rotation matrix via Rodrigues’ formula.

    Parameters
    ----------
    axis : ndarray, shape (3, N)
        Unit rotation axis for each of N rotations.
    angle : array_like, shape (N,)
        Rotation angles in radians.

    Returns
    -------
    rot_matrix : ndarray, shape (3, 3, N)
        The rotation matrices R such that R[..., i] rotates by angle[i]
        about axis[:, i].
    """
    xp = get_array_module(axis, angle)

    # skew-symmetric K-matrix for each axis
    # K_{ab} = ε_{abc} axis_c
    nsrc = axis.shape[1]
    K_matrix = xp.array(
        [
            [xp.zeros(nsrc), -axis[2], axis[1]],
            [axis[2], xp.zeros(nsrc), -axis[0]],
            [-axis[1], axis[0], xp.zeros(nsrc)],
        ]
    )

    I_matrix = xp.eye(3)
    # K^2 term
    K2 = xp.einsum("ab...,bc...->ac...", K_matrix, K_matrix)

    # Rodrigues: R = I + sin(angle) K + (1−cos(angle)) K^2
    rot_matrix = (
        I_matrix[..., None] + xp.sin(angle) * K_matrix + (1.0 - xp.cos(angle)) * K2
    )
    return rot_matrix


def spherical_basis_vector_rotation_matrix(theta, phi, rotation_matrix, beta, alpha):
    """
    Get the rotation matrix for vectors in theta/phi basis to a new reference frame.

    Parameters
    ----------
    theta, phi : array_like
        Colatitude and longitude in the original frame.
    rotation_matrix : ndarray, shape (3,3,...)
        Bulk 3D rotation carrying original frame → new frame.
    beta, alpha : array_like
        Colatitude and longitude in the new (rotated) frame.

    Returns
    -------
    bmat : ndarray, shape (2,2,...)
        2×2 rotation matrices in the spherical basis:
            [[ cos X,  sin X],
            [-sin X,  cos X]],

        where X is the angle between the two basis sets.
    """
    xp = get_array_module(theta, phi)

    # unit vectors in original frame
    th = theta_hat(theta, phi)
    ph = phi_hat(theta, phi)

    # rotate the new-frame theta hat into original-frame coordinates
    bh = xp.einsum("ba...,b...->a...", rotation_matrix, theta_hat(beta, alpha))

    # project rotated theta hat onto original theta hat and phi hat to get the mixing angle
    cosX = xp.einsum("a...,a...->...", bh, th)
    sinX = xp.einsum("a...,a...->...", bh, ph)

    return xp.array([[cosX, sinX], [-sinX, cosX]])


def _calc_average_rotation_matrix(time, telescope_location):
    """
    Compute the rigid-body rotation from ICRS (x,y,z) axes.

    Parameters
    ----------
    time : astropy.time.Time
        Observation time for Earth orientation.
    telescope_location : astropy.coordinates.EarthLocation
        Observatory geodetic location.

    Returns
    -------
    R_really_orthogonal : ndarray, shape (3,3)
        Orthogonal rotation matrix approximating ICRS→altaz.
    """
    # define unit ICRS axes
    x_c = np.array([1.0, 0.0, 0.0])
    y_c = np.array([0.0, 1.0, 0.0])
    z_c = np.array([0.0, 0.0, 1.0])

    # make a SkyCoord with cartesian representation
    axes_icrs = SkyCoord(
        x=x_c,
        y=y_c,
        z=z_c,
        obstime=time,
        location=telescope_location,
        frame="icrs",
        representation_type="cartesian",
    )
    # transform to altaz frame
    axes_altaz = axes_icrs.transform_to("altaz")
    axes_altaz.representation_type = "cartesian"

    # extract the 3×3 matrix (may not be perfectly orthogonal)
    R_screwy = axes_altaz.cartesian.xyz

    # force strict orthogonality via Procrustes to the identity
    R_really_orthogonal, _ = ortho_procr(R_screwy, np.eye(3))
    # transpose to match conventional multiplication order
    return R_really_orthogonal.T


def theta_hat(theta, phi):
    """
    Return the unit vector theta_hat in Cartesian coords for spherical angles.

    Parameters
    ----------
    theta : array_like
        Colatitude angle(s), in radians.
    phi : array_like
        Longitude angle(s), in radians.

    Returns
    -------
    vec : ndarray, shape (3, ...)
        Cartesian components of theta_hat.
    """
    xp = get_array_module(theta, phi)
    return xp.stack(
        [xp.cos(phi) * xp.cos(theta), xp.sin(phi) * xp.cos(theta), -xp.sin(theta)]
    )


def phi_hat(theta, phi):
    """
    Return the unit vector phi_hat in Cartesian coords for spherical angles.

    Parameters
    ----------
    theta : array_like
        Colatitude angle(s), in radians.
    phi : array_like
        Longitude angle(s), in radians.

    Returns
    -------
    vec : ndarray, shape (3, ...)
        Cartesian components of phi_hat.
    """
    xp = get_array_module(theta, phi)
    return xp.stack([-xp.sin(phi), xp.cos(phi), xp.zeros_like(phi)])
