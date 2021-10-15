"""Functions for converting coordinates."""

import astropy.units as u
import numpy as np
import pyuvdata.utils as uvutils
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.coordinates.builtin_frames import AltAz
from astropy.time import Time
from copy import deepcopy
from numpy import typing as npt
from pyuvdata.uvbeam import UVBeam

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


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
    assert orientation in [
        "astropy",
        "uvbeam",
    ], "orientation must be either 'astropy' or 'uvbeam'"

    lsqr = enu_n ** 2.0 + enu_e ** 2.0
    mask = lsqr < 1
    zeta = np.zeros_like(lsqr)
    zeta[mask] = np.sqrt(1 - lsqr[mask])

    az = np.arctan2(enu_e, enu_n)
    za = 0.5 * np.pi - np.arcsin(zeta)

    # Flip and rotate azimuth coordinate if uvbeam convention is used
    if orientation == "uvbeam":
        az = 0.5 * np.pi - az
    if periodic_azimuth:
        az = np.mod(az, 2 * np.pi)
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
        Equatorial coordinates of sources, in Cartesian
        system. Shape=(3, NSRCS).
    """
    return np.asarray([np.cos(ra) * np.cos(dec), np.cos(dec) * np.sin(ra), np.sin(dec)])


def equatorial_to_eci_coords(ra, dec, obstime, location, unit="rad", frame="icrs"):
    """Convert RA and Dec coordinates into the ECI system used by vis_cpu.

    Convert RA and Dec coordinates into the ECI (Earth-Centered Inertial)
    system used by vis_cpu. This ECI system is aligned with the celestial pole,
    not the Earth's axis.

    To ensure that all corrections are properly taken into account, this
    function uses Astropy to find the Alt/Az positions of the coordinates at a
    specified reference time and location. These are then transformed back to
    ENU (East-North-Up) coordinates, and then to Cartesian ECI coordinates,
    using the inverses of the same transforms that are used to do the forward
    transforms when running ``vis_cpu``. The ECI coordinates are then finally
    converted back into adjusted RA and Dec coordinates in the ECI system.

    While the time-dependent corrections are determined for the reference time,
    the adjusted coordinates are expected to yield a good approximation to the
    true Azimuth and Zenith angle of the sources at other times (but the same
    location) when passed into the ``vis_cpu`` function via the ``crd_eq``
    array and then converted using the standard conversions within ``vis_cpu``.

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
        by vis_cpu.
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
    el, az = alt_az.alt.rad, alt_az.az.rad
    astropy_enu = np.array(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)]
    )
    # astropy has Az oriented East of North, i.e. Az(N) = 0 deg, Az(E) = +90 deg

    # Convert to ECI coordinates using ENU->ECI transform
    astropy_eci = np.dot(m, astropy_enu)

    # Infer RA and Dec coords from astropy ECI
    px, py, pz = astropy_eci
    pdec = np.arcsin(pz)
    pra = np.arctan2(py, px)
    return pra, pdec


def prepare_beam(
    uvbeam: UVBeam, polarized: bool = False, use_feed: Literal["x", "y"] = "x"
) -> UVBeam:
    """Prepare an imput beam for either interpolation or simulation.

    The point of this function is to take an arbitrary UVBeam (or AnalyticBeam) and
    do the necessary checks and conversions to convert it to a format that can be
    interpolated to an (l,m) grid, or passed to vis_cpu. The output beam type is
    dependent on the input parameters ``polarized`` and ``use_feed``.
    """
    use_feed = use_feed.lower()
    if use_feed not in "xy":
        raise ValueError("use_feed must be either 'x' or 'y'")
    use_pol = use_feed * 2

    # Interpolate beam onto cube
    if polarized:
        if uvbeam.beam_type != "efield":
            raise ValueError("Beam type must be efield")
        uvbeam_ = uvbeam
    elif uvbeam.beam_type == "efield":
        uvbeam_ = uvbeam.copy() if isinstance(uvbeam, UVBeam) else deepcopy(uvbeam)
        # Analytic beams have no concept of feeds, so assume they have a "single" feed
        if getattr(uvbeam_, "Nfeeds", 1) > 1:
            uvbeam_.select(feeds=[use_feed])

        if isinstance(uvbeam, UVBeam):
            uvbeam_.efield_to_power(calc_cross_pols=False)
        else:
            uvbeam_.efield_to_power()

    elif getattr(uvbeam, "Npols", 1) > 1:
        pol = uvutils.polstr2num(use_pol)

        if pol not in uvbeam.polarization_array:
            raise ValueError(
                f"You want to use {use_feed} feed, but it does not exist in the UVBeam"
            )

        uvbeam_ = uvbeam.select(polarizations=[pol], inplace=False)
    else:
        uvbeam_ = uvbeam

    return uvbeam_


def uvbeam_to_lm(
    uvbeam: UVBeam,
    freqs: npt.ArrayLike,
    n_pix_lm: int = 63,
    polarized: bool = False,
    use_feed: Literal["x", "y"] = "x",
    **kwargs,
):
    """Evaluate a UVBeam object on a uniform direction cosine (l,m) grid.

    Here, (l, m) are the direction cosines, associated with the East and North
    ENU coordinates (and also the U and V directions for a zenith-pointing
    drift-scan telescope). For a vector in East-North-Up (ENU) coordinates
    vec{p}, we therefore have ``l = vec{p}.hat{e}`` etc.

    N.B. This function does not perform any beam normalization.

    Parameters
    ----------
    uvbeam : UVBeam object
        Beam to convert to an (l, m) grid.
    freqs : array_like
        Frequencies to interpolate to in [Hz]. Shape=(NFREQS,).
    n_pix_lm : int, optional
        Number of pixels for each side of the beam grid.
    polarized : bool, optional
        Whether to return full polarized beam information or not.
    use_feed
        Which feed to use in the case of an *unpolarized* simulation (for a polarized
        sim, it uses all available feeds).

    Notes
    -----
    When ``polarized=True``, all combinations of axes and feeds are returned. In this
    case, the input ``uvbeam`` must have a beam type of ``efield``, or else an error
    is raised. Conversely, if ``polarized=False``, the convention is to return the
    real-valued beam equal to the square root of the power beam of a linear polarization.
    That is, either the sqrt of XX or YY. Needless to say, the desired feed must be
    defined in the beam object. The input beam in this case can be in either 'efield'
    or 'power' mode.

    Returns
    -------
    ndarray
        The beam map cube. Shape: (NFREQS, BEAM_PIX, BEAM_PIX) if
        `polarized=False` or (NAXES, NFEEDS, NFREQS, BEAM_PIX, BEAM_PIX) if
        `polarized=True`.
    """
    # Define angle cosines
    L = np.linspace(-1, 1, n_pix_lm)
    L, m = np.meshgrid(L, L)
    L = L.flatten()
    m = m.flatten()

    uvbeam = prepare_beam(uvbeam, polarized=polarized, use_feed=use_feed)

    # Get azimuth and zenith angles (note the different azimuth convention
    # used by UVBeam)
    az, za = enu_to_az_za(enu_e=L, enu_n=m, orientation="uvbeam", periodic_azimuth=True)

    # Interpolate beam onto cube
    if polarized:
        efield_beam = uvbeam.interp(
            az_array=az, za_array=za, freq_array=freqs, **kwargs
        )[0]
        bm = efield_beam[:, 0, :, :, :]  # spw=0
    else:
        power_beam = uvbeam.interp(
            az_array=az, za_array=za, freq_array=freqs, **kwargs
        )[0]
        bm = np.sqrt(power_beam[0, 0, 0, :, :])

    # Reshape output
    if polarized:
        Naxes = bm.shape[0]  # polarization vector axes
        Nfeeds = bm.shape[1]  # polarized feeds
        return bm.reshape((Naxes, Nfeeds, len(freqs), n_pix_lm, n_pix_lm))
    else:
        return bm.reshape((len(freqs), n_pix_lm, n_pix_lm))
