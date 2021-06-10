"""Functions for converting coordinates."""

import numpy as np


def lm_to_az_za(el, m):
    """Convert l and m (on intervals -1, +1) to azimuth and zenith angle.

    Parameters
    ----------
    l, m : array_like
        Normalized angular coordinates on the interval (-1, +1).

    Returns
    -------
    az, za : array_like
        Corresponding azimuth and zenith angles (in radians).
    """
    lsqr = el ** 2.0 + m ** 2.0
    n = np.where(lsqr < 1.0, np.sqrt(1.0 - lsqr), 0.0)

    az = -np.arctan2(m, el)
    za = np.pi / 2.0 - np.arcsin(n)
    return az, za


def point_source_crd_eq(ra, dec):
    """Coordinate transform of source locations from equatorial to Cartesian.

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


def uvbeam_to_lm(uvbeam, freqs, n_pix_lm=63, polarized=False, **kwargs):
    """Convert a UVbeam to a uniform (l,m) grid.

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

    Returns
    -------
    ndarray
        The beam map cube. Shape: (NFREQS, BEAM_PIX, BEAM_PIX) if
        `polarized=False` or (NAXES, NFEEDS, NFREQS, BEAM_PIX, BEAM_PIX) if
        `polarized=True`.
    """
    # Define angle cosines
    L = np.linspace(-1, 1, n_pix_lm, dtype=np.float32)
    L, m = np.meshgrid(L, L)
    L = L.flatten()
    m = m.flatten()

    # Apply horizon cut
    lsqr = L ** 2.0 + m ** 2.0
    n = np.where(lsqr < 1.0, np.sqrt(1.0 - lsqr), 0.0)

    # Calculate azimuth and zenith angle
    az = -np.arctan2(m, L)
    za = np.pi / 2.0 - np.arcsin(n)

    # Interpolate beam onto cube
    efield_beam = uvbeam.interp(az, za, freqs, **kwargs)[0]
    if polarized:
        bm = efield_beam[:, 0, :, :, :]  # spw=0
    else:
        bm = efield_beam[0, 0, 1, :, :]  # (phi, e) == 'xx' component

    # Peak normalization and reshape output
    if polarized:
        Naxes = bm.shape[0]  # polarization vector axes
        Nfeeds = bm.shape[1]  # polarized feeds

        # Separately normalize each polarization channel
        for i in range(Naxes):
            for j in range(Nfeeds):
                if np.max(bm[i, j]) > 0.0:
                    bm /= np.max(bm[i, j])
        return bm.reshape((Naxes, Nfeeds, len(freqs), n_pix_lm, n_pix_lm))
    else:
        # Normalize single polarization channel
        if np.max(bm) > 0.0:
            bm /= np.max(bm)
        return bm.reshape((len(freqs), n_pix_lm, n_pix_lm))


def eq2top_m(ha, dec):
    """Calculate the equatorial to topocentric conversion matrix.

    Conversion at a given hour angle (ha) and declination (dec). Ripped
    straight from aipy.

    Parameters
    ----------
    ha : float
        Hour angle [rad].
    dec : float
        Declination [rad].

    Returns
    -------
    ndarray
        Coordinate transform matrix converting equatorial coordinates to
        topocentric coordinates. Shape=(3, 3).
    """
    sin_H, cos_H = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    zero = np.zeros_like(ha)

    m = np.array(
        [
            [sin_H, cos_H, zero],
            [-sin_d * cos_H, sin_d * sin_H, cos_d],
            [cos_d * cos_H, -cos_d * sin_H, sin_d],
        ]
    )

    if len(m.shape) == 3:
        m = m.transpose([2, 0, 1])

    return m


def get_eq2tops(lsts, latitude):
    """
    Calculate transformations from equatorial to topocentric coords.

    Parameters
    ----------
    lsts : array_like
        Local sidereal time values, in radians.

    latitiude : float
        Latitude of the array, in radians.

    Returns
    -------
    array_like of self._real_dtype
        The set of 3x3 transformation matrices converting equatorial
        to topocenteric co-ordinates at each LST.
        Shape=(NTIMES, 3, 3).
    """
    return np.array(
        [eq2top_m(-sid_time, latitude) for sid_time in lsts], dtype=lsts.dtype
    )
