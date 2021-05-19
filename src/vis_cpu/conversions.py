"""Functions for converting co-ordinates."""
import numpy as np


def lm_to_az_za(el, m):
    """
    Convert l and m (on intervals -1, +1) to azimuth and zenith angle.

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
