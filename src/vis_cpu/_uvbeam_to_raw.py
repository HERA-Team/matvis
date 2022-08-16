"""This module contains the functions for converting UVBeam objects to raw data.

This is useful when you want to perform beam interpolation in some other way than
simply with `beam.interp()`. Since this is purely for speed (obviously not for
convenience), this module makes many assumptions and restrictions on the UVBeam objects
that it can handle.
"""
from __future__ import annotations

import numpy as np
import warnings
from pyuvdata.uvbeam import UVBeam
from typing import Tuple


def uvbeam_to_azza_grid(
    uvbeam: UVBeam, naz: int | None = None, dza: float | None = None, **interp_kwargs
) -> tuple[np.ndarray, float, float]:
    """Extract raw data from the UVBeam object and prime it for interpolation.

    Parameters
    ----------
    uvbeam
        The UVBeam object containing the raw beam data.
    naz
        The desired number of pierce points along the azimuth axis to return.
    dza
        The grid size along the zenith angle axis to return.
    interp_kwargs
        Extra keyword arguments to be passed to the interpolation function.

    Returns
    -------
    beam_data
        The 2D array of beam data on the az/za grid.
    daz
        The grid size along the azimuth direction. The grid always has its first point
        at zero and its last point at 2pi, which means this value must be an integer
        divisor of 2pi.
    daz
        The grid size along the zenith angle direction. The grid always has its first
        point at zero, and its last point is equal to or greater than pi/2 (i.e the
        horizon).

    Notes
    -----
    For this function to work, the UVBeam object must be in the az_za coordinate system
    to begin with. This is because the GPU interpolation functions work only on a
    rectilinear grid of az/za.

    If the raw data in the beam is itself already on a regular az/za grid for which the
    az values go exactly from 0 to 2pi and the za values go from 0 to >=pi/2, then by
    default they are just returned as-is. Otherwise, the intrinsic ``.interp()`` method
    of the UVBeam object is used to interpolate it to a regular grid. Doing the latter
    is "less good" in the sense that it means that two overall spatial interpolations
    will be done when computing visibilities. Since they will both be in the same basis
    set, this should not be a problem (definitely so if the initial interpolation is
    linear).
    """
    if uvbeam.pixel_coordinate_system != "az_za":
        raise ValueError('pixel_coordinate_system must be "az_za"')

    if uvbeam.Nfreqs != 1:
        raise ValueError(
            "Can only handle one frequency -- interpolate over frequency first!"
        )

    az, za = uvbeam.axis1_array, uvbeam.axis2_array

    delta_az = np.diff(az)
    delta_za = np.diff(za)

    if naz is None:
        naz = len(az)

    is_regular_grid = np.allclose(delta_az, delta_az[0]) and np.allclose(
        delta_za, delta_za[0]
    )

    if is_regular_grid and dza is None:
        dza = delta_za[0]
    elif not is_regular_grid:
        raise ValueError(
            "Input UVBeam is not regular, so you must supply your desired dza. "
            f"az diffs between ({delta_az.min(), delta_az.max()}). "
            f"za diffs between ({delta_za.min(), delta_za.max()}). "
        )

    delta_az = delta_az[0]
    delta_za = delta_za[0]

    # covers_sky_strong is the best-case scenario, where the beam data covers the
    # upper hemisphere and is defined directly from az=0-2pi and za=0-pi/2. If we have
    # this *and* a regular grid, we don't have to do any interpolation of the data here,
    # since it's already exactly in the format required by the GPU interpolation function.
    covers_sky_strong = (
        np.isclose(az[0], 0)
        and np.isclose(az[-1], 2 * np.pi)
        and np.isclose(za[0], 0)
        and az[-1] >= np.pi / 2
        and np.allclose(uvbeam.data_array[..., 0], uvbeam.data_array[..., -1])
    )

    # In this case, we have almost the strong case, but the final az value is one delta
    # away from 2pi. Here, we can simply copy over the zero-value to the end of the
    # data, and don't need to actually interpolate.
    covers_sky_almost_strong = (
        np.isclose(az[0], 0)
        and np.isclose(az[-1], 2 * np.pi - delta_az)
        and np.isclose(za[0], 0)
        and az[-1] >= np.pi / 2
    )

    # "covers_sky_weak" corresponds to the case where the azimuth values cover the full
    # 2pi range (but where the final value may be Delta(az) smaller than 2pi), because
    # the actual 2pi value is known to be the zero value. We allow for a small epsilon
    # of 1e-5 to account for floating point errors. Also, the zenith angle needs to start
    # at zero and end over the horizon.
    covers_sky_weak = (
        (az.max() - az.min()) >= (2 * np.pi - delta_az - 1e-5)
        and np.isclose(za.min(), 0)
        and za.max() >= np.pi / 2
    )

    if not covers_sky_weak:
        raise ValueError(
            "The beam data does not cover the full sky. Cannot use in vis_cpu."
        )

    # Simplest Case: everything is already in the regular format we need.
    if (
        naz == len(az)
        and np.isclose(dza, delta_za)
        and is_regular_grid
        and covers_sky_strong
    ):
        # Returned data has shape (Nax, Nfeeds, Nza, Naz)
        return uvbeam.data_array[:, 0, :, 0], delta_az, dza
    elif (
        naz == len(az)
        and np.isclose(dza, delta_za)
        and is_regular_grid
        and covers_sky_almost_strong
    ):
        data = uvbeam.data_array[:, 0, :, 0]
        data = np.concatenate((data, data[..., [0]]), axis=-1)
        return data, delta_az, dza
    else:
        warnings.warn(
            "The raw beam data is either irregular, or does not have the spacing you "
            "desire. This means we need to interpolate to a grid, from which a second "
            "round of interpolation will be performed in the visibility calculation."
            "You might be able to avoid this by not specifying a desired naz and dza."
        )

        # We have to treat az and za differently. For az, we need to start at 0 and end at 2pi exactly.
        # This is because it wraps, and our interpolation function needs to be regular
        # over the wrap itself. On the other hand, za just needs to extend at least to
        # the horizon, and we can't know for certain with the input beam goes to the
        # antipode or the horizon before we run the function. Therefore, providing nza
        # is less precise -- we don't know what the eventual dza would be.
        new_az = np.linspace(0, 2 * np.pi, naz)
        new_za = np.arange(0, np.pi / 2 + dza, dza)

        out = uvbeam.interp(
            new_az, new_za, az_za_grid=True, check_azza_domain=False, **interp_kwargs
        )[0]
        out = out.reshape(out.shape[:-1] + (len(new_za), naz))

        # Returned data has shape (Nax, Nfeeds, Nza, Naz)
        if uvbeam.beam_type == "efield":
            return out[:, 0, :, 0], new_az[1] - new_az[0], dza
        else:
            # For a power beam, we just want to the I part of the XX pol.
            # Also, need the sqrt of the beam (to make it quasi X)
            out = out[0, 0, 0, 0]

            # But we need to return it with the nax and nfeed dimensions (both have size 1)
            return out[np.newaxis, np.newaxis], new_az[1] - new_az[0], dza
