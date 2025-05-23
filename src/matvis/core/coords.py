"""Core abstract class for coordinate rotation."""

import numpy as np
from abc import ABC, abstractmethod
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from .._utils import get_dtypes
from ..coordinates import calc_coherency_rotation, enu_to_az_za

try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False


class CoordinateRotation(ABC):
    """Abstract class for converting Equatorial (RA/DEC) coordinates to observed.

    Subclasses must, at the very least, implement the ``rotate(t)`` method, which takes
    an integer ``t``, which indexes into the ``times`` array, and sets the
    ``all_coords_topo`` attribute, which are unit-vector topocentric coordinates in the
    horizontal ENU frame of the telescope.

    All defined subclasses of this class can be found in the
    ``CoordinateRotation._methods`` dictionary.
    """

    _methods = {}
    requires_gpu: bool = False

    def __init_subclass__(cls) -> None:
        """Register the subclass."""
        CoordinateRotation._methods[cls.__name__] = cls
        return super().__init_subclass__()

    def __init__(
        self,
        flux: np.ndarray,
        times: Time,
        telescope_loc: EarthLocation,
        skycoords: SkyCoord,
        chunk_size: int | None = None,
        source_buffer: float = 0.55,
        precision: int = 1,
        gpu: bool = False,
    ):
        self.gpu = gpu
        if self.gpu and not HAVE_CUDA:
            raise ValueError("GPU requested but cupy not installed.")

        self.xp = cp if self.gpu else np

        self.precision = precision
        self.rtype, self.ctype = get_dtypes(precision)

        # Check if the flux is complex and set the dtype accordingly.
        if self.xp.iscomplexobj(flux):
            self.sky_model_dtype = self.ctype
        else:
            self.sky_model_dtype = self.rtype

        self.flux = self.xp.asarray(flux.astype(self.sky_model_dtype))
        self._polarized = flux.ndim == 4
        self.nsrc = len(flux)
        self.times = times
        self.telescope_loc = telescope_loc
        self.skycoords = skycoords

        assert times.ndim == 1
        assert len(skycoords) == self.nsrc
        assert len(flux) == self.nsrc

        self.chunk_size = chunk_size or self.nsrc
        self.source_buffer = source_buffer
        if self.chunk_size > 1000:
            self.nsrc_alloc = int(self.chunk_size * self.source_buffer)
        else:
            self.nsrc_alloc = self.chunk_size

    def setup(self):
        """Allocate memory for the rotation."""
        # Initialize arrays that all subclasses must use.
        self.all_coords_topo = self.xp.full(
            (3, self.nsrc), self.rtype(0.0), dtype=self.rtype
        )
        self.coords_above_horizon = self.xp.full(
            (3, self.nsrc_alloc), self.rtype(0.0), dtype=self.rtype
        )
        self.flux_above_horizon = self.xp.full(
            (self.nsrc_alloc,) + self.flux.shape[1:],
            self.sky_model_dtype(0.0),
            dtype=self.sky_model_dtype,
        )

    def select_chunk(self, chunk: int, t: int):
        """Set the chunk of coordinates to rotate."""
        # The last index can be larger than the actual size of the array without error.
        slc = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)

        topo = self.all_coords_topo[:, slc]
        flux = self.flux[slc]

        above_horizon = self.xp.where(topo[2] > 0)[0]
        n = len(above_horizon)
        if n > self.nsrc_alloc:
            raise ValueError(
                f"nsrc_alloc ({self.nsrc_alloc}) is too small for the number of "
                f"sources above horizon ({n}). Try increasing source_buffer."
            )

        if self._polarized:
            # Compute the alt/az coordinates for the sources above the horizon.
            az, za = enu_to_az_za(
                enu_e=topo[0, above_horizon],
                enu_n=topo[1, above_horizon],
                orientation="astropy",
            )

            # For polarized flux, rotate the frame coherency
            self.flux_above_horizon[:n] = self._rotate_frame_coherency(
                coherency_matrix=flux[above_horizon],
                ra=self.skycoords.ra.rad[above_horizon],
                dec=self.skycoords.dec.rad[above_horizon],
                alt=np.pi / 2 - za,
                az=az,
                time=self.times[t],
            )
        else:
            # For unpolarized flux, just copy the flux.
            self.flux_above_horizon[:n] = flux[above_horizon]

        self.coords_above_horizon[:, :n] = topo[:, above_horizon]
        self.flux_above_horizon[n:] = 0

        if self.gpu:
            self.xp.cuda.Device().synchronize()

        return self.coords_above_horizon, self.flux_above_horizon, n

    def _rotate_frame_coherency(self, coherency_matrix, ra, dec, alt, az, time) -> None:
        """
        Rotate the frame of the coherency matrix.

        This function rotates the coherency matrix from the equatorial frame to the
        alt/az frame. It is used in the `rotate` method of subclasses.
        """
        # Calculate the rotation matrix for the current time.
        coherency_rotator = calc_coherency_rotation(
            ra=ra,
            dec=dec,
            alt=alt,
            az=az,
            time=time,
            location=self.telescope_loc,
        )

        # Rotate the coherency matrix. Note that here the coherency rotator is a matrix of
        # size (2, 2, nsources), and coherency matrix is a matrix of size
        # (nsources, nfreq, 2, 2).
        coherency_matrix = self.xp.einsum(
            "abn,nfbc,dcn->nfad", coherency_rotator, coherency_matrix, coherency_rotator
        )
        return coherency_matrix

    @abstractmethod
    def rotate(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Perform the rotation for a single time."""
