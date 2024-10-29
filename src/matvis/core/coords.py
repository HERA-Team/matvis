"""Core abstract class for coordinate rotation."""

import numpy as np
from abc import ABC, abstractmethod
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from .._utils import get_dtypes

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
        self.rtype, _ = get_dtypes(precision)
        self.flux = self.xp.asarray(flux.astype(self.rtype))

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
            (self.nsrc_alloc,) + self.flux.shape[1:], self.rtype(0.0), dtype=self.rtype
        )

    def select_chunk(self, chunk: int):
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

        self.coords_above_horizon[:, :n] = topo[:, above_horizon]
        self.flux_above_horizon[:n] = flux[above_horizon]
        self.flux_above_horizon[n:] = 0

        if self.gpu:
            self.xp.cuda.Device().synchronize()

        return self.coords_above_horizon, self.flux_above_horizon, n

    @abstractmethod
    def rotate(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Perform the rotation for a single time."""
