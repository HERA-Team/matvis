"""Core abstract class for coordinate rotation."""

import numpy as np
from abc import ABC, abstractmethod

from .._utils import get_dtypes


class CoordinateRotation(ABC):
    """
    Abstract class for coordinate rotation.

    Parameters
    ----------
    flux
        Flux of each source. Shape=(Nsrc,).
    crd_eq
        Equatorial coordinates of each source. Shape=(3, Nsrc).
    eq2top
        Set of 3x3 transformation matrices to rotate the RA and Dec
        cosines in an ECI coordinate system (see `crd_eq`) to
        topocentric coordinates. Shape=(Nt, 3, 3).
    chunk_size
        Number of sources to rotate at a time.
    precision
        The precision of the data (1 or 2).
    """

    def __init__(
        self,
        flux: np.ndarray,
        crd_eq: np.ndarray,
        eq2top: np.ndarray,
        chunk_size: int | None = None,
        precision: int = 1,
    ):
        self.rtype, _ = get_dtypes(precision)
        self.flux = flux.astype(self.rtype)
        self.coords_eq = crd_eq.astype(self.rtype)
        self.eq2top = eq2top.astype(self.rtype)

        self.nsrc = len(flux)
        assert flux.ndim == 1
        self.chunk_size = chunk_size or self.nsrc

    def setup(self):
        """Allocate memory for the rotation."""
        self.coords_topo = np.empty((3, self.chunk_size), dtype=self.rtype)

    def set_rotation_matrix(self, t: int):
        """Set the rotation matrix for the given time index."""
        self.eq2top_t = self.eq2top[t]

    def set_chunk(self, chunk: int):
        """Set the chunk of coordinates to rotate."""
        slc = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)
        self.coords_eq_chunk = self.coords_eq[:, slc]
        self.flux_chunk = self.flux[slc]

    @abstractmethod
    def rotate(self) -> tuple[np.ndarray, np.ndarray]:
        """Rotate the given coordinates with the given 3x3 rotation matrix.

        Parameters
        ----------
        crd
            Coordinates to rotate. Shape=(3, Nsrc).
        rot
            Rotation matrix. Shape=(3, 3).

        Returns
        -------
        np.ndarray
            Rotated coordinates. Shape=(3, Nsrcs_above_horizon).
        np.ndarray
            Flux. Shape=(Nsrcs_above_horizon,).
        """
        pass
