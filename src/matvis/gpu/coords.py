"""Core abstract class for coordinate rotation."""
import cupy as cp
import numpy as np

from ..core.coords import CoordinateRotation


class GPUCoordinateRotation(CoordinateRotation):
    """GPU-based coordinate rotation."""

    def setup(self):
        """Allocate GPU memory for the rotation."""
        self.eq2top_t = cp.empty((3, 3), dtype=self.rtype, order="F")
        self.coords_topo = cp.empty((3, self.chunk_size), dtype=self.rtype, order="F")
        self.coords_eq_chunk = cp.empty(
            (3, self.chunk_size), dtype=self.rtype, order="F"
        )
        self.flux = cp.asarray(self.flux)

    def set_rotation_matrix(self, t: int):
        """Set the rotation matrix for the given time index."""
        self.eq2top_t[:].set(self.eq2top[t])

    def set_chunk(self, chunk: int):
        """Set the chunk of coordinates to rotate."""
        slc = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)
        self.coords_eq_chunk[:].set(self.coords_eq[:, slc])
        self.flux_chunk = self.flux[slc]  # hopefully it's a view

    def rotate(self) -> np.ndarray:
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
        cp.matmul(self.eq2top_t, self.coords_eq_chunk, out=self.coords_topo)
        above_horizon = cp.where(self.coords_topo[2] > 0)[0]
        crdtop_lim = self.coords_topo[:, above_horizon]
        flux = self.flux_chunk[above_horizon]

        cp.cuda.Device().synchronize()
        return crdtop_lim, flux
