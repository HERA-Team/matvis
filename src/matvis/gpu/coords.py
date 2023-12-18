"""Core abstract class for coordinate rotation."""
import cupy as cp
import numpy as np

from ..core.coords import CoordinateRotation


class GPUCoordinateRotation(CoordinateRotation):
    """GPU-based coordinate rotation."""

    def setup(self):
        """Allocate GPU memory for the rotation."""
        self.eq2top = cp.asarray(self.eq2top)
        self.coords_eq = cp.asarray(self.coords_eq)
        self.flux = cp.asarray(self.flux)
        self.all_coords_topo = cp.empty((3, self.nsrc), dtype=self.rtype)
        self.coords_above_horizon = cp.empty((3, self.nsrc_alloc), dtype=self.rtype)
        self.flux_above_horizon = cp.empty((self.nsrc_alloc,), dtype=self.rtype)
        self.xp = cp

    def rotate(self, t: int) -> np.ndarray:
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
        cp.matmul(self.eq2top[t], self.coords_eq, out=self.all_coords_topo)
        cp.cuda.Device().synchronize()
