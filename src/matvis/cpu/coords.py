"""CPU-based coordinate rotations."""
import numpy as np

from ..core.coords import CoordinateRotation


class CPUCoordinateRotation(CoordinateRotation):
    """CPU-based coordinate rotation."""

    def rotate(self, t: int) -> np.ndarray:
        """Rotate the given coordinates with the given 3x3 rotation matrix.

        Returns
        -------
        np.ndarray
            Rotated coordinates. Shape=(3, Nsrcs_above_horizon).
        np.ndarray
            Flux. Shape=(Nsrcs_above_horizon,).
        """
        self.eq2top[t].dot(self.coords_eq, out=self.all_coords_topo)
