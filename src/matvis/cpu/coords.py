"""CPU-based coordinate rotations."""
import numpy as np

from ..core.coords import CoordinateRotation


class CPUCoordinateRotation(CoordinateRotation):
    """CPU-based coordinate rotation."""

    def rotate(self) -> np.ndarray:
        """Rotate the given coordinates with the given 3x3 rotation matrix.

        Returns
        -------
        np.ndarray
            Rotated coordinates. Shape=(3, Nsrcs_above_horizon).
        np.ndarray
            Flux. Shape=(Nsrcs_above_horizon,).
        """
        self.eq2top_t.dot(self.coords_eq_chunk, out=self.coords_topo)
        above_horizon = self.coords_topo[2] > 0
        coords_topo = self.coords_topo[:, above_horizon]
        flux = self.flux[above_horizon]

        return coords_topo, flux
