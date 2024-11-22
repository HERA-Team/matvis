"""CPU-based beam evaluation."""

import numpy as np
from pyuvdata import UVBeam

from ..coordinates import enu_to_az_za
from ..core.beams import BeamInterpolator


class UVBeamInterpolator(BeamInterpolator):
    """Interpolate a UVBeam object."""

    def interp(self, tx: np.ndarray, ty: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Evaluate the beam on the CPU.

        This function will either interpolate the beam to the given coordinates tx, ty,
        or evaluate the beam there if it is an analytic beam.

        Parameters
        ----------
        tx, ty
            Coordinates to evaluate the beam at, in sin-projection.
        """
        # Primary beam pattern using direct interpolation of UVBeam object
        az, za = enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")

        kw = {
            "reuse_spline": True,
            "check_azza_domain": False,
            "interpolation_function": "az_za_map_coordinates",
            "spline_opts": self.spline_opts,
        }
        for i, bm in enumerate(self.beam_list):
            interp_beam = bm.compute_response(
                az_array=az,
                za_array=za,
                freq_array=np.array([self.freq]),
                **kw,
            )

            if self.polarized:
                interp_beam = interp_beam[:, :, 0, :].transpose((1, 0, 2))
            else:
                # Here we have already asserted that the beam is a power beam and
                # has only one polarization, so we just evaluate that one.
                interp_beam = np.sqrt(interp_beam[0, 0, 0, :])

            out[i] = interp_beam
