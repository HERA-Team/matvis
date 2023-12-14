"""CPU-based beam evaluation."""
import numpy as np
from pyuvdata import UVBeam

from ..coordinates import enu_to_az_za
from ..core.beams import BeamInterpolator


class UVBeamInterpolator(BeamInterpolator):
    """Interpolate a UVBeam object."""

    def interp(self, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
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
        A_s = np.full(
            (self.nbeam, self.nfeed, self.nax, len(tx)), 0.0, dtype=self.complex_dtype
        )

        for i, bm in enumerate(self.beam_list):
            kw = (
                {
                    "reuse_spline": True,
                    "check_azza_domain": False,
                    "spline_opts": self.spline_opts,
                }
                if isinstance(bm, UVBeam)
                else {}
            )
            if isinstance(bm, UVBeam) and not bm.future_array_shapes:
                bm.use_future_array_shapes()

            interp_beam = bm.interp(
                az_array=az,
                za_array=za,
                freq_array=np.array([self.freq]),
                **kw,
            )[0]

            if self.polarized:
                interp_beam = interp_beam[:, :, 0, :].transpose((1, 0, 2))
            else:
                # Here we have already asserted that the beam is a power beam and
                # has only one polarization, so we just evaluate that one.
                interp_beam = np.sqrt(interp_beam[0, 0, 0, :])

            A_s[i] = interp_beam

        return A_s  # Now (Nbeam, Nfeed, Nax, Nsrc)
