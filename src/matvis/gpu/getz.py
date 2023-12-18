"""GPU implementation for obtaining the Z matrix."""
import cupy as cp

from ..core.getz import ZMatrixCalc


class GPUZMatrixCalc(ZMatrixCalc):
    """GPU implementation for obtaining the Z matrix."""

    def setup(self):
        """Perform any necessary setup steps.

        Accepts no inputs and returns nothing.
        """
        self.z = cp.zeros(
            (self.nfeed * self.nant, self.nax * self.nsrc), dtype=self.ctype
        )

    def compute(
        self,
        sqrt_flux: cp.ndarray,
        beam: cp.ndarray,
        exptau: cp.ndarray,
        beam_idx: cp.ndarray,
    ) -> cp.ndarray:
        """Placeholder."""
        super().compute(sqrt_flux, beam, exptau, beam_idx)
        cp.cuda.Device().synchronize()

    compute.__doc__ = ZMatrixCalc.compute.__doc__
