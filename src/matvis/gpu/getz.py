"""Fused GPU computation of the Z matrix."""

from pathlib import Path

import cupy as cp
import numpy as np

from ..core.getz import ZMatrixCalc

KERNELS_PATH = Path(__file__).parent / "kernels"

# Z[ant, feed, ax, src] = A[beam_idx[ant], feed, ax, src] * exptau[ant, src] * sqrtI[src]
# See kernels/fused_z.cu for the kernel source.
_FUSED_Z_MODULE = cp.RawModule(code=(KERNELS_PATH / "fused_z.cu").read_text())


class GPUZMatrixCalc(ZMatrixCalc):
    """Compute the Z matrix on the GPU in a single fused kernel."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("gpu", True)
        super().__init__(*args, **kwargs)
        self._beam_idx_gpu = None

    def __call__(
        self,
        sqrt_flux: cp.ndarray,
        beam: cp.ndarray,
        exptau: cp.ndarray,
        beam_idx: np.ndarray | None,
    ) -> cp.ndarray:
        """Compute Z = A * sqrtI * exp(tau) in one pass.

        See :meth:`matvis.core.getz.ZMatrixCalc.__call__` for parameters.
        Unlike the base implementation, ``exptau`` is not modified in place.
        """
        if beam_idx is None:
            bidx = np.uint64(0)  # NULL pointer
            # A single beam is shared by all antennas; otherwise one per ant.
            bmul = np.int64(0 if beam.shape[0] == 1 else 1)
        else:
            if self._beam_idx_gpu is None:
                self._beam_idx_gpu = cp.asarray(beam_idx, dtype=np.int64)
            bidx = self._beam_idx_gpu
            bmul = np.int64(1)  # unused when beam_idx is given

        kern = _FUSED_Z_MODULE.get_function(
            "fused_z_c64" if self.ctype == np.complex64 else "fused_z_c128"
        )

        ntot = self.nant * self.nfeed * self.nax * self.nsrc
        # 256 is a conventional warp-multiple default, not empirically
        # tuned for this kernel/shape.
        block = 256
        rdtype = np.float32 if self.ctype == np.complex64 else np.float64
        sqrt_flux = cp.ascontiguousarray(sqrt_flux, dtype=rdtype)
        assert beam._c_contiguous and exptau._c_contiguous

        kern(
            ((ntot + block - 1) // block,),
            (block,),
            (
                beam,
                exptau,
                sqrt_flux,
                bidx,
                bmul,
                np.int32(self.nfeed),
                np.int32(self.nax),
                np.int64(self.nsrc),
                np.int64(ntot),
                self.z,
            ),
        )

        return self.z.reshape(self.nant * self.nfeed, self.nax * self.nsrc)
