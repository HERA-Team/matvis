"""GPU-accelerated source-summing operation."""

import cupy as cp
import numpy as np

from ..core.matprod import MatProd
from ._cublas import complex_matmul, zdotz


class GPUMatMul(MatProd):
    """Use cupy.gemm to perform the source-summing operation."""

    def allocate_vis(self):
        """Allocate memory for the visibilities.

        The shape here is (nchunks, nant, nfeed, nant, nfeed), which is backwards
        from what you'd expect (nfeed,nant, nfeed, nant). This is because the
        fortran ordering is used in CUBLAS, which is the same as the transpose of the
        expected shape.
        """
        # The shape is required to be like this to use the fortran ordering
        self.vis = [
            cp.full(
                (self.nfeed, self.nant, self.nfeed, self.nant),
                self.ctype(0.0),
                dtype=self.ctype,
                order="F",
            )
            for _ in range(self.nchunks)
        ]

    def compute(self, z: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
        """Perform the source-summing operation for a single time and chunk."""
        zdotz(z, out=out)
        cp.cuda.Device().synchronize()
        return out

    def sum_chunks(self, out: np.ndarray):
        """Sum the chunks into the output array.

        Here we need to also re-shape the visibility array into the output array.

        Parameters
        ----------
        out
            The output visibilities, with shape (Nfeed, Nfeed, Npairs).
        """
        if self.nchunks > 1:
            for i in range(1, len(self.vis)):
                self.vis[0] += self.vis[i]

        cpu = self.vis[0].get()

        # cpu = cpu.transpose((0, 2, 1, 3))
        cpu = cpu.transpose((1, 3, 2, 0))

        if self.all_pairs:
            cpu = cpu.reshape((self.nant * self.nant, self.nfeed, self.nfeed))
            out[:] = cpu
        else:
            out[:] = cpu[self.ant1_idx, self.ant2_idx]

        cp.cuda.Device().synchronize()


class GPUVectorDot(MatProd):
    """Use a loop over specific pairs, performing a vdot over the source axis."""

    def allocate_vis(self):
        """Allocate memory for the visibilities."""
        self.vis = [
            cp.full(
                (self.nfeed, self.nfeed, self.npairs),
                self.ctype(0.0),
                dtype=self.ctype,
                order="F",
            )
            for _ in range(self.nchunks)
        ]

    def compute(self, z: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
        """Perform the source-summing operation for a single time and chunk."""
        z = z.reshape((self.nant, self.nfeed, -1))

        for i, (ai, aj) in enumerate(self.antpairs):
            complex_matmul(z[ai], z[aj], out=out[:, :, i])

        cp.cuda.Device().synchronize()
        return out

    def sum_chunks(self, out: np.ndarray):
        """Sum the chunks into the output array."""
        if self.nchunks > 1:
            for i in range(1, len(self.vis)):
                self.vis[0] += self.vis[i]

        out[:] = self.vis[0].transpose((2, 1, 0)).get()
        cp.cuda.Device().synchronize()
