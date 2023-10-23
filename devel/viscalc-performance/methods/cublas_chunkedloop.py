"""Chunked-loop implementation in CuBLAS."""
import numpy as np

from . import _cublas as _cu
from ._cublas import _CuBLASRed, cuda_take_along_axis


class CuBLASChunkedLoop(_CuBLASRed):
    """Chunked-loop implementation in CuBLAS."""

    def setup(self):
        """Set it up."""
        super().setup()

        # Now, chunk the pairs into lists of pairs, where each list has the same
        # first antenna.
        ants = {}

        for a, b in self.pairs:
            if a not in ants:
                ants[a] = [b]
            else:
                ants[a].append(b)

        # Put them on the GPU
        self.ants = {
            k: _cu.gpuarray.to_gpu(np.sort(v).astype(np.int32)) for k, v in ants.items()
        }
        most_pairs = max(len(v) for v in ants.values())
        self.out_chunk = _cu.gpuarray.empty((most_pairs,), dtype=self.z.dtype)

    def compute(self):
        """Compute it."""
        out = np.zeros(len(self.pairs), dtype=self.z.dtype)
        n = 0

        if self.transposed:
            nsrc, nant = self.z.shape
            zc = self.z.conj()

            for a, b in self.ants.items():
                # Make new contiguous array for these antennas.
                m = cuda_take_along_axis(zc, b, axis=1)
                thisn = len(b)
        
                self.gemv(
                    self.h,
                    "n",  # conjugate transpose for first (remember fortran order)
                    thisn,
                    nsrc,
                    1.0,
                    m.gpudata,
                    thisn,
                    self.z[:, a].gpudata,
                    thisn,
                    0.0,
                    self.out_chunk.gpudata,
                    1,
                )
                out[n : n + thisn] = self.out_chunk[:thisn].get()
                n += thisn
        else:
            nant, nsrc = self.z.shape

            out = np.zeros(len(self.pairs), dtype=self.z.dtype)

            n = 0
            for a, b in self.ants.items():
                # Make new contiguous array for these antennas.
                m = cuda_take_along_axis(self.z, b)
                thisn = len(b)

                self.gemv(
                    self.h,
                    "c",  # conjugate transpose for first (remember fortran order)
                    nsrc,
                    thisn,
                    1.0,
                    m.gpudata,
                    nsrc,
                    self.z[a].gpudata,
                    1,
                    0.0,
                    self.out_chunk.gpudata,
                    1,
                )
                out[n : n + thisn] = self.out_chunk[:thisn].get()
                n += thisn

        return out
