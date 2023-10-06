import numpy as np

from . import _cublas as _cu
from ._cublas import _CuBLASRed, cuda_take_along_axis


class CuBLASChunkedLoop(_CuBLASRed):
    def setup(self):
        super().setup()

        # Now, chunk the pairs into lists of pairs, where each list has the same
        # first antenna.
        ants = {}

        if self.transposed:
            for a, b in self.pairs:
                if b not in ants:
                    ants[b] = [a]
                else:
                    ants[b].append(a)
        else:
            for a, b in self.pairs:
                if a not in ants:
                    ants[a] = [b]
                else:
                    ants[a].append(b)

        # Put them on the GPU
        self.ants = {
            k: _cu.gpuarray.to_gpu(np.sort(v).astype(np.int32)) for k, v in ants.items()
        }
        # most_pairs = max(len(v) for v in ants.values())

    def compute(self):
        if self.transposed:
            nsrc, nant = self.z.shape

            out = np.zeros(len(self.pairs), dtype=self.z.dtype)

            n = 0
            for a, b in self.ants.items():
                # Make new contiguous array for these antennas.
                m = cuda_take_along_axis(self.z, b, axis=1)
                print("m shape", m.shape, b.shape, b)
                thisn = len(b)
                print("thisn", thisn)
                this = _cu.gpuarray.empty((thisn,), dtype=self.z.dtype)

                self.gemv(
                    self.h,
                    "n",  # conjugate transpose for first (remember fortran order)
                    thisn,
                    nsrc,
                    1.0,
                    m.gpudata,
                    nsrc,
                    self.z[a].gpudata,
                    1,
                    0.0,
                    this.gpudata,
                    1,
                )
                out[n : n + thisn] = this.get()
                n += thisn

            return out

        else:
            nant, nsrc = self.z.shape

            out = np.zeros(len(self.pairs), dtype=self.z.dtype)

            n = 0
            for a, b in self.ants.items():
                # Make new contiguous array for these antennas.
                m = cuda_take_along_axis(self.z, b)
                thisn = len(b)
                this = _cu.gpuarray.empty((thisn,), dtype=self.z.dtype)

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
                    this.gpudata,
                    1,
                )
                out[n : n + thisn] = this.get()
                n += thisn

            return out
