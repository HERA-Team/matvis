import numba
import numpy as np

from ._lib import RedundantSolver


@numba.njit
def _dumbloopjit(z, pairs, out):
    zc = z.conj()
    for i, (a, b) in enumerate(pairs):
        out[i] = np.dot(z[a], zc[b])


class SingleLoopNumba(RedundantSolver):
    def setup(self):
        self.z = self._z
        self.out = np.empty(len(self.pairs), dtype=self._z.dtype)

    def compute(self):
        _dumbloopjit(self.z, self.pairs, self.out)
        return self.out
