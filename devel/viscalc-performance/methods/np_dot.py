import numpy as np

from ._lib import Solver


class NpDot(Solver):
    def compute(self):
        if self.transposed:
            return np.dot(self.z.T, self.z.conj())
        else:
            return np.dot(self.z, self.z.conj().T)
