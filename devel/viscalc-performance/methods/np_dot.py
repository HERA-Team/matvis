import numpy as np

from ._lib import Solver


class NpDot(Solver):
    def compute(self):
        return np.dot(self.z, self.z.conj().T)
