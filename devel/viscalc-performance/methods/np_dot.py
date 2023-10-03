from ._lib import Solver
import numpy as np

class NpDot(Solver):
    def compute(self):
        return np.dot(self.z, self.z.conj().T)