from ._lib import Solver
from scipy.linalg import blas

class NpZgemm(Solver):
    def compute(self):
        return blas.zgemm(alpha=1, a=self.z, b=self.z.conj(), trans_b=True)