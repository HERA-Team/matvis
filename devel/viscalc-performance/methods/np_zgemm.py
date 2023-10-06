from scipy.linalg import blas

from ._lib import Solver

class NpZgemm(Solver):
    def compute(self):
        if self.transposed:
            return blas.zgemm(alpha=1, a=self.z, b=self.z.conj(), trans_a=True)
        else:
            return blas.zgemm(alpha=1, a=self.z, b=self.z.conj(), trans_b=True)
