from scipy.linalg import blas

from ._lib import Solver


class NpZgemm(Solver):
    def compute(self):
        return blas.zgemm(alpha=1, a=self.z, b=self.z.conj(), trans_b=True)
