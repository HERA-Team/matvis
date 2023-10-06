from scipy.linalg import blas

from ._lib import Solver


class NpZherk(Solver):
    def compute(self):
        if self.transposed:
            return blas.zherk(alpha=1, a=self.z.T)
        else:
            return blas.zherk(alpha=1, a=self.z)
