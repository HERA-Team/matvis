from scipy.linalg import blas

from ._lib import Solver


class NpZherk(Solver):
    def compute(self):
        return blas.zherk(alpha=1, a=self.z)
