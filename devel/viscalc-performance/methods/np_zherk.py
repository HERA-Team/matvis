from ._lib import Solver
from scipy.linalg import blas

class NpZherk(Solver):
    def compute(self):
        return blas.zherk(alpha=1, a=self.z)