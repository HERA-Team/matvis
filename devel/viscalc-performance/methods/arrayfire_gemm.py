from ._lib import Solver

try:
    import arrayfire
    HAVE_AF = True
except:
    HAVE_AF = False


class ArrayFireGemm(Solver):
    def setup(self):
        self.out = arrayfire.Array(dtype=self._z.dtype.char, dims=(self._z.shape[0], self._z.shape[0]))
        self.z = arrayfire.from_ndarray(self._z)
        
    def compute(self):
        return arrayfire.blas.gemm(
            self.z, self.z, lhs_opts=arrayfire.MATPROP.NONE, rhs_opts=arrayfire.MATPROP.CTRANS, C=self.out
        )