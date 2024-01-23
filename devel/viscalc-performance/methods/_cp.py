import cupy as cp
from ._lib import Solver, RedundantSolver


from skcuda.cublas import (
    cublasCgemm,
    cublasCgemv,
    cublasCherk,
    cublasZgemm,
    cublasZgemv,
    cublasZherk,
    cublasCreate,
)


class _CuPyCommon:
    def create(self):
        self.h = cublasCreate()
        if self._z.dtype.name == "complex128":
            self.gemm = cublasZgemm
            self.herk = cublasZherk
            self.gemv = cublasZgemv
        else:
            self.gemm = cublasCgemm
            self.herk = cublasCherk
            self.gemv = cublasCgemv

class _CuPy(_CuPyCommon, Solver):
    def setup(self):
        super().setup()
        if self.transposed:
            nant = self.z.shape[1]
        else:
            nant = self.z.shape[0]
        self.out = cp.empty(shape=(nant, nant), dtype=self._z.dtype)
        self.z = cp.asarray(self._z)
        

class _CuPyRed(_CuPyCommon, RedundantSolver):
    def setup(self):
        super().setup()
        self.out = cp.empty(self.pairs.shape[0], dtype=self._z.dtype)
        self.z = cp.asarray(self._z)
        