from ._lib import Solver, RedundantSolver

try:
    import pycuda.autoinit
    from pycuda import cumath as cm
    from pycuda import driver, gpuarray
    from skcuda.cublas import (
        cublasCreate,
        cublasDestroy,
        cublasZgemm,
        cublasZherk,
        cublasZdotc,
        cublasCgemm,
        cublasCherk,
        cublasCdotc,
    )
    HAVE_PYCUDA = True 
except:
    HAVE_PYCUDA = False

class _CuBLASCommon:
    def setup(self):
        self.z = gpuarray.to_gpu(self._z)
        self.h = cublasCreate()

        if self._z.dtype is complex:
            self.gemm = cublasZgemm
            self.herk = cublasZherk
            self.dotc = cublasZdotc
        else:
            self.gemm = cublasCgemm
            self.herk = cublasCherk
            self.dotc = cublasCdotc

class _CuBLAS(Solver, _CuBLASCommon):

    def setup(self):
        super().setup()
        nant = self.z.shape[0]
        self.out = gpuarray.empty(shape=(nant, nant), dtype=self._z.dtype)
        
class _CuBLASRed(RedundantSolver, _CuBLASCommon):
    def setup(self):
        super().setup()
        self.out = gpuarray.empty(self.pairs.shape[0], dtype=self._z.dtype)