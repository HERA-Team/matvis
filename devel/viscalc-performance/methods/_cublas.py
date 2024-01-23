import ctypes
import numpy as np
import os
import warnings
from functools import cache
from jinja2 import Template

from ._lib import RedundantSolver, Solver

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from pycuda import autoinit
        from pycuda import compiler
        from pycuda import cumath as cm
        from pycuda import driver, gpuarray
        from skcuda.cublas import (
            _libcublas,
            cublasCgemm,
            cublasCgemv,
            cublasCherk,
            cublasCreate,
            cublasDestroy,
            cublasZgemm,
            cublasZgemv,
            cublasZherk,
            cuda,
        )

    HAVE_PYCUDA = True
except Exception:
    HAVE_PYCUDA = False


def cublasZdotc_inplace(handle, n, x, incx, y, incy, result):
    """In-place Zdotc that doesn't transfer data to the host."""
    _libcublas.cublasZdotc_v2(handle, n, int(x), incx, int(y), incy, int(result))


def cublasCdotc_inplace(handle, n, x, incx, y, incy, result):
    """In-place Cdotc that doesn't transfer data to the host."""
    _libcublas.cublasCdotc_v2(handle, n, int(x), incx, int(y), incy, int(result))


class _CuBLASCommon:
    def create(self):
        self.h = cublasCreate()

        if self._z.dtype.name == "complex128":
            self.gemm = cublasZgemm
            self.herk = cublasZherk
            self.dotc = cublasZdotc_inplace
            self.gemv = cublasZgemv
        else:
            self.gemm = cublasCgemm
            self.herk = cublasCherk
            self.dotc = cublasCdotc_inplace
            self.gemv = cublasCgemv


class _CuBLAS(_CuBLASCommon, Solver):
    def setup(self):
        self.create()
        if self.transposed:
            nant = self._z.shape[1]
        else:
            nant = self._z.shape[0]
        self.z = gpuarray.to_gpu(self._z)

        self.out = gpuarray.empty(shape=(nant, nant), dtype=self._z.dtype)


class _CuBLASRed(_CuBLASCommon, RedundantSolver):
    def setup(self):
        self.create()
        self.out = gpuarray.empty(self.pairs.shape[0], dtype=self._z.dtype)
        self.z = gpuarray.to_gpu(self._z)



take_along_axis_kernel_code = r"""
// CUDA code for copying values from a 2D array into another 2D array along an axis.

#include <cuComplex.h>
#include <pycuda-helpers.hpp>
#include <stdio.h>

__global__ void TakeAlongFirstAxis2D(
    {{ TYPE }} *in_matrix,
    int m,
    int n,
    int *indices,
    int k,
    {{ TYPE }} *out_matrix
){
    const uint i    = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j    = blockIdx.y * blockDim.y + threadIdx.y;

    const uint outel = j*n + i;

    // exit if we're out of scope
    if (j >= k || i >= n) return;

    int idx = indices[j];
    out_matrix[outel] = in_matrix[idx*n+i];
}


__global__ void TakeAlongSecondAxis2D(
    {{ TYPE }} *in_matrix,
    int m,
    int n,
    int *indices,
    int k,
    {{ TYPE }} *out_matrix
){
    const uint i    = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j    = blockIdx.y * blockDim.y + threadIdx.y;


    // exit if we're out of scope
    if (i >= k || j >= m) return;

    int idx = indices[i];
    const uint outel = j*k + i;

    out_matrix[j*k+i] = in_matrix[j*n+idx];
}
"""

if HAVE_PYCUDA:

    def sync():
        autoinit.context.synchronize()

    def _compile_take(dtype):
        take_along_axis_template = Template(take_along_axis_kernel_code)
        take_along_axis_code = take_along_axis_template.render(TYPE=dtype)
        return (
            compiler.SourceModule(take_along_axis_code).get_function(
                "TakeAlongFirstAxis2D"
            ),
            compiler.SourceModule(take_along_axis_code).get_function(
                "TakeAlongSecondAxis2D"
            ),
        )

    take_along_first_axis_z, take_along_second_axis_z = _compile_take("cuDoubleComplex")
    take_along_first_axis_c, take_along_second_axis_c = _compile_take("cuFloatComplex")

    @cache
    def _cudatake_blocksize(m: int, n: int, k: int, axis=0):
        nthreads_max = 1024
        if axis == 0:
            threadsx = min(nthreads_max, n)
            threadsy = min(k, nthreads_max // threadsx)
            grid = (int(np.ceil(n / threadsx)), int(np.ceil(k / threadsy)), 1)

        else:
            threadsx = min(nthreads_max, k)
            threadsy = min(m, nthreads_max // threadsx)
            grid = (int(np.ceil(k / threadsx)), int(np.ceil(m / threadsy)), 1)

        block = (threadsx, threadsy, 1)
        return block, grid

    def cuda_take_along_axis(
        x: gpuarray.GPUArray, idx: gpuarray.GPUArray, axis=0
    ) -> gpuarray.GPUArray:
        """Create a new GPUArray by taking indices along a given axis."""
        m, n = x.shape
        if axis == 0:
            out = gpuarray.empty((idx.shape[0], n), dtype=x.dtype)
        else:
            out = gpuarray.empty((m, idx.shape[0]), dtype=x.dtype)

        if idx.dtype.name == "int64":
            idx = idx.astype(np.int32)

        block, grid = _cudatake_blocksize(m, n, idx.shape[0], axis=axis)
        print(m, n, idx.shape[0], block, grid)
        if axis == 0:
            fnc = (
                take_along_first_axis_z
                if x.dtype.name == "complex128"
                else take_along_first_axis_c
            )
        else:
            fnc = (
                take_along_second_axis_z
                if x.dtype.name == "complex128"
                else take_along_second_axis_c
            )

        fnc(
            x.gpudata,
            np.int32(m),
            np.int32(n),
            idx.gpudata,
            np.int32(idx.shape[0]),
            out.gpudata,
            block=block,
            grid=grid,
        )
        return out

else:
    cuda_take_along_axis = None
