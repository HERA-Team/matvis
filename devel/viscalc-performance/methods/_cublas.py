from ._lib import RedundantSolver, Solver
import ctypes
import numpy as np
from functools import cache
from jinja2 import Template
import os
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        import pycuda.autoinit
        from pycuda import cumath as cm

        from pycuda import driver, gpuarray, compiler
        from skcuda.cublas import (
            cublasCgemm,
            cublasCherk,
            cublasCreate,
            cublasDestroy,
            cublasZgemm,
            cublasZherk,
            cublasZgemv,
            cublasCgemv,
            _libcublas,
            cuda,
        )

    HAVE_PYCUDA = True
except:
    HAVE_PYCUDA = False

def cublasZdotc_inplace(handle, n, x, incx, y, incy, result):
    _libcublas.cublasZdotc_v2(
        handle, n, int(x), incx, int(y), incy, int(result)
    )   

def cublasCdotc_inplace(handle, n, x, incx, y, incy, result):
    _libcublas.cublasCdotc_v2(
        handle, n, int(x), incx, int(y), incy,
        int(result)
    )


class _CuBLASCommon:
    def setup(self):
        self.z = gpuarray.to_gpu(self._z)
        self.h = cublasCreate()

        if self._z.dtype.name == 'complex128':
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
        super().setup()
        if self.transposed:
            nant = self.z.shape[1]
        else:
            nant = self.z.shape[0]
        self.out = gpuarray.empty(shape=(nant, nant), dtype=self._z.dtype)


class _CuBLASRed(_CuBLASCommon, RedundantSolver):
    def setup(self):
        super().setup()
        self.out = gpuarray.empty(self.pairs.shape[0], dtype=self._z.dtype)


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
    def compile_take(dtype):
        take_along_axis_template = Template(take_along_axis_kernel_code)
        take_along_axis_code = take_along_axis_template.render(TYPE=dtype)
        return (
            compiler.SourceModule(take_along_axis_code).get_function(f"TakeAlongFirstAxis2D"),
            compiler.SourceModule(take_along_axis_code).get_function(f"TakeAlongSecondAxis2D"),
        )


    take_along_first_axis_z, take_along_second_axis_z = compile_take('cuDoubleComplex')
    take_along_first_axis_c, take_along_second_axis_c = compile_take('cuFloatComplex')

    @cache
    def _cudatake_blocksize(n: int,k: int):
        nthreads_max = 1024
        threadsx = min(nthreads_max, n) 
        threadsy = min(k, nthreads_max // threadsx)
        block = (threadsx, threadsy, 1)
        grid = (int(np.ceil(n / threadsx)), int(np.ceil(k / threadsy)), 1)
        return block, grid

    def cuda_take_along_axis(x: gpuarray.GPUArray, idx: gpuarray.GPUArray, axis=0) -> gpuarray.GPUArray:
        m, n = x.shape
        if axis==0:
            out = gpuarray.empty((idx.shape[0], n), dtype=x.dtype)
        else:
            out = gpuarray.empty((m, idx.shape[0]), dtype=x.dtype)

        if idx.dtype.name == "int64":
            idx = idx.astype(np.int32)

        block, grid = _cudatake_blocksize(n, idx.shape[0])
        if axis==0:
            fnc = take_along_first_axis_z if x.dtype.name == 'complex128' else take_along_first_axis_c
        else:
            fnc = take_along_second_axis_z if x.dtype.name == 'complex128' else take_along_second_axis_c

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