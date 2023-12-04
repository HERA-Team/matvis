import numpy as np
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from pycuda import autoinit, gpuarray
        from skcuda.cublas import (
            _libcublas,
            cublasCgemm,
            cublasCreate,
            cublasDgemm,
            cublasSgemm,
            cublasZgemm,
        )

    HAVE_PYCUDA = True
except Exception:
    HAVE_PYCUDA = False

if HAVE_PYCUDA:
    _DEFAULT_CUBLAS_HANDLE = cublasCreate()  # handle for managing cublas

    DOT_MAP = {
        np.dtype(np.complex64): _libcublas.cublasCdotc_v2,
        np.dtype(np.complex128): _libcublas.cublasZdotc_v2,
        np.dtype(np.float32): _libcublas.cublasSdot_v2,
        np.dtype(np.float64): _libcublas.cublasDdot_v2,
    }

    def _cublas_dotc_inplace(handle, n, x, incx, y, incy, result):
        """In-place Zdotc that doesn't transfer data to the host."""
        fnc = DOT_MAP[x.dtype.type]
        fnc(handle, n, int(x), incx, int(y), incy, int(result))

    def dotc(
        a: gpuarray.GPUArray,
        b: gpuarray.GPUArray,
        out: gpuarray.GPUArray | None = None,
        h=_DEFAULT_CUBLAS_HANDLE,
    ) -> gpuarray.GPUArray:
        """Dot-product of two vectors on the GPU.

        Note that if the arrays are complex, the second one is conjugated.
        """
        if out is None:
            out = gpuarray.empty((1,), a.dtype)

        return _cublas_dotc_inplace(h, a.size, a.gpudata, 1, b.gpudata, 1, out.gpudata)

    GEMM_MAP = {
        np.dtype(np.complex64): cublasCgemm,
        np.dtype(np.complex128): cublasZgemm,
        np.dtype(np.float32): cublasSgemm,
        np.dtype(np.float64): cublasDgemm,
    }

    def gemm(
        a: gpuarray.GPUArray,
        b: gpuarray.GPUArray,
        out: gpuarray.GPUArray | None = None,
        h=_DEFAULT_CUBLAS_HANDLE,
        trans_a="n",
        trans_b="n",
    ) -> gpuarray.GPUArray:
        """Matrix-matrix multiplication on the GPU."""
        fnc = GEMM_MAP[a.dtype.type]

        if out is None:
            out = gpuarray.empty((a.shape[0], b.shape[1]), a.dtype)

        fnc(  # compute crdtop = dot(eq2top,crd_eq)
            h,
            trans_a,
            trans_b,
            a.shape[0],
            a.shape[1],
            b.shape[1],
            1.0,
            a.gpudata,  # crd_eq_gpu.gpudata,
            a.shape[0],
            b.gpudata,  # eq2top_gpu.gpudata,
            b.shape[0],
            0.0,
            out.gpudata,
            a.shape[0],
        )
        return out

    def zz(
        z: gpuarray.GPUArray,
        out: gpuarray.GPUArray | None = None,
        h=_DEFAULT_CUBLAS_HANDLE,
    ) -> gpuarray.GPUArray:
        """Compute the matrix product of a matrix with itself (conjugated)."""
        return gemm(z, z, out, trans_a="h", h=h)
