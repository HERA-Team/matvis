"""Thin, fast wrappers around cuBLAS for the matvis hot path.

Two entry points:

``zdotz(a, out)``
    The Gram product ``a.conj() @ a.T`` (the matvis V = Z Z^H). Uses the
    Hermitian rank-k routine ``cherk``/``zherk`` (half the FLOPs of a general
    GEMM: only one triangle is computed), then mirrors the triangle with a
    small kernel. Falls back to ``complex_matmul`` if the cuBLAS shared
    library cannot be loaded directly.

``complex_matmul(a, b, out)``
    General ``a.conj() @ b.T``. For complex64 uses ``cgemm3m`` (Gauss
    3M algorithm, ~25% fewer real FLOPs — measured ~2x faster than ``cgemm``
    for the skinny matvis shapes) when available, otherwise ``cgemm``/``zgemm``.

The 3M/HERK routines are not exposed by cupy, so they are bound with ctypes
from the same libcublas that cupy loaded, and run on cupy's handle and the
current cupy stream.
"""

import ctypes
import logging

import cupy as cp
import numpy as np
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas

logger = logging.getLogger(__name__)

CUBLAS_FILL_MODE_LOWER = 0

_PTR, _INT = ctypes.c_void_p, ctypes.c_int


def _load_cublas_ext():
    """Bind cgemm3m/cherk/zherk from the libcublas already loaded by cupy."""
    # cupy has already loaded libcublas into the process, so dlopen-ing by
    # soname resolves to the same library (no new load).
    for soname in (
        "libcublas.so.13",
        "libcublas.so.12",
        "libcublas.so.11",
        "libcublas.so",
    ):
        try:
            lib = ctypes.CDLL(soname)
            break
        except OSError:
            continue
    else:  # pragma: no cover
        return None

    try:
        gemm_sig = (
            [_PTR] + [_INT] * 5 + [_PTR, _PTR, _INT, _PTR, _INT, _PTR, _PTR, _INT]
        )
        herk_sig = [_PTR] + [_INT] * 4 + [_PTR, _PTR, _INT, _PTR, _PTR, _INT]
        for name, sig in [
            ("cublasCgemm3m", gemm_sig),
            ("cublasZgemm3m", gemm_sig),
            ("cublasCherk_v2", herk_sig),
            ("cublasZherk_v2", herk_sig),
        ]:
            fn = getattr(lib, name)
            fn.restype = _INT
            fn.argtypes = sig
    except AttributeError:  # pragma: no cover
        return None
    return lib


_LIB = _load_cublas_ext()
if _LIB is None:  # pragma: no cover
    logger.warning(
        "Could not bind cgemm3m/cherk from libcublas; falling back to cgemm/zgemm."
    )

# Mirror the (valid) lower triangle of a column-major hermitian matrix into
# the upper triangle.
_MIRROR_MODULE = cp.RawModule(
    code=r"""
#include <cupy/complex.cuh>
extern "C" {
__global__ void mirror_c(complex<float>* C, long n) {
    long p = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (p >= n * n) return;
    long c = p / n, r = p % n;  // column-major: p = c*n + r
    if (r < c) C[p] = conj(C[r * n + c]);
}
__global__ void mirror_z(complex<double>* C, long n) {
    long p = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (p >= n * n) return;
    long c = p / n, r = p % n;
    if (r < c) C[p] = conj(C[r * n + c]);
}
}
"""
)


def _mirror_hermitian(out: cp.ndarray, n: int):
    """Fill the upper triangle of column-major ``out`` from the lower one.

    ``out`` may have any shape; only its underlying buffer (n*n contiguous
    elements, column-major) is used.
    """
    kern = _MIRROR_MODULE.get_function(
        "mirror_c" if out.dtype == np.complex64 else "mirror_z"
    )
    total = n * n
    block = 256
    kern(((total + block - 1) // block,), (block,), (out, np.int64(n)))


def _sync_handle_stream(handle):
    """Point the cuBLAS handle at the current cupy stream."""
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)


def zdotz(a, out=None, alpha=1.0, beta=0.0):
    """Compute the Hermitian Gram product a.conj() @ a.T."""
    m, k = a.shape
    assert a._c_contiguous

    if out is None:
        out = cp.empty((m, m), dtype=a.dtype, order="F")
    else:
        assert out._f_contiguous

    if _LIB is None:
        return complex_matmul(a, a, out=out, alpha=alpha, beta=beta)

    if a.dtype == "complex64":
        func = _LIB.cublasCherk_v2
        rtype = np.float32
    elif a.dtype == "complex128":
        func = _LIB.cublasZherk_v2
        rtype = np.float64
    else:
        raise TypeError(f"invalid dtype: {a.dtype}")

    alpha = np.array(alpha, dtype=rtype)
    beta = np.array(beta, dtype=rtype)

    handle = device.get_cublas_handle()
    _sync_handle_stream(handle)
    orig_mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    try:
        status = func(
            handle,
            CUBLAS_FILL_MODE_LOWER,
            cublas.CUBLAS_OP_C,
            m,
            k,
            alpha.ctypes.data,
            a.data.ptr,
            k,
            beta.ctypes.data,
            out.data.ptr,
            m,
        )
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if status != 0:
        raise RuntimeError(f"cublas herk failed with status {status}")

    _mirror_hermitian(out, m)
    return out


def complex_matmul(a, b, out=None, alpha=1.0, beta=0.0):
    """Computes a.conj() @ b.T."""
    assert a.shape == b.shape
    use_3m = _LIB is not None
    if a.dtype == "complex64":
        func = _LIB.cublasCgemm3m if use_3m else cublas.cgemm
    elif a.dtype == "complex128":
        # zgemm3m is only faster on hardware where fp64 FMA is the bottleneck;
        # it also isn't implemented on all arches, so keep zgemm here.
        use_3m = False
        func = cublas.zgemm
    else:
        raise TypeError(f"invalid dtype: {a.dtype}")

    transa = cublas.CUBLAS_OP_C
    transb = cublas.CUBLAS_OP_N
    m, k = a.shape
    n = m
    assert a._c_contiguous

    if out is None:
        out = cp.empty((m, n), dtype=a.dtype, order="F")
    else:
        assert out._f_contiguous

    alpha = np.array(alpha, dtype=a.dtype)
    beta = np.array(beta, dtype=a.dtype)

    handle = device.get_cublas_handle()
    _sync_handle_stream(handle)
    orig_mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    lda = a.shape[1]
    ldb = a.shape[1]

    try:
        if use_3m:
            status = func(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha.ctypes.data,
                a.data.ptr,
                lda,
                b.data.ptr,
                ldb,
                beta.ctypes.data,
                out.data.ptr,
                m,
            )
            if status != 0:
                raise RuntimeError(f"cublas gemm3m failed with status {status}")
        else:
            func(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha.ctypes.data,
                a.data.ptr,
                lda,
                b.data.ptr,
                ldb,
                beta.ctypes.data,
                out.data.ptr,
                m,
            )
    finally:
        cublas.setPointerMode(handle, orig_mode)

    return out
