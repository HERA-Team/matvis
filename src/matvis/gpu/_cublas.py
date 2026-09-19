"""Thin cuBLAS wrappers exposing ``zdotz`` and ``complex_matmul``."""

import ctypes
import logging
from math import isqrt
from pathlib import Path

import cupy as cp
import numpy as np
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas

logger = logging.getLogger(__name__)

KERNELS_PATH = Path(__file__).parent / "kernels"

CUBLAS_FILL_MODE_LOWER = 0

_PTR, _INT = ctypes.c_void_p, ctypes.c_int

_SO_NAMES = (
    "libcublas.so.13",
    "libcublas.so.12",
    "libcublas.so.11",
    "libcublas.so",
)


def _load_cublas_ext():
    """Bind cgemm3m/cherk/zherk from the libcublas already loaded by cupy.

    These routines are not exposed by cupy, so they are bound with ctypes
    from the same libcublas that cupy loaded, and run on cupy's handle and
    the current cupy stream.
    """
    # cupy has already loaded libcublas into the process, so dlopen-ing by
    # soname resolves to the same library (no new load).
    for soname in _SO_NAMES:
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
_MIRROR_MODULE = cp.RawModule(code=(KERNELS_PATH / "mirror_hermitian.cu").read_text())


def _mirror_hermitian(out: cp.ndarray, n: int):
    """Fill the upper triangle of column-major ``out`` from the lower one.

    ``out`` may have any shape; only its underlying buffer (n*n contiguous
    elements, column-major) is used.
    """
    kern = _MIRROR_MODULE.get_function(
        "mirror_c" if out.dtype == np.complex64 else "mirror_z"
    )
    total = n * n
    # 256 is a conventional warp-multiple default, not empirically tuned
    # for this kernel/shape.
    block = 256
    kern(((total + block - 1) // block,), (block,), (out, np.int64(n)))


def _sync_handle_stream(handle):
    """Point the cuBLAS handle at the current cupy stream."""
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)


def finalize_zdotz(out):
    """Complete a chain of ``zdotz(..., mirror=False)`` calls on ``out``.

    ``zdotz`` normally mirrors the herk-computed lower triangle into the upper
    one on every call. When several calls accumulate into the same buffer
    (``beta=1``) only the final result needs mirroring, so they pass
    ``mirror=False`` and call this once at the end. It is a no-op on the
    :func:`complex_matmul` fallback path, which writes the full matrix.

    ``out`` may have any shape; its buffer must hold ``n*n`` contiguous
    elements in column-major order.
    """
    if _LIB is None:  # pragma: no cover
        return out
    n = isqrt(out.size)
    if n * n != out.size:
        raise ValueError(f"out must hold a square matrix, got {out.size} elements")
    _mirror_hermitian(out, n)
    return out


def zdotz(a, out=None, alpha=1.0, beta=0.0, mirror=True):
    """Compute the Hermitian Gram product ``a.conj() @ a.T``.

    Note that this is the convention used throughout matvis, rather than
    ``aa^H``. Uses the Hermitian rank-k routine ``cherk``/``zherk`` to compute one half
    of the visibility matrix, then fills in the other half with a small mirroring kernel.
    Falls back to :func:`complex_matmul` if the
    cuBLAS shared library cannot be loaded directly (see ``_load_cublas_ext``).

    Parameters
    ----------
    mirror
        If False, skip the mirroring kernel and leave the upper triangle of
        ``out`` undefined. Use this when accumulating several products into one
        buffer with ``beta=1``, and call :func:`finalize_zdotz` on the result.
        Note that ``beta`` likewise only applies to the lower triangle on the
        herk path, so an un-mirrored buffer must not be read before then.
    """
    m, k = a.shape
    if not a._c_contiguous:
        raise ValueError("a must be C-contiguous")

    if out is None:
        out = cp.empty((m, m), dtype=a.dtype, order="F")
    elif not out._f_contiguous:
        raise ValueError("out must be F-contiguous")

    if _LIB is None:
        return complex_matmul(a, a, out=out, alpha=alpha, beta=beta)

    if a.dtype == "complex64":
        func = _LIB.cublasCherk_v2
        rtype = np.float32
    elif a.dtype == "complex128":
        func = _LIB.cublasZherk_v2
        rtype = np.float64
    else:
        raise TypeError(
            f"invalid dtype for a: {a.dtype} (must be complex64 or complex128)"
        )

    alpha = np.array(alpha, dtype=rtype)
    beta = np.array(beta, dtype=rtype)

    handle = device.get_cublas_handle()
    _sync_handle_stream(handle)
    # alpha/beta are host (numpy) scalars passed by pointer, so the handle needs
    # HOST pointer mode here; restore the caller's mode afterwards since the
    # handle is cupy's shared global one and other code may rely on its mode.
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

    if mirror:
        _mirror_hermitian(out, m)
    return out


def complex_matmul(a, b, out=None, alpha=1.0, beta=0.0):
    """Compute ``a.conj() @ b.T``.

    For complex64 uses ``cgemm3m`` (Gauss 3M algorithm, which is roughly 2x faster than
    cgemm for typical matvis shapes) when available, otherwise ``cgemm``/``zgemm``.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"a and b must have the same shape, got {a.shape} and {b.shape}"
        )
    use_3m = _LIB is not None
    if a.dtype == "complex64":
        func = _LIB.cublasCgemm3m if use_3m else cublas.cgemm
    elif a.dtype == "complex128":
        # zgemm3m is only faster on hardware where fp64 FMA is the bottleneck;
        # it also isn't implemented on all arches, so keep zgemm here.
        use_3m = False
        func = cublas.zgemm
    else:
        raise TypeError(
            f"invalid dtype for a: {a.dtype} (must be complex64 or complex128)"
        )

    transa = cublas.CUBLAS_OP_C
    transb = cublas.CUBLAS_OP_N
    m, k = a.shape
    n = m
    if not a._c_contiguous:
        raise ValueError("a must be C-contiguous")

    if out is None:
        out = cp.empty((m, n), dtype=a.dtype, order="F")
    elif not out._f_contiguous:
        raise ValueError("out must be F-contiguous")

    alpha = np.array(alpha, dtype=a.dtype)
    beta = np.array(beta, dtype=a.dtype)

    handle = device.get_cublas_handle()
    _sync_handle_stream(handle)
    # alpha/beta are host (numpy) scalars passed by pointer, so the handle needs
    # HOST pointer mode here; restore the caller's mode afterwards since the
    # handle is cupy's shared global one and other code may rely on its mode.
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
        # Restore the original pointer mode for CUBLAS (see comment above)
        cublas.setPointerMode(handle, orig_mode)

    return out
