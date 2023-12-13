import cupy as cp
import numpy as np
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas


def zdotz(a, out=None, alpha=1.0, beta=0.0):
    """Computes a.conj() @ a.T."""
    if a.dtype == "complex64":
        func = cublas.cgemm
    elif a.dtype == "complex128":
        func = cublas.zgemm
    else:
        raise TypeError(f"invalid dtype: {a.dtype}")

    transa = cublas.CUBLAS_OP_C  # _trans_to_cublas_op(transa)
    transb = cublas.CUBLAS_OP_N
    m, k = a.shape
    n = m
    assert a._c_contiguous

    if out is None:
        out = cp.empty((m, n), dtype=a.dtype, order="F")
    else:
        assert out._f_contiguous

    alpha = np.array(alpha, dtype=a.dtype)
    alpha_ptr = alpha.ctypes.data

    beta = np.array(beta, dtype=a.dtype)
    beta_ptr = beta.ctypes.data

    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    lda = a.shape[1]
    ldb = a.shape[1]

    c = out
    func(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        a.data.ptr,
        lda,
        a.data.ptr,
        ldb,
        beta_ptr,
        c.data.ptr,
        m,
    )
    cublas.setPointerMode(handle, orig_mode)

    return out


def dotc(x, y, out: cp.ndarray | None = None):
    """Computes the dot product of x.conj() and y."""
    if x.dtype == "complex64":
        func = cublas.cdotc
    elif x.dtype == "complex128":
        func = cublas.zdotc
    else:
        raise TypeError(f"invalid dtype: {x.dtype}")

    if out is None:
        out = cp.empty((), dtype=x.dtype)

    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)

    try:
        func(handle, x.size, x.data.ptr, 1, y.data.ptr, 1, out.data.ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)

    return out
