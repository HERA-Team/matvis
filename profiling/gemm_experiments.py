"""Compare cuBLAS strategies for the matvis Gram product V = Z Z^H.

Candidates:
- cgemm (current implementation, via matvis.gpu._cublas.zdotz)
- cgemm3m (Gauss 3M complex algorithm; ~25% fewer real FLOPs)
- cherk (Hermitian rank-k update; computes one triangle, half the FLOPs)
- operand-order variants (OP_N x OP_C instead of OP_C x OP_N)

cupy's cublas backend does not expose cgemm3m/cherk, so those are called
through ctypes on libcublas directly, reusing cupy's handle/stream.

Run: uv run python profiling/gemm_experiments.py --nant 350 --nsrc 50000
"""

import argparse
import ctypes
import ctypes.util
import time

import cupy as cp
import numpy as np
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas

from matvis.gpu._cublas import zdotz

# cublasStatus_t cublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda,
#                              B, ldb, beta, C, ldc)  -- same signature as cgemm.
# cublasStatus_t cublasCherk(handle, uplo, trans, n, k, alpha(float*), A, lda,
#                            beta(float*), C, ldc)
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1


def load_cublas():
    """Load the same libcublas cupy is using."""
    for name in ("cublas", "cublas.so.13", "cublas.so.12", "cublas.so.11"):
        path = ctypes.util.find_library(name)
        if path:
            return ctypes.CDLL(path)
    return ctypes.CDLL("libcublas.so")


def timeit(fn, n=10):
    """Average wall time of fn() over n runs after a warmup call."""
    fn()
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - t0) / n


def main():
    """Run the GEMM strategy comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--nant", type=int, default=350)
    ap.add_argument("--nsrc", type=int, default=50_000)
    ap.add_argument("--nfeed", type=int, default=2)
    ap.add_argument("--nax", type=int, default=2)
    args = ap.parse_args()

    M = args.nant * args.nfeed
    K = args.nax * args.nsrc
    flops = 8 * M * M * K

    rng = cp.random.default_rng(0)
    r = rng.standard_normal((M, K), dtype=np.float32)
    i = rng.standard_normal((M, K), dtype=np.float32)
    z = (r + 1j * i).astype(np.complex64)
    out = cp.empty((M, M), dtype=np.complex64, order="F")

    dev = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {dev['name'].decode()}   M={M} K={K}  complex64")
    print(f"{'strategy':<28}{'ms':>10}{'TFLOPS':>10}{'max rel err':>14}")

    ref = None

    def report(name, t, result, eff_flops=flops):
        nonlocal ref
        if ref is None:
            ref = result.copy()
            err = 0.0
        else:
            err = float(cp.abs(result - ref).max() / cp.abs(ref).max())
        print(f"{name:<28}{t * 1e3:>10.2f}{eff_flops / t / 1e12:>10.2f}{err:>14.2e}")

    # --- current implementation --------------------------------------------
    t = timeit(lambda: zdotz(z, out=out))
    report("cgemm (current zdotz)", t, out)

    lib = load_cublas()
    handle = device.get_cublas_handle()
    one = np.array(1.0 + 0j, dtype=np.complex64)
    zero = np.array(0.0 + 0j, dtype=np.complex64)
    fone = np.array(1.0, dtype=np.float32)
    fzero = np.array(0.0, dtype=np.float32)

    # The v2 symbols must be used, with explicit prototypes: the legacy
    # non-_v2 symbols take the old API and silently misinterpret arguments.
    ptr, i32 = ctypes.c_void_p, ctypes.c_int
    lib.cublasCgemm3m.restype = i32
    lib.cublasCgemm3m.argtypes = [
        ptr,
        i32,
        i32,
        i32,
        i32,
        i32,
        ptr,
        ptr,
        i32,
        ptr,
        i32,
        ptr,
        ptr,
        i32,
    ]
    lib.cublasCherk_v2.restype = i32
    lib.cublasCherk_v2.argtypes = [
        ptr,
        i32,
        i32,
        i32,
        i32,
        ptr,
        ptr,
        i32,
        ptr,
        ptr,
        i32,
    ]

    orig_mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    # --- cgemm3m -------------------------------------------------------------
    out3m = cp.empty((M, M), dtype=np.complex64, order="F")

    def gemm3m():
        st = lib.cublasCgemm3m(
            handle,
            cublas.CUBLAS_OP_C,
            cublas.CUBLAS_OP_N,
            M,
            M,
            K,
            one.ctypes.data,
            z.data.ptr,
            K,
            z.data.ptr,
            K,
            zero.ctypes.data,
            out3m.data.ptr,
            M,
        )
        assert st == 0, f"cublasCgemm3m failed: {st}"

    t = timeit(gemm3m)
    report("cgemm3m", t, out3m)

    # --- cherk (one triangle) ------------------------------------------------
    outherk = cp.zeros((M, M), dtype=np.complex64, order="F")

    def herk():
        # C (MxM, fortran) = alpha * op(A) op(A)^H + beta C with op=CUBLAS_OP_C:
        # in cublas terms A is KxM fortran (our C-contiguous MxK), trans=C gives
        # A^H A of size MxM = conj(z) z^T elementwise == zdotz.
        st = lib.cublasCherk_v2(
            handle,
            CUBLAS_FILL_MODE_LOWER,
            cublas.CUBLAS_OP_C,
            M,
            K,
            fone.ctypes.data,
            z.data.ptr,
            K,
            fzero.ctypes.data,
            outherk.data.ptr,
            M,
        )
        assert st == 0, f"cublasCherk failed: {st}"

    t = timeit(herk)
    # fill the other triangle for the correctness check
    full = cp.tril(outherk) + cp.tril(outherk, -1).conj().T
    report("cherk (tri only)", t, full, eff_flops=flops)

    # herk + mirror cost
    def herk_mirror():
        herk()
        cp.tril(outherk)  # placeholder mirror cost; a real impl uses a tiny kernel

    t = timeit(herk_mirror)
    report("cherk + mirror", t, full, eff_flops=flops)

    cublas.setPointerMode(handle, orig_mode)


if __name__ == "__main__":
    main()
