"""Tests of the cublas wrapper functions."""

import pytest

pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

import cupy as cp
import numpy as np

from matvis.gpu import _cublas as cb


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(2, 3), (7, 129), (64, 5000)])
def test_zdotz(dtype, shape):
    """Check zdotz produces the full hermitian Gram matrix a.conj() @ a.T."""
    rng = np.random.default_rng(1234)
    a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)

    c = cb.zdotz(cp.asarray(a))
    np.testing.assert_allclose(
        c.get(),
        np.dot(a.conj(), a.T),
        rtol=1e-4 if dtype == np.complex64 else 1e-10,
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(2, 3), (7, 129), (64, 5000)])
def test_complex_matmul(dtype, shape):
    """Check complex_matmul computes a.conj() @ b.T.

    complex64 goes through cublasCgemm3m (the Gauss 3M algorithm), which
    trades some rounding accuracy for fewer real multiplies; a somewhat
    looser tolerance than plain cgemm is expected and documented cuBLAS
    behaviour, not a correctness bug.
    """
    rng = np.random.default_rng(42)
    a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    b = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)

    c = cb.complex_matmul(cp.asarray(a), cp.asarray(b))
    np.testing.assert_allclose(
        c.get(),
        np.dot(a.conj(), b.T),
        rtol=1e-3 if dtype == np.complex64 else 1e-10,
    )


def test_zdotz_out_and_beta():
    """Check zdotz honours a preallocated out and accumulates with beta=1."""
    rng = np.random.default_rng(7)
    a = (rng.standard_normal((8, 100)) + 1j * rng.standard_normal((8, 100))).astype(
        np.complex64
    )
    expected = np.dot(a.conj(), a.T)

    out = cp.zeros((8, 8), dtype=np.complex64, order="F")
    cb.zdotz(cp.asarray(a), out=out)
    cb.zdotz(cp.asarray(a), out=out, beta=1.0)
    np.testing.assert_allclose(out.get(), 2 * expected, rtol=1e-4)


def test_zdotz_invalid_dtype():
    """Zdotz should reject non-complex input dtypes."""
    a = cp.zeros((4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="invalid dtype"):
        cb.zdotz(a)


def test_zdotz_raises_on_non_c_contiguous_a():
    """Zdotz should reject an `a` that is not C-contiguous."""
    a = cp.zeros((4, 8), dtype=np.complex64, order="F")
    with pytest.raises(ValueError, match="a must be C-contiguous"):
        cb.zdotz(a)


def test_zdotz_raises_on_non_f_contiguous_out():
    """Zdotz should reject a preallocated `out` that is not F-contiguous."""
    a = cp.zeros((4, 8), dtype=np.complex64)
    out = cp.zeros((4, 4), dtype=np.complex64, order="C")
    with pytest.raises(ValueError, match="out must be F-contiguous"):
        cb.zdotz(a, out=out)


def test_complex_matmul_invalid_dtype():
    """complex_matmul should reject non-complex input dtypes."""
    a = cp.zeros((4, 4), dtype=np.float64)
    with pytest.raises(TypeError, match="invalid dtype"):
        cb.complex_matmul(a, a)


def test_complex_matmul_raises_on_k_mismatch():
    """complex_matmul should reject a and b whose source (column) axes disagree.

    ``a`` and ``b`` are allowed to have a different number of *rows* (this is what
    lets matprod block-dispatch multiply a rectangular antenna-group block), but
    the shared source/K axis (their number of columns) must match.
    """
    a = cp.zeros((4, 8), dtype=np.complex64)
    b = cp.zeros((5, 9), dtype=np.complex64)
    with pytest.raises(ValueError, match="same number of columns"):
        cb.complex_matmul(a, b)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_matmul_rectangular(dtype):
    """complex_matmul must support a and b with different numbers of rows (M != N).

    This is required by non-square matprod blocks (e.g. a 3-antenna x 5-antenna block).
    """
    rng = np.random.default_rng(11)
    a = (rng.standard_normal((3, 20)) + 1j * rng.standard_normal((3, 20))).astype(dtype)
    b = (rng.standard_normal((5, 20)) + 1j * rng.standard_normal((5, 20))).astype(dtype)

    c = cb.complex_matmul(cp.asarray(a), cp.asarray(b))
    assert c.shape == (3, 5)
    np.testing.assert_allclose(
        c.get(),
        np.dot(a.conj(), b.T),
        rtol=1e-3 if dtype == np.complex64 else 1e-10,
    )


def test_complex_matmul_rectangular_matches_square_case_when_equal():
    """Rectangular support must not change behaviour for the existing equal-shape case.

    This is the square-output usage that GPUVectorDot relies on.
    """
    rng = np.random.default_rng(12)
    a = (rng.standard_normal((2, 30)) + 1j * rng.standard_normal((2, 30))).astype(
        np.complex64
    )
    b = (rng.standard_normal((2, 30)) + 1j * rng.standard_normal((2, 30))).astype(
        np.complex64
    )
    c = cb.complex_matmul(cp.asarray(a), cp.asarray(b))
    assert c.shape == (2, 2)
    np.testing.assert_allclose(c.get(), np.dot(a.conj(), b.T), rtol=1e-3)


def test_complex_matmul_raises_on_preallocated_out_wrong_shape():
    """A preallocated `out` for a rectangular product must match (M, N), not (M, M)."""
    a = cp.zeros((3, 8), dtype=np.complex64)
    b = cp.zeros((5, 8), dtype=np.complex64)
    out = cp.zeros((3, 3), dtype=np.complex64, order="F")
    with pytest.raises(ValueError, match="shape"):
        cb.complex_matmul(a, b, out=out)


def test_complex_matmul_raises_on_non_c_contiguous_b():
    """B's contiguity must be validated too, not just a's.

    Block dispatch routinely passes a fancy-indexed (non-contiguous-by-default)
    `b`; silently accepting it would corrupt results rather than raising.
    """
    a = cp.zeros((3, 8), dtype=np.complex64)
    # A C-order (8, 5) array transposed to (5, 8) is F-contiguous, not C-contiguous.
    b = cp.zeros((8, 5), dtype=np.complex64, order="C").T
    assert b.shape == (5, 8) and not b._c_contiguous
    with pytest.raises(ValueError, match="b must be C-contiguous"):
        cb.complex_matmul(a, b)


@pytest.mark.parametrize("m,n", [(3, 5), (5, 3)])
def test_complex_matmul_rectangular_both_orientations(m, n):
    """Rectangular support must work with either operand being the larger one."""
    rng = np.random.default_rng(13)
    a = (rng.standard_normal((m, 20)) + 1j * rng.standard_normal((m, 20))).astype(
        np.complex64
    )
    b = (rng.standard_normal((n, 20)) + 1j * rng.standard_normal((n, 20))).astype(
        np.complex64
    )
    c = cb.complex_matmul(cp.asarray(a), cp.asarray(b))
    assert c.shape == (m, n)
    np.testing.assert_allclose(c.get(), np.dot(a.conj(), b.T), rtol=1e-3)


def test_complex_matmul_rectangular_out_and_beta_accumulation():
    """A preallocated rectangular `out` with beta=1 must accumulate.

    Exercises ldc/alpha/beta under M != N, which nothing else in this file does.
    """
    rng = np.random.default_rng(14)
    a = (rng.standard_normal((3, 15)) + 1j * rng.standard_normal((3, 15))).astype(
        np.complex64
    )
    b = (rng.standard_normal((5, 15)) + 1j * rng.standard_normal((5, 15))).astype(
        np.complex64
    )
    expected = np.dot(a.conj(), b.T)

    out = cp.zeros((3, 5), dtype=np.complex64, order="F")
    cb.complex_matmul(cp.asarray(a), cp.asarray(b), out=out)
    cb.complex_matmul(cp.asarray(a), cp.asarray(b), out=out, beta=1.0)
    np.testing.assert_allclose(out.get(), 2 * expected, rtol=1e-3)


def test_complex_matmul_rectangular_falls_back_without_lib(monkeypatch):
    """The cgemm/zgemm fallback path must also support M != N.

    Used when libcublas can't be bound directly; it has its own m/n/ld arguments.
    """
    rng = np.random.default_rng(15)
    a = (rng.standard_normal((3, 12)) + 1j * rng.standard_normal((3, 12))).astype(
        np.complex64
    )
    b = (rng.standard_normal((6, 12)) + 1j * rng.standard_normal((6, 12))).astype(
        np.complex64
    )
    monkeypatch.setattr(cb, "_LIB", None)
    c = cb.complex_matmul(cp.asarray(a), cp.asarray(b))
    assert c.shape == (3, 6)
    np.testing.assert_allclose(c.get(), np.dot(a.conj(), b.T), rtol=1e-3)


def test_complex_matmul_raises_on_non_c_contiguous_a():
    """complex_matmul should reject an `a` that is not C-contiguous."""
    a = cp.zeros((4, 8), dtype=np.complex64, order="F")
    with pytest.raises(ValueError, match="a must be C-contiguous"):
        cb.complex_matmul(a, a)


def test_complex_matmul_raises_on_non_f_contiguous_out():
    """complex_matmul should reject a preallocated `out` that is not F-contiguous."""
    a = cp.zeros((4, 8), dtype=np.complex64)
    out = cp.zeros((4, 4), dtype=np.complex64, order="C")
    with pytest.raises(ValueError, match="out must be F-contiguous"):
        cb.complex_matmul(a, a, out=out)


def test_zdotz_falls_back_to_complex_matmul_without_lib(monkeypatch):
    """When libcublas can't be bound directly, zdotz should use cgemm/zgemm."""
    rng = np.random.default_rng(99)
    a = (rng.standard_normal((5, 40)) + 1j * rng.standard_normal((5, 40))).astype(
        np.complex64
    )

    monkeypatch.setattr(cb, "_LIB", None)
    c = cb.zdotz(cp.asarray(a))
    expected = cb.complex_matmul(cp.asarray(a), cp.asarray(a))
    np.testing.assert_allclose(c.get(), expected.get(), rtol=1e-4)


@pytest.mark.parametrize(
    "dtype,symbol",
    [(np.complex64, "cublasCherk_v2"), (np.complex128, "cublasZherk_v2")],
)
def test_zdotz_raises_on_herk_failure(monkeypatch, dtype, symbol):
    """A non-zero cuBLAS status from herk should raise a RuntimeError."""
    a = cp.ones((4, 10), dtype=dtype)
    monkeypatch.setattr(cb._LIB, symbol, lambda *args: 13)
    with pytest.raises(RuntimeError, match="cublas herk failed"):
        cb.zdotz(a)


def test_complex_matmul_raises_on_gemm3m_failure(monkeypatch):
    """A non-zero cuBLAS status from cgemm3m should raise a RuntimeError."""
    a = cp.ones((4, 10), dtype=np.complex64)
    monkeypatch.setattr(cb._LIB, "cublasCgemm3m", lambda *args: 13)
    with pytest.raises(RuntimeError, match="cublas gemm3m failed"):
        cb.complex_matmul(a, a)


def test_load_cublas_ext_retries_sonames(monkeypatch):
    """_load_cublas_ext should try each soname until one loads successfully."""
    attempted = []

    class FakeCDLL:
        def __init__(self, name):
            attempted.append(name)
            if len(attempted) < len(cb._SO_NAMES) - 1:
                raise OSError(f"cannot load {name}")
            self._fns = {}

        def __getattr__(self, name):
            fn = lambda *a, **kw: 0  # noqa: E731
            self._fns[name] = fn
            return fn

    class FakeCtypes:
        CDLL = FakeCDLL

    monkeypatch.setattr(cb, "ctypes", FakeCtypes())
    lib = cb._load_cublas_ext()

    assert lib is not None
    assert len(attempted) == len(cb._SO_NAMES) - 1


def test_load_cublas_ext_returns_none_if_all_sonames_fail(monkeypatch):
    """_load_cublas_ext should return None (not raise) if no soname loads."""

    class FakeCDLL:
        def __init__(self, name):
            raise OSError(f"cannot load {name}")

    class FakeCtypes:
        CDLL = FakeCDLL

    monkeypatch.setattr(cb, "ctypes", FakeCtypes())
    assert cb._load_cublas_ext() is None


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_zdotz_deferred_mirror_accumulates(dtype):
    """zdotz(mirror=False) + beta=1 + finalize_zdotz == the summed full products.

    This is the accumulation path used by GPUMatMul: several chunks add into
    one buffer with only the lower triangle valid, and the upper triangle is
    filled in once at the end.
    """
    rng = np.random.default_rng(7)
    shape = (16, 257)
    chunks = [
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
        for _ in range(3)
    ]

    out = cp.zeros((shape[0], shape[0]), dtype=dtype, order="F")
    for a in chunks:
        cb.zdotz(cp.asarray(a), out=out, beta=1.0, mirror=False)
    cb.finalize_zdotz(out)

    expected = sum(np.dot(a.conj(), a.T) for a in chunks)
    np.testing.assert_allclose(
        out.get(), expected, rtol=1e-4 if dtype == np.complex64 else 1e-10
    )


def test_finalize_zdotz_raises_on_non_square():
    """finalize_zdotz needs a buffer holding a square matrix."""
    out = cp.zeros(7, dtype=np.complex64)
    with pytest.raises(ValueError, match="square matrix"):
        cb.finalize_zdotz(out)
