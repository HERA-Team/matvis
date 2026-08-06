"""Tests of the cublas wrapper functions."""

import pytest

pytest.importorskip("cupy")

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


def test_complex_matmul_invalid_dtype():
    """complex_matmul should reject non-complex input dtypes."""
    a = cp.zeros((4, 4), dtype=np.float64)
    with pytest.raises(TypeError, match="invalid dtype"):
        cb.complex_matmul(a, a)


def test_zdotz_falls_back_to_complex_matmul_without_lib(monkeypatch):
    """When libcublas can't be bound directly, zdotz should use cgemm/zgemm."""
    rng = np.random.default_rng(99)
    a = (rng.standard_normal((5, 40)) + 1j * rng.standard_normal((5, 40))).astype(
        np.complex64
    )

    monkeypatch.setattr(cb, "_LIB", None)
    c = cb.zdotz(cp.asarray(a))
    np.testing.assert_allclose(c.get(), np.dot(a.conj(), a.T), rtol=1e-4)


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
            if len(attempted) < 3:
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
    assert len(attempted) == 3


def test_load_cublas_ext_returns_none_if_all_sonames_fail(monkeypatch):
    """_load_cublas_ext should return None (not raise) if no soname loads."""

    class FakeCDLL:
        def __init__(self, name):
            raise OSError(f"cannot load {name}")

    class FakeCtypes:
        CDLL = FakeCDLL

    monkeypatch.setattr(cb, "ctypes", FakeCtypes())
    assert cb._load_cublas_ext() is None
