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
    """Check complex_matmul computes a.conj() @ b.T."""
    rng = np.random.default_rng(42)
    a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    b = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)

    c = cb.complex_matmul(cp.asarray(a), cp.asarray(b))
    np.testing.assert_allclose(
        c.get(),
        np.dot(a.conj(), b.T),
        rtol=1e-4 if dtype == np.complex64 else 1e-10,
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
