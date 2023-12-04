"""Tests of the cublas wrapper functions."""


import pytest

import numpy as np
from pycuda.gpuarray import to_gpu

from matvis import _cublas as cb


@pytest.mark.parameterize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_dotc(dtype):
    """Test the dotc function."""
    a = np.random.randn(10).astype(dtype)
    b = np.random.randn(10).astype(dtype)
    if np.iscomplex(a):
        a += 1j * np.random.randn(10).astype(dtype)
        b += 1j * np.random.randn(10).astype(dtype)

    c = cb.dotc(to_gpu(a), to_gpu(b))
    assert np.allclose(c, np.vdot(a, b))


@pytest.mark.parameterize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_gemm(dtype):
    """Test the gemm function."""
    a = np.random.randn(10, 12).astype(dtype)
    b = np.random.randn(12, 10).astype(dtype)
    if np.iscomplex(a):
        a += 1j * np.random.randn(10, 12).astype(dtype)
        b += 1j * np.random.randn(12, 10).astype(dtype)

    c = cb.gemm(to_gpu(a), to_gpu(b))
    assert np.allclose(c, np.dot(a, b))


@pytest.mark.parameterize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_zz(dtype):
    """Test the zz function."""
    a = np.random.randn(10, 15).astype(dtype)
    if np.iscomplex(a):
        a += 1j * np.random.randn(10, 15).astype(dtype)

    c = cb.zz(to_gpu(a))
    assert np.allclose(c, np.dot(a.conj(), a.T))
