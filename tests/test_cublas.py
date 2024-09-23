"""Tests of the cublas wrapper functions."""

import pytest

pytest.importorskip("cupy")

import cupy as cp
import numpy as np

from matvis.gpu import _cublas as cb


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_zdotz(dtype):
    """Test the gemm function."""
    a = np.arange(6).reshape((2, 3)).astype(dtype)
    a += 1j
    # agpu = cp.asarray(a, order='F')
    # c = cp.cublas.gemm('N', 'H', agpu, agpu)#
    c = cb.zdotz(cp.asarray(a))
    np.testing.assert_allclose(
        c.get(),
        np.dot(a.conj(), a.T),
        rtol=1e-5 if dtype in [np.float32, np.complex64] else 1e-10,
    )
