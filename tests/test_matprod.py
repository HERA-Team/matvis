"""Tests that the various matprod methods produce the same result."""

import numpy as np
import pytest

from matvis._utils import get_dtypes


def simple_matprod(z):
    """The simplest matrix product, to compare to."""
    return z.conj().dot(z.T)


@pytest.mark.parametrize("nfeed", [1, 2])
@pytest.mark.parametrize("antpairs", [True, False])
@pytest.mark.parametrize("precision", [1, 2])
@pytest.mark.parametrize(
    "method",
    [
        "CPUMatMul",
        "CPUVectorDot",
        pytest.param("GPUMatMul", marks=pytest.mark.gpu),
        pytest.param("GPUVectorDot", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("nchunks", [1, 2])
def test_matprod(nfeed, antpairs, precision, method, nchunks):
    """Test that the various matprod methods produce the same result."""
    if method.startswith("GPU"):
        pytest.importorskip("cupy")
        import cupy as cp

        from matvis.gpu import matprod as module
    else:
        from matvis.cpu import matprod as module

    cls = getattr(module, method)
    nant = 5
    nsrc = 15
    if antpairs:
        antpairs = np.array([(i, j) for i in range(nant) for j in range(nant)])
    else:
        antpairs = None

    obj = cls(
        nchunks=nchunks, nfeed=nfeed, nant=nant, antpairs=antpairs, precision=precision
    )
    obj.setup()

    ctype = get_dtypes(precision)[1]
    z = np.arange(nfeed * nant * nsrc, dtype=ctype).reshape((nfeed * nant, nsrc))
    z = z + 1j

    if method.startswith("GPU"):
        z = cp.asarray(z)

    out = np.zeros((obj.npairs, nfeed, nfeed), dtype=ctype)

    obj(z, chunk=0)
    obj.sum_chunks(out)

    # Use a simple method to get the true answer
    # NOTE: we have transposed the feed-feed submatrices here (i.e. (3,1) instead (1, 3))
    #       this is because otherwise the outputs do not match pyuvsim. This should be
    #       checked properly.
    true = (
        simple_matprod(z)
        .reshape((nant, nfeed, nant, nfeed))
        .transpose((0, 2, 3, 1))
        .reshape((-1, nfeed, nfeed))
    )
    if method.startswith("GPU"):
        true = true.get()

    assert out.dtype.type == ctype
    assert out.shape == (obj.npairs, nfeed, nfeed)
    np.testing.assert_allclose(out, true)


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("GPUMatMul", marks=pytest.mark.gpu),
        pytest.param("GPUVectorDot", marks=pytest.mark.gpu),
    ],
)
def test_matprod_skipped_chunk_is_not_stale(method):
    """A chunk not computed this integration must contribute nothing.

    matvis skips chunks with no sources above the horizon, so ``compute`` is
    never called for them. The accumulator must therefore be reset by
    ``sum_chunks``; otherwise the skipped chunk silently contributes the
    *previous* integration's visibilities.
    """
    pytest.importorskip("cupy")
    import cupy as cp

    from matvis.gpu import matprod as module

    cls = getattr(module, method)
    nant, nfeed, nsrc, nchunks = 5, 2, 15, 2
    ctype = get_dtypes(1)[1]

    obj = cls(nchunks=nchunks, nfeed=nfeed, nant=nant, antpairs=None, precision=1)
    obj.setup()

    z = cp.asarray(
        np.arange(nfeed * nant * nsrc, dtype=ctype).reshape((nfeed * nant, nsrc)) + 1j
    )

    # Integration 0: both chunks have sources up.
    both = np.zeros((obj.npairs, nfeed, nfeed), dtype=ctype)
    obj(z, chunk=0)
    obj(z, chunk=1)
    obj.sum_chunks(both)

    # Integration 1: chunk 1 is entirely below the horizon, so it is skipped.
    one = np.zeros((obj.npairs, nfeed, nfeed), dtype=ctype)
    obj(z, chunk=0)
    obj.sum_chunks(one)

    np.testing.assert_allclose(both, 2 * one, rtol=1e-5)
