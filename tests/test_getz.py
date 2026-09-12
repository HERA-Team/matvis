"""Tests for the fused GPU Z-matrix kernel."""

import pytest

pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

import cupy as cp
import numpy as np

from matvis.gpu.getz import GPUZMatrixCalc


def _random_complex(rng, shape, dtype):
    r = rng.standard_normal(shape)
    i = rng.standard_normal(shape)
    return (r + 1j * i).astype(dtype)


def _reference_z(beam, exptau, sqrt_flux, beam_idx, nant, nfeed, nax, nsrc):
    if beam_idx is None:
        # No explicit indexing: either one shared beam, or one beam per antenna
        # (aligned with antenna order), matching the broadcast semantics of
        # matvis.core.getz.ZMatrixCalc.
        a = beam if beam.shape[0] == 1 else beam
        a = np.broadcast_to(a, (nant, nfeed, nax, nsrc))
    else:
        a = beam[beam_idx]
    return (a * exptau[:, None, None, :] * sqrt_flux).reshape(nant * nfeed, nax * nsrc)


@pytest.mark.parametrize("ctype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "nant,nbeam,beam_idx",
    [
        (4, 4, np.array([2, 0, 3, 1])),  # explicit beam_idx, permuted
        (5, 1, None),  # single beam shared by all antennas
        (4, 4, None),  # one beam per antenna, aligned to antenna order
    ],
    ids=["beam_idx", "shared_beam", "beam_per_antenna_implicit"],
)
def test_fused_z(ctype, nant, nbeam, beam_idx):
    """Fused Z must match the reference computation, including on a cached call."""
    rng = np.random.default_rng(0)
    nfeed, nax, nsrc = 2, 2, 10
    rtype = np.float32 if ctype == np.complex64 else np.float64

    beam = _random_complex(rng, (nbeam, nfeed, nax, nsrc), ctype)
    exptau = _random_complex(rng, (nant, nsrc), ctype)
    sqrt_flux = rng.standard_normal(nsrc).astype(rtype)

    calc = GPUZMatrixCalc(nant=nant, nfeed=nfeed, nax=nax, nsrc=nsrc, ctype=ctype)
    calc.setup()

    expected = _reference_z(beam, exptau, sqrt_flux, beam_idx, nant, nfeed, nax, nsrc)
    rtol = 1e-5 if ctype == np.complex64 else 1e-10

    # First call populates the cached device beam_idx array...
    z1 = calc(cp.asarray(sqrt_flux), cp.asarray(beam), cp.asarray(exptau), beam_idx)
    np.testing.assert_allclose(z1.get(), expected, rtol=rtol)

    # ...and the second call reuses it (same instance, same beam_idx).
    z2 = calc(cp.asarray(sqrt_flux), cp.asarray(beam), cp.asarray(exptau), beam_idx)
    np.testing.assert_allclose(z2.get(), expected, rtol=rtol)
