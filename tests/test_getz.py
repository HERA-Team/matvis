"""Tests for the Z-matrix construction (numpy implementation and fused GPU kernel)."""

import numpy as np
import pytest

from matvis.core.getz import ZMatrixCalc

BACKENDS = ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]


def _random_complex(rng, shape, dtype):
    r = rng.standard_normal(shape)
    i = rng.standard_normal(shape)
    return (r + 1j * i).astype(dtype)


def _reference_z(
    beam, exptau, sqrt_flux, beam_idx, antenna_order, nant, nfeed, nax, nsrc
):
    """Z[p] = beam[beam_idx[ant]] * exptau[ant] * sqrt_flux, with ant = order[p]."""
    order = np.arange(nant) if antenna_order is None else np.asarray(antenna_order)
    if beam.shape[0] == 1:
        # One shared beam: broadcast, whatever the indexing says.
        a = np.broadcast_to(beam, (nant, nfeed, nax, nsrc))
    else:
        # Without beam_idx there is one beam per antenna, in antenna order.
        bidx = np.arange(nant) if beam_idx is None else np.asarray(beam_idx)
        a = beam[bidx[order]]
    return (a * exptau[order][:, None, None, :] * sqrt_flux).reshape(
        nant * nfeed, nax * nsrc
    )


def _run(backend, ncalls, *, antenna_order, sqrt_flux, beam, exptau, beam_idx, **kw):
    """Build a Z calculator on `backend` and call it `ncalls` times.

    Returns the results as numpy arrays. Calling more than once checks that
    cached device-side state (``beam_idx``, ``antenna_order``) is reusable.
    """
    if backend == "gpu":
        cp = pytest.importorskip("cupy")
        from matvis.gpu.getz import GPUZMatrixCalc

        calc = GPUZMatrixCalc(antenna_order=antenna_order, **kw)
        calc.setup()
        return [
            calc(
                cp.asarray(sqrt_flux), cp.asarray(beam), cp.asarray(exptau), beam_idx
            ).get()
            for _ in range(ncalls)
        ]

    calc = ZMatrixCalc(antenna_order=antenna_order, **kw)
    calc.setup()
    # The numpy implementation scales exptau in place, so hand it a fresh copy.
    return [
        np.array(calc(sqrt_flux, beam, exptau.copy(), beam_idx)) for _ in range(ncalls)
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ctype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "nant,nbeam,beam_idx",
    [
        (4, 4, np.array([2, 0, 3, 1])),  # explicit beam_idx, permuted
        (5, 1, None),  # single beam shared by all antennas
        (4, 4, None),  # one beam per antenna, aligned to antenna order
        (6, 3, np.array([0, 1, 2, 0, 1, 2])),  # 1 < nbeam < nant, beams reused
    ],
    ids=[
        "beam_idx",
        "shared_beam",
        "beam_per_antenna_implicit",
        "nbeam_between_1_and_nant",
    ],
)
@pytest.mark.parametrize("permuted", [False, True], ids=["natural", "permuted"])
def test_getz(backend, ctype, nant, nbeam, beam_idx, permuted):
    """Z must match the reference computation, including on a cached second call.

    ``antenna_order`` relabels which antenna each *row* of Z is built for; it has
    to compose correctly with every way the beams can be indexed, since the
    antenna -> beam map is what moves with it.
    """
    rng = np.random.default_rng(0)
    nfeed, nax, nsrc = 2, 2, 10
    rtype = np.float32 if ctype == np.complex64 else np.float64

    beam = _random_complex(rng, (nbeam, nfeed, nax, nsrc), ctype)
    exptau = _random_complex(rng, (nant, nsrc), ctype)
    sqrt_flux = rng.standard_normal(nsrc).astype(rtype)
    antenna_order = rng.permutation(nant) if permuted else None

    expected = _reference_z(
        beam, exptau, sqrt_flux, beam_idx, antenna_order, nant, nfeed, nax, nsrc
    )
    rtol = 1e-5 if ctype == np.complex64 else 1e-10

    for z in _run(
        backend,
        2,
        antenna_order=antenna_order,
        sqrt_flux=sqrt_flux,
        beam=beam,
        exptau=exptau,
        beam_idx=beam_idx,
        nant=nant,
        nfeed=nfeed,
        nax=nax,
        nsrc=nsrc,
        ctype=ctype,
    ):
        np.testing.assert_allclose(z, expected, rtol=rtol)


@pytest.mark.parametrize("backend", BACKENDS)
def test_getz_rejects_wrong_length_antenna_order(backend):
    """A mis-sized antenna_order is a hard error, not a silent mis-attribution."""
    kw = {"nant": 4, "nfeed": 2, "nax": 2, "nsrc": 5, "ctype": np.complex64}
    if backend == "gpu":
        pytest.importorskip("cupy")
        from matvis.gpu.getz import GPUZMatrixCalc as cls
    else:
        cls = ZMatrixCalc

    with pytest.raises(ValueError, match="antenna_order must have shape"):
        cls(antenna_order=np.arange(3), **kw)
