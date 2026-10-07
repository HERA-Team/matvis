"""Independent full-Stokes visibility and accumulation regressions."""

import numpy as np
import pytest

from matvis.core.coherency import process_polarized_chunk
from matvis.core.getz import ZMatrixCalc
from matvis.cpu import matprod as cpu_mp

# These fixture metadata warnings do not affect the Jones or visibility values.
pytestmark = [
    pytest.mark.filterwarnings("ignore:The mount_type parameter must be set"),
    pytest.mark.filterwarnings("ignore:No beam information:UserWarning"),
    pytest.mark.filterwarnings("ignore:mount_type, antenna_diameters:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:The default baseline conjugation convention has changed:UserWarning"
    ),
]


METHODS = ["CPUMatMul", "CPUVectorDot", "CPUMatBlock"] + [
    pytest.param(name, marks=pytest.mark.gpu)
    for name in ("GPUMatMul", "GPUVectorDot", "GPUMatBlock")
]


def _components(method: str, precision: int, nant: int, nsrc: int, nbeam: int):
    gpu = method.startswith("GPU")
    xp = pytest.importorskip("cupy") if gpu else np
    if gpu:
        from matvis.gpu import matprod as mp
        from matvis.gpu.getz import GPUZMatrixCalc as zcls
    else:
        mp, zcls = cpu_mp, ZMatrixCalc
    order = np.array([2, 0, 1]) if method.endswith("MatBlock") else None
    blocks = [(np.array([i]), np.arange(i, nant)) for i in range(nant)]
    kw = {"antenna_blocks": blocks, "antenna_order": order} if order is not None else {}
    pos = getattr(mp, method)(2, 2, nant, None, precision=precision, **kw)
    neg = getattr(mp, method)(2, 2, nant, None, precision=precision, **kw)
    zcalc = zcls(
        nant,
        2,
        2,
        nsrc,
        np.complex64 if precision == 1 else np.complex128,
        antenna_order=order,
    )
    for obj in (pos, neg, zcalc):
        obj.setup()
    return xp, pos, neg, zcalc


def _coherency(stokes: np.ndarray) -> np.ndarray:
    i, q, u, v = stokes
    return 0.5 * np.array([[i + q, u + 1j * v], [u - 1j * v, i - q]])


def _oracle(
    beam: np.ndarray, phase: np.ndarray, coherency: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    nant, nsrc = phase.shape
    out = np.zeros((nant * nant, 2, 2), dtype=np.complex128)
    for i in range(nant):
        for j in range(nant):
            for s in range(nsrc):
                out[i * nant + j] += (
                    beam[indices[i], :, :, s]
                    @ coherency[:, :, s]
                    @ beam[indices[j], :, :, s].conj().T
                    * phase[i, s].conj()
                    * phase[j, s]
                )
    return out


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("precision", [1, 2])
@pytest.mark.parametrize("sky", ["positive", "partition", "mixed", "near_diagonal"])
@pytest.mark.parametrize("mapping", ["shared", "implicit", "indexed", "reused"])
def test_polarized_rime(method: str, precision: int, sky: str, mapping: str):
    """All product methods recover the direct RIME with complex, distinct beams."""
    rng = np.random.default_rng(123)
    nant, nsrc = 3, 7
    nbeam = {"shared": 1, "implicit": 3, "indexed": 3, "reused": 2}[mapping]
    indices = {
        "shared": [0, 0, 0],
        "implicit": [0, 1, 2],
        "indexed": [2, 0, 1],
        "reused": [1, 0, 1],
    }[mapping]
    indices = np.array(indices)
    bidx = indices if mapping in ("indexed", "reused") else None
    dtype = np.float32 if precision == 1 else np.float64
    ctype = np.complex64 if precision == 1 else np.complex128
    stokes = np.array(
        [
            [2, 2, 2, 2, 0, 1, 2],
            [0.5, 0, 0, 0.2, 0, 1, -0.3],
            [0, 0.5, 0, 0.3, 0, 0, 0.1],
            [0, 0, 0.5, 0.4, 0, 0, -0.2],
        ],
        dtype=dtype,
    )
    if sky == "near_diagonal":
        stokes[:, 0] = [2, 1, 1e-4, 1e-4]
    elif sky == "partition":
        stokes[:, 5:] *= -1
    elif sky == "mixed":
        stokes[0, 0] = 0.1
    original = _coherency(stokes)
    processed = stokes.copy()
    if sky == "partition":
        processed[:, 5:] *= -1
    flux = _coherency(processed).transpose(2, 0, 1)[:, None].astype(ctype)
    beam = (
        rng.normal(size=(nbeam, 2, 2, nsrc)) + 1j * rng.normal(size=(nbeam, 2, 2, nsrc))
    ).astype(ctype)
    phase = np.exp(1j * rng.normal(size=(nant, nsrc))).astype(ctype)
    expected = _oracle(beam, phase, original, indices)
    xp, pos, neg, zcalc = _components(method, precision, nant, nsrc, nbeam)
    phase_device = xp.asarray(phase)
    phase_before = phase_device.copy()
    beam_device = xp.asarray(beam)
    beam_before = beam_device.copy()
    flux_device = xp.asarray(flux * 0.5)
    flux_before = flux_device.copy()
    # Reuse the same objects across integrations; each chunk holds half the sky flux.
    for _ in range(2):
        for chunk in range(2):
            process_polarized_chunk(
                flux_device,
                zcalc,
                beam_device,
                phase_device,
                bidx,
                pos,
                chunk,
                sky in ("partition", "mixed"),
                matprod_neg=neg,
                use_partition=sky == "partition",
                n_P_chunk=5,
                n_N_chunk=2,
                xp=xp,
            )
        out = np.zeros_like(expected, dtype=ctype)
        minus = np.zeros_like(out)
        pos.sum_chunks(out)
        neg.sum_chunks(minus)
        tol = 2e-5 if precision == 1 else 1e-12
        np.testing.assert_allclose(out - minus, expected, rtol=tol, atol=tol)
        matrix = (out - minus).reshape(nant, nant, 2, 2)
        np.testing.assert_allclose(
            matrix, matrix.transpose(1, 0, 3, 2).conj(), rtol=tol, atol=tol
        )
    xp.testing.assert_array_equal(phase_device, phase_before)
    xp.testing.assert_array_equal(beam_device, beam_before)
    xp.testing.assert_array_equal(flux_device, flux_before)


@pytest.mark.parametrize("method", METHODS)
def test_partition_clears_empty_contributions(method: str):
    """A partition that disappears at the next time cannot reuse old visibilities."""
    xp, pos, neg, zcalc = _components(method, 2, 3, 2, 1)
    beam = xp.ones((1, 2, 2, 2), dtype=complex)
    phase = xp.ones((3, 2), dtype=complex)
    for npos, nneg in [(1, 1), (0, 1), (1, 0), (0, 0)]:
        flux = xp.zeros((2, 1, 2, 2), dtype=complex)
        flux[: npos + nneg, 0] = xp.eye(2)
        for chunk in range(2):
            process_polarized_chunk(
                flux,
                zcalc,
                beam,
                phase,
                None,
                pos,
                chunk,
                True,
                matprod_neg=neg,
                use_partition=True,
                n_P_chunk=npos,
                n_N_chunk=nneg,
                xp=xp,
            )
        out = np.zeros((9, 2, 2), dtype=complex)
        minus = np.zeros_like(out)
        pos.sum_chunks(out)
        neg.sum_chunks(minus)
        np.testing.assert_allclose(out - minus, 4 * (npos - nneg), rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mapping", ["shared", "implicit", "indexed"])
def test_stokes_i_complex_beams_matches_legacy(method: str, mapping: str):
    """Stokes-I-only agrees with scalar flux, including complex antenna beams."""
    from matvis import simulate_vis
    from matvis._test_utils import get_standard_sim_params, perturbed_beam

    gpu = method.startswith("GPU")
    if gpu:
        pytest.importorskip("cupy")
    kw, *_ = get_standard_sim_params(False, True, ntime=2, nsource=15)
    beam0 = kw["beams"][0]
    beam1 = beam0.clone(beam=perturbed_beam(beam0.beam, "feed_phase"))
    extra = {}
    if mapping == "implicit":
        kw["beams"] = [beam0, beam1, beam0]
    elif mapping == "indexed":
        kw["beams"] = [beam0, beam1]
        extra["beam_idx"] = np.array([1, 0, 1])
    if method.endswith("MatBlock"):
        from matvis.redundancy import antpairs_to_blocks

        extra["antenna_blocks"] = antpairs_to_blocks(
            [(i, j) for i in range(3) for j in range(i, 3)]
        )
    reference = simulate_vis(
        **kw, **extra, precision=2, use_gpu=gpu, matprod_method=method
    )
    stokes = np.zeros((4, *kw["fluxes"].shape))
    stokes[0] = kw.pop("fluxes")
    actual = simulate_vis(
        **kw, **extra, stokes=stokes, precision=2, use_gpu=gpu, matprod_method=method
    )
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "mode",
    [
        "flux",
        "stokes",
        "reject_polarized",
        "missing_sky",
        "both_skies",
        "reject_negative",
    ],
)
def test_backend_sky_contract(backend: str, mode: str):
    """Direct backend calls enforce the same sky contract and defaults as the wrapper."""
    import importlib

    from astropy.coordinates import SkyCoord

    from matvis._test_utils import get_standard_sim_params

    if backend == "gpu":
        pytest.importorskip("cupy")
    simulate = importlib.import_module(f"matvis.{backend}.{backend}").simulate
    kw, *_ = get_standard_sim_params(True, True, nsource=3, ntime=1)
    arguments = {
        "antpos": np.array(list(kw["ants"].values())),
        "freq": kw["freqs"][0],
        "times": kw["times"],
        "skycoords": SkyCoord(ra=kw["ra"], dec=kw["dec"], unit="rad"),
        "telescope_loc": kw["telescope_loc"],
        "beam_list": kw["beams"],
        "precision": 2,
    }
    stokes = np.zeros((4, 3))
    stokes[0] = kw["fluxes"][:, 0]
    if mode in ("flux", "both_skies"):
        arguments["I_sky"] = stokes[0]
    if mode in ("stokes", "reject_polarized", "both_skies", "reject_negative"):
        arguments["stokes"] = stokes
    if mode == "reject_polarized":
        arguments["polarized"] = False
    elif mode == "reject_negative":
        arguments["raise_on_negative_flux"] = True
        stokes[:, 0] *= -1
    if mode in ("flux", "stokes"):
        result = simulate(**arguments)
        assert result.shape == ((1, 9) if mode == "flux" else (1, 9, 2, 2))
        assert np.isfinite(result).all()
    else:
        message = {
            "reject_polarized": "incompatible with stokes",
            "reject_negative": "Negative eigenvalue",
            "missing_sky": "exactly one",
            "both_skies": "exactly one",
        }[mode]
        with pytest.raises(ValueError, match=message):
            simulate(**arguments)
