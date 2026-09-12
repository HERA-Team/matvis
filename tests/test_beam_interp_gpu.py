"""Test the GPU beam interpolation routine."""

import itertools

import numpy as np
import pytest

pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

import cupy as cp
from cupyx.scipy import ndimage

from matvis.gpu.beams import gpu_beam_interpolation, prepare_for_map_coords


@pytest.fixture(scope="module")
def efield_beam_1freq(uvbeam):
    """A single-frequency, full-sphere e-field UVBeam (complex, 2 feeds x 2 axes)."""
    return uvbeam.select(freq_chans=[0], inplace=False)


@pytest.fixture(scope="module")
def power_beam_1freq(uvbeam_unpol):
    """A single-frequency, single-feed, real power UVBeam derived from the same beam."""
    return uvbeam_unpol.select(freq_chans=[0], inplace=False)


def _grid(uvb):
    """Get (beam data, daz, dza, azmin, az grid nodes, za grid nodes) for a UVBeam.

    The beam data shape is (1, Npols, Nza, Naz) for power beams. For Efield
    beams it is (Naxes_vec, Nfeeds, Nza, Naz).
    """
    d0, daz, dza, azmin = prepare_for_map_coords(uvb)
    nza, naz = d0.shape[-2:]
    az_nodes = azmin + daz * np.arange(naz)
    za_nodes = dza * np.arange(nza)
    return d0, daz, dza, azmin, az_nodes, za_nodes


@pytest.mark.parametrize("efield_or_power", ["efield", "power"])
@pytest.mark.parametrize("out_dtype", [np.complex64, np.complex128, None])
def test_noop_interpolation_matches_input(
    efield_or_power, out_dtype, efield_beam_1freq, power_beam_1freq
):
    """Interpolating at exactly the input grid nodes must reproduce the input.

    Covers the full input x output dtype matrix: real power beams (needing
    sqrt) and complex e-field beams (not), each into a matching, mismatched,
    or default-allocated output buffer.
    """
    is_power = efield_or_power == "power"
    uvb = power_beam_1freq if is_power else efield_beam_1freq
    d0, daz, dza, azmin, az_nodes, za_nodes = _grid(uvb)
    nax, nfeed, nza, naz = d0.shape
    AZ, ZA = np.meshgrid(az_nodes, za_nodes, indexing="xy")
    nsrc = AZ.size

    beam = cp.asarray(d0[np.newaxis])  # add nbeam=1 axis

    beam_at_src = None
    if out_dtype is not None:
        beam_at_src = cp.zeros((1, nfeed, nax, nsrc), dtype=out_dtype)

    out = gpu_beam_interpolation(
        beam,
        [daz],
        [dza],
        [azmin],
        cp.asarray(AZ.flatten()),
        cp.asarray(ZA.flatten()),
        beam_at_src=beam_at_src,
        power_beam=is_power,
    ).get()

    expected = np.sqrt(d0) if is_power else d0
    expected = expected.transpose(1, 0, 2, 3).reshape(nfeed, nax, nza, naz)
    out = out[0].reshape(nfeed, nax, nza, naz)

    tol = 1e-6 if (out_dtype or d0.dtype) == np.complex128 else 1e-4
    np.testing.assert_allclose(out, expected, atol=tol, rtol=tol)


def test_order_gt_1_matches_uvbeam_interp(efield_beam_1freq):
    """Order != 1 falls back to map_coordinates; cross-check against UVBeam.interp.

    Evaluated at non-node points (offset from the native grid), where a
    higher-order spline actually differs from linear interpolation, unlike
    testing at grid nodes (which any correctly-implemented interpolator
    reproduces exactly regardless of order).
    """
    order = 2
    d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)

    rng = np.random.default_rng(0)
    az = az_nodes[2:-2:7] + rng.uniform(0, daz, size=len(az_nodes[2:-2:7]))
    za = za_nodes[2:-2:5] + rng.uniform(0, dza, size=len(za_nodes[2:-2:5]))
    AZ, ZA = np.meshgrid(az, za, indexing="xy")

    beam = cp.asarray(d0[np.newaxis])
    out = gpu_beam_interpolation(
        beam,
        [daz],
        [dza],
        [azmin],
        cp.asarray(AZ.flatten()),
        cp.asarray(ZA.flatten()),
        order=order,
    ).get()
    nax, nfeed, nza, naz = d0.shape
    out = out[0].reshape(nfeed, nax, AZ.shape[0], AZ.shape[1])

    ref, _ = efield_beam_1freq.interp(
        az_array=AZ.flatten(),
        za_array=ZA.flatten(),
        interpolation_function="az_za_map_coordinates",
        spline_opts={"order": order},
        freq_array=np.atleast_1d(efield_beam_1freq.freq_array[0]),
        reuse_spline=False,
        return_basis_vector=False,
    )
    # UVBeam.interp returns (Naxes_vec, Nfeeds, Nfreqs, Npix); drop Nfreqs.
    ref = ref[:, :, 0, :].reshape(nax, nfeed, AZ.shape[0], AZ.shape[1])
    ref = ref.transpose(1, 0, 2, 3)

    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_bilinear_kernel_matches_map_coordinates(efield_beam_1freq):
    """The custom bilinear CUDA kernel (order=1) must agree with generic linear interpolation.

    ``gpu_beam_interpolation`` uses a hand-written CUDA kernel for order=1
    and only falls back to ``map_coordinates`` for order!=1, so no other
    test in this file cross-checks the kernel's numerics against a trusted,
    independent linear-interpolation implementation. This replicates the
    grid-unit coordinate transform matvis uses internally and feeds it
    through cupyx's map_coordinates directly (order=1), at points offset
    from the native grid so the comparison is non-trivial.
    """
    d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
    nax, nfeed, nza, naz = d0.shape

    rng = np.random.default_rng(1)
    az = az_nodes[2:-2:7] + rng.uniform(0, daz, size=len(az_nodes[2:-2:7]))
    za = za_nodes[2:-2:5] + rng.uniform(0, dza, size=len(za_nodes[2:-2:5]))
    AZ, ZA = np.meshgrid(az, za, indexing="xy")

    beam = cp.asarray(d0[np.newaxis])
    out = gpu_beam_interpolation(
        beam,
        [daz],
        [dza],
        [azmin],
        cp.asarray(AZ.flatten()),
        cp.asarray(ZA.flatten()),
    ).get()
    out = out[0].reshape(nfeed, nax, AZ.shape[0], AZ.shape[1])

    coords = cp.asarray([ZA.flatten() / dza, (AZ.flatten() - azmin) / daz])
    ref = np.zeros((nax, nfeed, AZ.size), dtype=d0.dtype)
    for ax, fd in itertools.product(range(nax), range(nfeed)):
        plane_out = cp.zeros(AZ.size, dtype=d0.dtype)
        ndimage.map_coordinates(
            cp.asarray(d0[ax, fd]), coords, order=1, output=plane_out
        )
        ref[ax, fd] = plane_out.get()
    ref = ref.transpose(1, 0, 2).reshape(nfeed, nax, AZ.shape[0], AZ.shape[1])

    np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)


def test_out_of_bounds_clamps_to_boundary(efield_beam_1freq):
    """Points beyond the za range must be assigned the boundary beam values.

    The kernel documents clamp-to-edge behaviour for out-of-range points
    (see bilinear_interp.cu) rather than extrapolating or erroring; this
    checks that behaviour directly by querying just below za=0 and just
    above the top of the za grid, at azimuths that fall exactly on grid
    nodes (so there's no azimuthal interpolation blur to account for).
    """
    d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
    nax, nfeed, nza, naz = d0.shape

    az = az_nodes[10:15]
    cases = [
        (za_nodes[0] - 10 * dza, 0),
        (za_nodes[-1] + 10 * dza, nza - 1),
    ]

    beam = cp.asarray(d0[np.newaxis])
    for za_val, za_idx in cases:
        za = np.full_like(az, za_val)
        out = gpu_beam_interpolation(
            beam, [daz], [dza], [azmin], cp.asarray(az), cp.asarray(za)
        ).get()
        out = out[0].reshape(nfeed, nax, len(az))
        expected = d0[:, :, za_idx, 10:15].transpose(1, 0, 2)
        np.testing.assert_allclose(out, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "real_dtype,expected_complex_dtype",
    [(np.float32, np.complex64), (np.float64, np.complex128)],
)
def test_power_beam_cast_to_complex_of_equivalent_precision(
    power_beam_1freq, real_dtype, expected_complex_dtype
):
    """Power-beam interpolation must always return a complex array at matching precision.

    This holds even with no explicit output buffer, which is otherwise
    allocated at the *input's* real dtype -- see the ``beam_at_src is None``
    branch of ``gpu_beam_interpolation``.
    """
    d0, daz, dza, azmin, az_nodes, za_nodes = _grid(power_beam_1freq)
    AZ, ZA = np.meshgrid(az_nodes[:3], za_nodes[:3], indexing="xy")

    beam = cp.asarray(d0[np.newaxis].astype(real_dtype))
    out = gpu_beam_interpolation(
        beam,
        [daz],
        [dza],
        [azmin],
        cp.asarray(AZ.flatten()),
        cp.asarray(ZA.flatten()),
        power_beam=True,
    )
    assert out.dtype == expected_complex_dtype


def test_power_beam_inferred_from_dtype(power_beam_1freq, efield_beam_1freq):
    """``power_beam=None`` must be inferred from whether the beam array is real or complex.

    The result should match the explicit equivalent for each case.
    """
    d0p, dazp, dzap, azminp, az_p, za_p = _grid(power_beam_1freq)
    d0e, daze, dzae, azmine, az_e, za_e = _grid(efield_beam_1freq)

    AZp, ZAp = np.meshgrid(az_p[:3], za_p[:3], indexing="xy")
    AZe, ZAe = np.meshgrid(az_e[:3], za_e[:3], indexing="xy")

    beam_p = cp.asarray(d0p[np.newaxis])
    out_p_auto = gpu_beam_interpolation(
        beam_p,
        [dazp],
        [dzap],
        [azminp],
        cp.asarray(AZp.flatten()),
        cp.asarray(ZAp.flatten()),
    ).get()
    out_p_explicit = gpu_beam_interpolation(
        beam_p,
        [dazp],
        [dzap],
        [azminp],
        cp.asarray(AZp.flatten()),
        cp.asarray(ZAp.flatten()),
        power_beam=True,
    ).get()
    np.testing.assert_array_equal(out_p_auto, out_p_explicit)

    beam_e = cp.asarray(d0e[np.newaxis])
    out_e_auto = gpu_beam_interpolation(
        beam_e,
        [daze],
        [dzae],
        [azmine],
        cp.asarray(AZe.flatten()),
        cp.asarray(ZAe.flatten()),
    ).get()
    out_e_explicit = gpu_beam_interpolation(
        beam_e,
        [daze],
        [dzae],
        [azmine],
        cp.asarray(AZe.flatten()),
        cp.asarray(ZAe.flatten()),
        power_beam=False,
    ).get()
    np.testing.assert_array_equal(out_e_auto, out_e_explicit)


def test_complex_beam_with_single_efield_axis_raises():
    """A complex beam with only one E-field axis should raise a clear error.

    This is most likely a mislabeled power beam (power beams should be
    real), so it should raise rather than silently skip the power-beam sqrt.
    """
    za = np.linspace(0, 1, 5)
    az = np.linspace(0, 1, 5)
    AZ, ZA = np.meshgrid(az, za, indexing="xy")
    beam = np.zeros((1, 1, 1, 5, 5), dtype=np.complex128)  # nax=1

    with pytest.raises(ValueError, match="only one Efield axis"):
        gpu_beam_interpolation(
            beam,
            az[1] - az[0],
            za[1] - za[0],
            az.min(),
            AZ.flatten(),
            ZA.flatten(),
        )


def test_wrong_beamtype():
    """Tests that a meaningful error is raised for a dumb beamtype."""
    za = np.linspace(0, 1, 5)
    az = np.linspace(0, 1, 5)

    AZ, ZA = np.meshgrid(az, za, indexing="xy")
    beam = np.array([1 - ZA])
    beam = beam[:, np.newaxis, np.newaxis]  # give it nax=1, nfeed=1

    dec_beam = beam[:, :, :, ::2][..., ::2]

    dza = za[1] - za[0]
    daz = az[1] - az[0]

    with pytest.raises(
        ValueError, match="as the dtype for beam, which is unrecognized"
    ):
        gpu_beam_interpolation(
            dec_beam.astype(int), daz * 2, dza * 2, 0.0, AZ.flatten(), ZA.flatten()
        )
