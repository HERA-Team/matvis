"""Test the GPU beam interpolation routine."""

import itertools

import numpy as np
import pytest

pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

import cupy as cp
from cupyx.scipy import ndimage
from scipy import ndimage as scipy_ndimage

from matvis import simulate_vis
from matvis._test_utils import get_standard_sim_params
from matvis.gpu.beams import (
    _KERNEL_ORDERS,
    gpu_beam_interpolation,
    prefilter_beam,
    prepare_for_map_coords,
)


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


def _scipy_reference(d0, daz, dza, azmin, az, za, order, mode="nearest"):
    """Interpolate ``d0`` at ``(az, za)`` with scipy, in matvis's output layout.

    This is the trusted, independent reference the CUDA kernels are checked
    against: plain ``scipy.ndimage.map_coordinates`` on the host, fed the same
    grid-unit coordinate transform that matvis uses internally. Returns shape
    ``(nfeed, nax, nsrc)``.
    """
    coords = np.array([np.asarray(za) / dza, (np.asarray(az) - azmin) / daz])
    nax, nfeed = d0.shape[:2]
    ref = np.empty((nax, nfeed, coords.shape[1]), dtype=d0.dtype)
    for ax, fd in itertools.product(range(nax), range(nfeed)):
        ref[ax, fd] = scipy_ndimage.map_coordinates(
            d0[ax, fd], coords, order=order, mode=mode
        )
    return ref.transpose(1, 0, 2)


def _cubic_interp(d0, daz, dza, azmin, az, za, **kwargs):
    """Run ``gpu_beam_interpolation`` at order 3 for a single beam, returning host data."""
    return gpu_beam_interpolation(
        prefilter_beam(d0[np.newaxis]),
        [daz],
        [dza],
        [azmin],
        cp.asarray(np.asarray(az)),
        cp.asarray(np.asarray(za)),
        order=3,
        **kwargs,
    ).get()[0]


def test_order_gt_1_matches_uvbeam_interp(efield_beam_1freq):
    """Orders other than 1 and 3 fall back to map_coordinates; cross-check against UVBeam.interp.

    Evaluated at non-node points (offset from the native grid), where a
    higher-order spline actually differs from linear interpolation, unlike
    testing at grid nodes (which any correctly-implemented interpolator
    reproduces exactly regardless of order).

    Both sides are pinned to mode="nearest" -- matvis's default, and what the
    fused kernels implement. For order >= 2 the mode selects the B-spline
    prefilter, so leaving the reference on scipy's "constant" default would
    compare two different interpolants.
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
        spline_opts={"order": order, "mode": "nearest"},
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
        order=1,
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


@pytest.mark.parametrize("efield_or_power", ["efield", "power"])
def test_power_beam_inferred_from_dtype(
    efield_or_power, power_beam_1freq, efield_beam_1freq
):
    """``power_beam=None`` must be inferred from whether the beam array is real or complex.

    The result should match the explicit equivalent for each case.
    """
    is_power = efield_or_power == "power"
    uvb = power_beam_1freq if is_power else efield_beam_1freq
    d0, daz, dza, azmin, az_nodes, za_nodes = _grid(uvb)
    AZ, ZA = np.meshgrid(az_nodes[:3], za_nodes[:3], indexing="xy")

    beam = cp.asarray(d0[np.newaxis])
    out_auto = gpu_beam_interpolation(
        beam,
        [daz],
        [dza],
        [azmin],
        cp.asarray(AZ.flatten()),
        cp.asarray(ZA.flatten()),
    ).get()
    out_explicit = gpu_beam_interpolation(
        beam,
        [daz],
        [dza],
        [azmin],
        cp.asarray(AZ.flatten()),
        cp.asarray(ZA.flatten()),
        power_beam=is_power,
    ).get()
    np.testing.assert_array_equal(out_auto, out_explicit)


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


class TestBicubic:
    """Tests for the order=3 (bicubic B-spline) CUDA kernel.

    The kernel is checked against ``scipy.ndimage.map_coordinates(order=3,
    mode="nearest")``, which is the interpolant matvis's cubic path is defined
    to reproduce. Note that ``mode`` only matters *outside* the beam grid -- for
    any coordinate inside it, every scipy mode gives the same answer, so these
    comparisons also hold against scipy's default ``mode="constant"``.
    """

    def test_matches_scipy_at_interior_points(self, efield_beam_1freq):
        """The bicubic kernel must reproduce scipy's cubic spline interpolation.

        Evaluated at random points well inside the grid and offset from the
        native nodes, where a cubic spline differs from both linear
        interpolation and from the raw grid values.
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)

        rng = np.random.default_rng(42)
        az = az_nodes[4:-4:7] + rng.uniform(0, daz, size=len(az_nodes[4:-4:7]))
        za = za_nodes[4:-4:5] + rng.uniform(0, dza, size=len(za_nodes[4:-4:5]))
        AZ, ZA = np.meshgrid(az, za, indexing="xy")

        out = _cubic_interp(d0, daz, dza, azmin, AZ.flatten(), ZA.flatten())
        ref = _scipy_reference(d0, daz, dza, azmin, AZ.flatten(), ZA.flatten(), order=3)

        np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)

    def test_differs_from_linear(self, efield_beam_1freq):
        """Cubic must actually be doing something different from the bilinear path.

        A guard against the order=3 request being silently serviced by the
        order=1 kernel (which would make every other comparison here pass
        trivially at grid nodes but be wrong in between).
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
        az = az_nodes[20:30] + 0.5 * daz
        za = za_nodes[20:30] + 0.5 * dza

        cubic = _cubic_interp(d0, daz, dza, azmin, az, za)
        linear = gpu_beam_interpolation(
            cp.asarray(d0[np.newaxis]),
            [daz],
            [dza],
            [azmin],
            cp.asarray(az),
            cp.asarray(za),
            order=1,
        ).get()[0]

        assert np.abs(cubic - linear).max() > 1e-6 * np.abs(linear).max()

    def test_reproduces_input_at_grid_nodes(self, efield_beam_1freq):
        """Cubic B-spline interpolation is interpolating: at nodes it returns the data.

        This is the property the prefilter exists to provide -- evaluating the
        B-spline basis against the *raw* grid values (i.e. forgetting to
        prefilter) would smooth the data by the 1/6, 4/6, 1/6 kernel instead.
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
        nax, nfeed, nza, naz = d0.shape
        AZ, ZA = np.meshgrid(az_nodes, za_nodes, indexing="xy")

        out = _cubic_interp(d0, daz, dza, azmin, AZ.flatten(), ZA.flatten())
        expected = d0.transpose(1, 0, 2, 3).reshape(nfeed, nax, nza * naz)

        np.testing.assert_allclose(out, expected, atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize("corner", ["low", "high"])
    def test_matches_scipy_inside_edge_cells(self, corner, efield_beam_1freq):
        """Points in the first/last grid cell exercise the stencil's halo.

        The 4-point cubic stencil reaches one node beyond each edge of the
        grid, so these points -- unlike interior ones -- depend on how the
        spline coefficients are extended past the boundary. They are the most
        likely place for an off-by-one in the halo to show up, and they are
        physically relevant: za=0 is the zenith and az wraps at the grid edge.
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)

        # Sub-cell offsets spanning the first (or last) cell, including both
        # endpoints exactly, where floor() lands on the boundary node itself.
        frac = np.array([0.0, 1e-6, 0.25, 0.5, 0.75, 1.0 - 1e-6, 1.0])
        if corner == "low":
            az = az_nodes[0] + frac * daz
            za = za_nodes[0] + frac * dza
        else:
            az = az_nodes[-2] + frac * daz
            za = za_nodes[-2] + frac * dza
        AZ, ZA = np.meshgrid(az, za, indexing="xy")

        out = _cubic_interp(d0, daz, dza, azmin, AZ.flatten(), ZA.flatten())
        ref = _scipy_reference(d0, daz, dza, azmin, AZ.flatten(), ZA.flatten(), order=3)

        np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)

    def test_exact_far_corner_node(self, efield_beam_1freq):
        """The very last node must not read past the end of the coefficient array.

        ``floor(n-1) == n-1`` would put the stencil's last tap at index ``n+1``,
        one beyond the halo; the kernel avoids this by clamping the cell index
        the same way the bilinear kernel does. Checked separately from the
        edge-cell sweep because it is the single coordinate where that clamp
        is load-bearing.
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
        nax, nfeed = d0.shape[:2]

        az = np.array([az_nodes[-1], az_nodes[-1], az_nodes[0]])
        za = np.array([za_nodes[-1], za_nodes[0], za_nodes[-1]])

        out = _cubic_interp(d0, daz, dza, azmin, az, za)
        expected = np.array(
            [d0[:, :, -1, -1], d0[:, :, 0, -1], d0[:, :, -1, 0]]
        ).transpose(2, 1, 0)

        np.testing.assert_allclose(out, expected, atol=1e-9, rtol=1e-9)

    def test_out_of_bounds_clamps_to_boundary(self, efield_beam_1freq):
        """Out-of-range points clamp to the grid edge, as the bilinear kernel does.

        This is a deliberate departure from scipy, whose ``mode="nearest"``
        interpolates through a 12-node edge-replicated pad (so it rings
        slightly just outside the grid) before saturating. matvis clamps
        immediately, which keeps the two orders consistent with each other and
        keeps sub-horizon sources pinned to the horizon value.
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
        nza = d0.shape[2]

        az = az_nodes[10:15]
        for za_val, za_idx in [
            (za_nodes[0] - 10 * dza, 0),
            (za_nodes[-1] + 10 * dza, nza - 1),
        ]:
            out = _cubic_interp(d0, daz, dza, azmin, az, np.full_like(az, za_val))
            expected = d0[:, :, za_idx, 10:15].transpose(1, 0, 2)
            np.testing.assert_allclose(out, expected, atol=1e-9, rtol=1e-9)

    def test_power_beam_matches_scipy(self, power_beam_1freq):
        """Real power beams interpolate in power and are square-rooted afterwards.

        Same convention as the order=1 path, and the same order of operations
        as the map_coordinates fallback, so the reference is sqrt(scipy(...)).
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(power_beam_1freq)

        rng = np.random.default_rng(7)
        az = az_nodes[4:-4:11] + rng.uniform(0, daz, size=len(az_nodes[4:-4:11]))
        za = za_nodes[4:-4:9] + rng.uniform(0, dza, size=len(za_nodes[4:-4:9]))
        AZ, ZA = np.meshgrid(az, za, indexing="xy")

        out = _cubic_interp(
            d0, daz, dza, azmin, AZ.flatten(), ZA.flatten(), power_beam=True
        )
        ref = np.sqrt(
            _scipy_reference(d0, daz, dza, azmin, AZ.flatten(), ZA.flatten(), order=3)
        )

        assert out.dtype == np.complex128
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)

    def test_single_precision_matches_scipy(self, efield_beam_1freq):
        """The complex64 kernel must agree with the float64 scipy reference.

        Only to single-precision tolerance, but this checks the separate
        complex64 kernel instantiation is wired up and that prefiltering in
        double before casting down keeps the result at full float32 accuracy.
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)

        rng = np.random.default_rng(3)
        az = az_nodes[4:-4:13] + rng.uniform(0, daz, size=len(az_nodes[4:-4:13]))
        za = za_nodes[4:-4:11] + rng.uniform(0, dza, size=len(za_nodes[4:-4:11]))
        AZ, ZA = np.meshgrid(az, za, indexing="xy")

        out = gpu_beam_interpolation(
            prefilter_beam(d0[np.newaxis].astype(np.complex64)),
            [daz],
            [dza],
            [azmin],
            cp.asarray(AZ.flatten()),
            cp.asarray(ZA.flatten()),
            order=3,
        ).get()[0]
        ref = _scipy_reference(d0, daz, dza, azmin, AZ.flatten(), ZA.flatten(), order=3)

        assert out.dtype == np.complex64
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-4)

    def test_multiple_beams_use_their_own_grids(self, efield_beam_1freq):
        """Each beam must be prefiltered and addressed with its own grid spacing.

        The second beam holds different data *and* declares a different grid
        spacing and azimuth origin. A kernel that indexed every beam with beam
        0's spacing, or a prefilter that mixed coefficients across the beam
        axis, would fail here but pass every single-beam test above.
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
        d1 = np.ascontiguousarray(d0[:, :, ::-1] * 0.5)
        # Grid metadata is independent of the data, so just declare a different
        # (still regular) grid for the second beam rather than resampling it.
        daz1, dza1, azmin1 = 0.7 * daz, 1.3 * dza, azmin + 0.05

        rng = np.random.default_rng(11)
        # Sources kept inside both declared grids, so neither beam's answer is
        # a clamped edge value.
        az = azmin1 + rng.uniform(0, 0.7 * daz * (len(az_nodes) - 1), size=40)
        za = rng.uniform(0, dza * (len(za_nodes) - 1), size=40)

        out = gpu_beam_interpolation(
            prefilter_beam(np.stack([d0, d1])),
            [daz, daz1],
            [dza, dza1],
            [azmin, azmin1],
            cp.asarray(az),
            cp.asarray(za),
            order=3,
        ).get()

        np.testing.assert_allclose(
            out[0],
            _scipy_reference(d0, daz, dza, azmin, az, za, order=3),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            out[1],
            _scipy_reference(d1, daz1, dza1, azmin1, az, za, order=3),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_raw_grid_input_is_prefiltered_internally(self, efield_beam_1freq):
        """Handing over raw grid values must give the same answer, just more slowly.

        ``GPUBeamInterpolator`` prefilters once during setup and passes the
        resulting ``BeamCoefficients`` on every chunk, but the standalone
        function has to stay correct when handed a bare beam grid.
        """
        d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
        az = az_nodes[5:-5:23] + 0.3 * daz
        za = za_nodes[5:-5:19] + 0.7 * dza
        AZ, ZA = np.meshgrid(az, za, indexing="xy")

        out = gpu_beam_interpolation(
            cp.asarray(d0[np.newaxis]),
            [daz],
            [dza],
            [azmin],
            cp.asarray(AZ.flatten()),
            cp.asarray(ZA.flatten()),
            order=3,
        ).get()[0]
        ref = _scipy_reference(d0, daz, dza, azmin, AZ.flatten(), ZA.flatten(), order=3)

        np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)

    def test_prefiltered_shape_has_halo(self, efield_beam_1freq):
        """``prefilter_beam`` grows only the two grid axes, by one node per side.

        ``grid_shape`` must undo that growth exactly -- it is what the kernel
        is told the beam grid's extent is, so an error here would silently
        shift every interpolated coordinate.
        """
        d0, *_ = _grid(efield_beam_1freq)
        coeff = prefilter_beam(d0[np.newaxis])

        nax, nfeed, nza, naz = d0.shape
        assert coeff.coeffs.shape == (1, nax, nfeed, nza + 2, naz + 2)
        assert coeff.coeffs.dtype == d0.dtype
        assert coeff.grid_shape == (nza, naz)

    def test_coefficients_rejected_at_other_orders(self, efield_beam_1freq):
        """Coefficients must not be silently accepted by a different order's kernel.

        Feeding a halo-carrying coefficient array to the bilinear kernel would
        interpolate the wrong array over the wrong grid extent and return
        plausible-looking nonsense rather than failing.
        """
        d0, daz, dza, azmin, *_ = _grid(efield_beam_1freq)

        with pytest.raises(ValueError, match="order-3 spline coefficients"):
            gpu_beam_interpolation(
                prefilter_beam(d0[np.newaxis]),
                [daz],
                [dza],
                [azmin],
                cp.asarray(np.zeros(3)),
                cp.asarray(np.zeros(3)),
                order=1,
            )

    def test_prefilter_beam_rejects_other_orders(self):
        """``prefilter_beam`` is specific to the cubic kernel and should say so."""
        with pytest.raises(ValueError, match="only supports order=3"):
            prefilter_beam(np.zeros((1, 1, 1, 4, 4)), order=5)

    @pytest.mark.parametrize("polarized", [True, False])
    def test_end_to_end_matches_cpu(self, polarized):
        """A full cubic simulation on the GPU must match the same one on the CPU.

        The unit tests above drive ``gpu_beam_interpolation`` directly with an
        explicitly prefiltered beam. This covers the other half: that
        ``GPUBeamInterpolator.setup`` notices ``order=3``, prefilters once, and
        then hands the resulting coefficients over on every chunk. Getting any
        of that wrong would leave the interpolation
        subtly over-smoothed rather than obviously broken, which is exactly the
        kind of error a visibility-level comparison catches.
        """
        kw, *_ = get_standard_sim_params(
            use_analytic_beam=False, polarized=polarized, nsource=250
        )
        kw |= {"precision": 2, "beam_spline_opts": {"order": 3}}

        vis_cpu = simulate_vis(use_gpu=False, **kw)
        vis_gpu = simulate_vis(use_gpu=True, **kw)

        np.testing.assert_allclose(vis_gpu, vis_cpu, rtol=1e-4, atol=5e-4)


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


@pytest.mark.parametrize("order", sorted(_KERNEL_ORDERS))
def test_fused_kernels_reject_other_modes(order, efield_beam_1freq):
    """The fused kernels implement mode="nearest" only, and must say so.

    Silently ignoring the request would be worse than refusing it: for
    ``order >= 2`` the mode changes the prefilter, so the caller would get
    values that differ from the mode they asked for *inside* the grid, not just
    beyond its edges.
    """
    d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
    beam = d0[np.newaxis] if order == 1 else prefilter_beam(d0[np.newaxis])

    with pytest.raises(ValueError, match='mode="nearest"'):
        gpu_beam_interpolation(
            beam,
            [daz],
            [dza],
            [azmin],
            cp.asarray(az_nodes[:4]),
            cp.asarray(za_nodes[:4]),
            order=order,
            mode="constant",
        )


@pytest.mark.parametrize("order", sorted(_KERNEL_ORDERS))
def test_setup_rejects_other_modes_before_uploading_beams(order, uvbeam):
    """An unsupported mode is caught at setup, not on the first source chunk."""
    from pyuvdata.beam_interface import BeamInterface

    from matvis.gpu.beams import GPUBeamInterpolator

    bm = GPUBeamInterpolator(
        beam_list=[BeamInterface(uvbeam.select(freq_chans=[0], inplace=False))],
        beam_idx=np.zeros(1, dtype=int),
        polarized=True,
        nant=1,
        freq=100e6,
        nsrc=10,
        spline_opts={"order": order, "mode": "mirror"},
        precision=2,
    )
    with pytest.raises(ValueError, match='mode="nearest"'):
        bm.setup()


def test_fallback_honours_the_requested_mode(efield_beam_1freq):
    """Orders without a fused kernel pass `mode` through to map_coordinates.

    Checked just outside the grid, where the modes are unambiguously different:
    "constant" returns zero, "nearest" returns the edge value.
    """
    d0, daz, dza, azmin, az_nodes, za_nodes = _grid(efield_beam_1freq)
    az = cp.asarray(az_nodes[:1])
    za = cp.asarray(za_nodes[:1] - 2 * dza)  # one grid cell below the first node

    kw = {"order": 2}
    nearest = gpu_beam_interpolation(
        d0[np.newaxis], [daz], [dza], [azmin], az, za, mode="nearest", **kw
    ).get()
    constant = gpu_beam_interpolation(
        d0[np.newaxis], [daz], [dza], [azmin], az, za, mode="constant", **kw
    ).get()

    assert np.all(constant == 0)
    assert not np.allclose(nearest, constant)
