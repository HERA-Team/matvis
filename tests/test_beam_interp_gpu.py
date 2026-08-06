"""Test the GPU beam interpolation routine."""

import pytest

pytest.importorskip("cupy")

import cupy as cp
import numpy as np

from matvis.gpu.beams import gpu_beam_interpolation


def test_identity():
    """Test a simple case in which all interpolation points are nodes."""
    # Create a small and simple beam with Nax=1, Nfeed=1
    za = np.linspace(0, 1, 3)
    az = np.linspace(0, 2 * np.pi, 6)

    AZ, ZA = np.meshgrid(az, za, indexing="xy")
    beam = np.array([1 - ZA**2, 2 - ZA**2 + np.sin(AZ)])
    beam = beam[:, np.newaxis, np.newaxis]  # give it nax=1, nfeed=1
    dza = za[1] - za[0]
    daz = az[1] - az[0]

    # output shape is (nbeam, nfeed, nax, nsrc)
    new_beam = gpu_beam_interpolation(
        beam, [daz] * 2, [dza] * 2, [0.0] * 2, AZ.flatten(), ZA.flatten()
    ).get()

    for i, (b1, b2) in enumerate(zip(beam[:, 0, 0], new_beam[:, 0, 0], strict=True)):
        print(f"Beam {i}")
        np.testing.assert_allclose(np.sqrt(b1.flatten()), b2)

    assert np.allclose(new_beam[0, 0, 0, 0], 1)
    assert np.allclose(new_beam[0, 0, 0, -1], 0)
    assert np.allclose(new_beam[1, 0, 0, -1], 1)


def test_non_identity_linear():
    """Test a simple linear "beam" at non-nodal points."""
    # Create a small and simple beam with Nax=1, Nfeed=1
    za = np.linspace(0, 1, 5)
    az = np.linspace(0, 1, 5)

    AZ, ZA = np.meshgrid(az, za, indexing="xy")
    beam = np.array([1 - ZA])
    beam = beam[:, np.newaxis, np.newaxis]  # give it nax=1, nfeed=1

    dec_beam = beam[:, :, :, ::2][..., ::2]

    dza = za[1] - za[0]
    daz = az[1] - az[0]

    new_beam = gpu_beam_interpolation(
        dec_beam,
        [daz * 2],
        [dza * 2],
        [0.0],
        AZ.flatten(),
        ZA.flatten(),
    ).get()

    for i, (b1, b2) in enumerate(zip(beam[:, 0, 0], new_beam[:, 0, 0], strict=True)):
        print(f"Beam {i}", b2)
        assert np.allclose(np.sqrt(b1.flatten()), b2)


def test_wrong_beamtype():
    """Tests that a meaningful error is raised for a dumb beamtype."""
    # Create a small and simple beam with Nax=1, Nfeed=1
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


def test_order_gt_1_uses_map_coordinates_fallback():
    """Order != 1 must fall back to (and correctly use) cupyx map_coordinates.

    The fused bilinear kernel only handles order=1; higher orders go through
    the original per-plane ``map_coordinates`` loop. At grid nodes, a spline
    of any order should exactly reproduce the input values, same as order=1.
    """
    za = np.linspace(0, 1, 5)
    az = np.linspace(0, 2 * np.pi, 8)

    AZ, ZA = np.meshgrid(az, za, indexing="xy")
    beam = np.array([1 - ZA**2])
    beam = beam[:, np.newaxis, np.newaxis]  # nax=1, nfeed=1
    dza = za[1] - za[0]
    daz = az[1] - az[0]

    new_beam = gpu_beam_interpolation(
        beam, [daz], [dza], [0.0], AZ.flatten(), ZA.flatten(), order=2
    ).get()

    np.testing.assert_allclose(
        np.sqrt(beam[0, 0, 0].flatten()), new_beam[0, 0, 0], atol=1e-6
    )


def test_order_gt_1_with_complex_beam():
    """Order != 1 with an already-complex beam must skip the power-beam sqrt."""
    za = np.linspace(0, 1, 5)
    az = np.linspace(0, 2 * np.pi, 8)

    AZ, ZA = np.meshgrid(az, za, indexing="xy")
    beam_real = 1 - ZA**2
    beam = np.array([beam_real + 1j * beam_real]).astype(np.complex128)
    beam = beam[:, np.newaxis, np.newaxis]  # (nbeam=1, nax=1, nfeed=1, nza, naz)
    dza = za[1] - za[0]
    daz = az[1] - az[0]

    new_beam = gpu_beam_interpolation(
        beam, [daz], [dza], [0.0], AZ.flatten(), ZA.flatten(), order=2
    ).get()

    np.testing.assert_allclose(beam[0, 0, 0].flatten(), new_beam[0, 0, 0], atol=1e-6)


def test_complex_beam_with_mismatched_output_dtype():
    """A complex beam interpolated into a differently-typed output buffer.

    This forces the fused kernel to write into a scratch buffer of the beam's
    own dtype and cast on copy, without applying sqrt (the beam is already
    complex, e.g. an E-field beam, not a power beam).
    """
    za = np.linspace(0, 1, 3)
    az = np.linspace(0, 2 * np.pi, 6)
    AZ, ZA = np.meshgrid(az, za, indexing="xy")

    beam_real = 1 - ZA**2
    beam = np.array([beam_real + 1j * beam_real]).astype(np.complex128)
    beam = beam[:, np.newaxis, np.newaxis]  # (nbeam=1, nax=1, nfeed=1, nza, naz)
    dza = za[1] - za[0]
    daz = az[1] - az[0]

    az_flat, za_flat = AZ.flatten(), ZA.flatten()
    out = cp.zeros((1, 1, 1, az_flat.size), dtype=np.complex64)

    new_beam = gpu_beam_interpolation(
        beam, [daz], [dza], [0.0], az_flat, za_flat, beam_at_src=out
    ).get()

    assert new_beam.dtype == np.complex64
    np.testing.assert_allclose(
        new_beam[0, 0, 0], beam[0, 0, 0].flatten().astype(np.complex64), atol=1e-6
    )
