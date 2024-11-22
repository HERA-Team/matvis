"""Test the GPU beam interpolation routine."""

import pytest

pytest.importorskip("cupy")

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

    for i, (b1, b2) in enumerate(zip(beam[:, 0, 0], new_beam[:, 0, 0])):
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

    for i, (b1, b2) in enumerate(zip(beam[:, 0, 0], new_beam[:, 0, 0])):
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
