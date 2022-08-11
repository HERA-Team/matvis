"""Test the GPU beam interpolation routine."""
import pytest

pytest.importorskip("pycuda")

import numpy as np
from pathlib import Path
from pycuda.driver import Stream
from pyuvdata import UVBeam

from vis_cpu import DATA_PATH
from vis_cpu._uvbeam_to_raw import uvbeam_to_azza_grid
from vis_cpu.gpu import gpu_beam_interpolation


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

    print(beam[0].flatten())
    new_beam = gpu_beam_interpolation(beam, daz, dza, AZ.flatten(), ZA.flatten())

    for i, (b1, b2) in enumerate(zip(beam[:, 0, 0], new_beam[0, 0])):
        print(f"Beam {i}")
        assert np.allclose(np.sqrt(b1.flatten()), b2)

    assert np.allclose(new_beam[0, 0, 0, 0], 1)
    assert np.allclose(new_beam[0, 0, 0, -1], 0)

    assert np.allclose(new_beam[0, 0, 1, -1], 1)


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
        daz * 2,
        dza * 2,
        AZ.flatten(),
        ZA.flatten(),
        stream=Stream(),
    )

    for i, (b1, b2) in enumerate(zip(beam[:, 0, 0], new_beam[0, 0])):
        print(f"Beam {i}", b2)
        assert np.allclose(np.sqrt(b1.flatten()), b2)


@pytest.mark.parametrize("polarized", [True, False])
def test_identity_beamfile(polarized):
    """Test interpolation of a complicated beam file, at nodal points."""
    beam_file = DATA_PATH / "NF_HERA_Dipole_small.fits"
    beam = UVBeam()
    beam.read_beamfits(beam_file)
    beam = beam.interp(freq_array=np.array([1e8]), new_object=True, run_check=False)

    beam.interp(
        freq_array=beam.freq_array[0],
    )
    if not polarized:
        beam.efield_to_power(calc_cross_pols=False, inplace=True)
        beam.select(polarizations=["xx"], inplace=True)

    beam_raw, daz, dza = uvbeam_to_azza_grid(beam)

    naz = int(2 * np.pi / daz) + 1
    az = np.linspace(0, 2 * np.pi, naz)
    za = np.arange(0, np.pi / 2 + dza, dza)

    assert beam_raw.shape == (
        2 if polarized else 1,
        2 if polarized else 1,
        len(za),
        len(az),
    )

    # Shape (nbeam, nax, nfeed, nza, naz)
    beam_raw = np.array([beam_raw, 2 * beam_raw])

    AZ, ZA = np.meshgrid(az, za, indexing="xy")

    # Shape (nax, nfeed, nbeam, nsrc)

    new_beam = gpu_beam_interpolation(beam_raw, daz, dza, AZ.flatten(), ZA.flatten())

    if not polarized:
        beam_raw = np.sqrt(beam_raw)

    assert new_beam.dtype.name.startswith("complex")

    for (
        iax,
        axx,
    ) in enumerate(new_beam):
        for ifd, feed in enumerate(axx):
            for ibeam, bm in enumerate(feed):
                print(iax, ifd, ibeam)
                assert np.allclose(beam_raw[ibeam, iax, ifd].flatten(), bm)


@pytest.mark.parametrize("polarized", [True, False])
def test_non_identity_beamfile(polarized):
    """Test a complex beam file at non-nodal points."""
    beam_file = DATA_PATH / "NF_HERA_Dipole_small.fits"
    beam = UVBeam()
    beam.read_beamfits(beam_file)
    beam = beam.interp(freq_array=np.array([1e8]), new_object=True, run_check=False)

    beam.interp(
        freq_array=beam.freq_array[0],
    )
    if not polarized:
        beam.efield_to_power(calc_cross_pols=False, inplace=True)
        beam.select(polarizations=["xx"], inplace=True)

    beam_raw, daz, dza = uvbeam_to_azza_grid(beam)

    naz = int(2 * np.pi / daz) + 1
    az = np.linspace(0, 2 * np.pi, naz)
    za = np.arange(0, np.pi / 2 + dza, dza)

    raw_az = np.linspace(0, 2 * np.pi, naz)  # from uvbeam_to_raw
    raw_za = np.arange(0, np.pi / 2 + dza, dza)

    beam = beam.interp(raw_az, raw_za, az_za_grid=True, new_object=True)

    assert beam_raw.shape == (
        2 if polarized else 1,
        2 if polarized else 1,
        len(za),
        len(az),
    )

    # Shape (nbeam, nax, nfeed, nza, naz)
    beam_raw = np.array([beam_raw])

    AZ = np.linspace(0, 2 * np.pi, 10)
    ZA = np.linspace(0, np.pi / 2, 10)

    print("dZA", dza)
    print("dAZ", daz)

    # Shape (nax, nfeed, nbeam, nsrc)
    new_beam_gpu = gpu_beam_interpolation(beam_raw, daz, dza, AZ, ZA)
    new_beam_uvb = beam.interp(
        az_array=AZ, za_array=ZA, spline_opts={"kx": 1, "ky": 1}
    )[0]

    if not polarized:
        new_beam_uvb = np.sqrt(new_beam_uvb)

    for (
        iax,
        axx,
    ) in enumerate(new_beam_gpu):
        for ifd, feed in enumerate(axx):
            for ibeam, bm in enumerate(feed):
                print(iax, ifd, ibeam)
                np.testing.assert_allclose(
                    new_beam_uvb[iax, 0, ifd, 0].real, bm.real, rtol=1e-6
                )
                np.testing.assert_allclose(
                    new_beam_uvb[iax, 0, ifd, 0].imag, bm.imag, rtol=1e-6
                )


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
            dec_beam.astype(int), daz * 2, dza * 2, AZ.flatten(), ZA.flatten()
        )
