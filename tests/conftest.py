"""Global configuration for pytest."""
import pytest

from pyuvdata.uvbeam import UVBeam

from vis_cpu import DATA_PATH


@pytest.fixture(scope="session")
def uvbeam():
    """Default CST UVBeam."""
    beam_file = DATA_PATH / "NF_HERA_Dipole_small.fits"
    beam = UVBeam()
    beam.read_beamfits(beam_file)
    return beam


@pytest.fixture(scope="session")
def uvbeam_unpol(uvbeam):
    """CST Beam made unpolarized."""
    beam = uvbeam.copy()
    beam.efield_to_power(calc_cross_pols=False, inplace=True)
    beam.select(polarizations=["xx"], inplace=True)
    return beam
