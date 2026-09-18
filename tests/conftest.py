"""Global configuration for pytest."""

import os

import pytest

# Restrict UCX to shared-memory/loopback transports before anything can
# initialise MPI: on runners exposing an RDMA-capable NIC, UCX fails to open
# the verbs transport and aborts the interpreter. These tests are single-rank,
# so no network transport is needed. See PR #159.
os.environ.setdefault("UCX_TLS", "self,sm,tcp")

from pyuvdata.uvbeam import UVBeam

from matvis import DATA_PATH


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
