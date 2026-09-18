"""Global configuration for pytest."""

import os

import pytest

# Restrict UCX to shared-memory and loopback transports before anything can
# initialise MPI.
#
# pyuvsim uses mpi4py, and the MPICH builds it binds to are configured
# ``ch4:ucx``. At MPI_Init, UCX enumerates the host's network devices and tries
# to open a transport on each one. On the subset of Azure CI runners that expose
# an RDMA-capable MANA NIC, opening the verbs transport fails outright::
#
#     UCX  ERROR uct_iface_open(ud_verbs/mana_0:1) failed: Input/output error
#     Abort(404899215): Fatal error in internal_Init_thread: Other MPI error
#
# That aborts the interpreter from C, so the whole pytest session dies with a
# bare exit status and no traceback. Since these tests only ever run MPI with a
# single rank, no network transport is needed at all; allowing just ``self``,
# ``sm`` and ``tcp`` keeps UCX away from the verbs devices that fail. Set with
# ``setdefault`` so an explicit UCX_TLS from the environment still wins.
os.environ.setdefault("UCX_TLS", "self,sm,tcp")

from pyuvdata.uvbeam import UVBeam  # noqa: E402

from matvis import DATA_PATH  # noqa: E402


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
