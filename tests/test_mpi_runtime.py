"""Guard tests for the MPI runtime that ``pyuvsim`` requires.

``pyuvsim`` imports ``mpi4py``, and the ``mpi4py`` PyPI wheels do not bundle an
MPI implementation: they bind at import time to whatever ``libmpi`` the host
makes available. A broken or mismatched MPI stack therefore shows up as a
*fatal* ``MPI_Abort``, which calls ``exit()`` in C and tears the interpreter
down without unwinding Python. Under pytest's default file-descriptor capture
the abort diagnostic is written to a captured fd that is never drained, so the
whole run dies with a bare non-zero exit status and no traceback, no ``FAILED``
line and no summary -- which is very hard to diagnose after the fact.

These tests run MPI start-up in a *subprocess* so that such an abort is
reported as an ordinary test failure, with the MPI diagnostic attached, instead
of killing the pytest session.
"""

import subprocess
import sys

import pytest

# Long enough to absorb a slow cold start on a loaded CI runner, short enough
# that a genuinely hung MPI start-up still fails the job in reasonable time.
_TIMEOUT = 180

_INIT_SCRIPT = """
from mpi4py import MPI

print("vendor:", MPI.get_vendor())
print("library:", MPI.Get_library_version().strip().splitlines()[0])
assert MPI.COMM_WORLD.Get_size() >= 1
assert MPI.COMM_WORLD.Get_rank() == 0
MPI.COMM_WORLD.Barrier()
"""

_PYUVSIM_SCRIPT = """
from pyuvsim import mpi

mpi.start_mpi()
assert mpi.world_comm is not None
assert mpi.node_comm is not None
assert mpi.rank == 0
assert mpi.Npus >= 1
"""


def _run(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter and return the completed process.

    Parameters
    ----------
    script : str
        Python source to execute via ``python -c``.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process, with ``stdout`` and ``stderr`` decoded as text.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )


def _explain(what: str, proc: subprocess.CompletedProcess) -> str:
    """Build an assertion message describing a failed MPI subprocess.

    Parameters
    ----------
    what : str
        Short description of the operation that was attempted.
    proc : subprocess.CompletedProcess
        The completed process to describe.

    Returns
    -------
    str
        A message quoting the exit status and both captured streams.
    """
    return (
        f"{what} failed with exit status {proc.returncode}.\n"
        "A fatal MPI error aborts the interpreter without a Python traceback, "
        "so the MPI diagnostic below is the only clue.\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


@pytest.mark.filterwarnings("ignore")
def test_mpi_initializes_without_aborting():
    """MPI must initialise and run a collective without a fatal abort."""
    proc = _run(_INIT_SCRIPT)
    assert proc.returncode == 0, _explain("MPI initialisation", proc)
    assert "vendor:" in proc.stdout


@pytest.mark.filterwarnings("ignore")
def test_pyuvsim_start_mpi_succeeds():
    """``pyuvsim.mpi.start_mpi`` must set up its communicators without aborting.

    This is the exact start-up path taken by
    :func:`pyuvsim.uvsim.run_uvdata_uvsim`, which the comparison tests in
    ``test_compare_pyuvsim.py`` exercise.
    """
    proc = _run(_PYUVSIM_SCRIPT)
    assert proc.returncode == 0, _explain("pyuvsim MPI start-up", proc)
