"""Core functionality of the matvis package.

This sub-package defines template routines for the core functionality of the
algorithm. These routines are then implemented in the ``cpu`` and ``gpu``
sub-packages.
"""
import numpy as np


def _validate_inputs(
    precision: int,
    polarized: bool,
    antpos: np.ndarray,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
):
    """Validate input shapes and types."""
    assert precision in {1, 2}

    # Specify number of polarizations (axes/feeds)
    if polarized:
        nax = nfeed = 2
    else:
        nax = nfeed = 1

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)."
    ncrd, nsrcs = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NSRCS)."
    assert (
        I_sky.ndim == 1 and I_sky.shape[0] == nsrcs
    ), "I_sky must have shape (NSRCS,)."

    return nax, nfeed, nant, ntimes
