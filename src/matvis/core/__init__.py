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
    times: np.ndarray,
    I_sky: np.ndarray | None = None,
    stokes: np.ndarray | None = None,
):
    """Validate input shapes and types.

    Exactly one of ``I_sky`` or ``stokes`` must be provided. ``nsrc`` is
    derived from whichever is given.
    """
    assert precision in {1, 2}

    # Specify number of polarizations (axes/feeds)
    if polarized:
        nax = nfeed = 2
    else:
        nax = nfeed = 1

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes = len(times)

    if (I_sky is None) == (stokes is None):
        raise ValueError("Provide exactly one of `I_sky` or `stokes`.")

    if stokes is not None:
        assert stokes.ndim == 2 and stokes.shape[0] == 4, (
            "stokes must have shape (4, NSRCS)."
        )
        nsrc = stokes.shape[1]
    else:
        assert I_sky.ndim == 1, "I_sky must have shape (NSRCS,)."
        nsrc = len(I_sky)

    return nax, nfeed, nant, ntimes, nsrc
