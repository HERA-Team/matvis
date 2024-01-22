"""Module containing several options for computing sub-matrices for the MatChunk method."""
import numpy as np


def get_matrix_sets(bls, ndecimals: int = 2):
    """Find redundant baselines."""
    uvbins = set()
    msets = []

    # Everything here is in wavelengths
    bls = np.round(bls, decimals=ndecimals)
    nant = bls.shape[0]

    # group redundant baselines
    for i in range(nant):
        for j in range(i + 1, nant):
            u, v = bls[i, j]
            if (u, v) not in uvbins and (-u, -v) not in uvbins:
                uvbins.add((u, v))
                msets.append([np.array([i]), np.array([j])])

    return msets
