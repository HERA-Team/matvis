"""CPU-based source-summing operations."""

import numpy as np

from ..core.matprod import MatProd


class CPUMatMul(MatProd):
    """Use simple numpy.dot to perform the source-summing operation."""

    def compute(self, z: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nfeed, Nant, Nax, Nsrc).
        out
            Output array, shaped as (Nfeed, Nfeed, Npairs).
        """
        v = z.conj().dot(z.T)

        # Separate feed/ant axes to make indexing easier
        v.shape = (self.nant, self.nfeed, self.nant, self.nfeed)
        v = v.transpose((0, 2, 3, 1))  # transpose always returns a view

        if self.all_pairs:
            out[:] = v.reshape((self.nant * self.nant, self.nfeed, self.nfeed))
        else:
            out[:] = v[self.ant1_idx, self.ant2_idx]

        return out


class CPUVectorDot(MatProd):
    """Use a loop over specific pairs, performing a vdot over the source axis."""

    def compute(self, z: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nfeed, Nant, Nax, Nsrc).
        out
            Output array, shaped as (Nfeed, Nfeed, Npairs).
        """
        z = z.reshape((self.nant, self.nfeed, -1))

        for i, (ai, aj) in enumerate(self.antpairs):
            out[i] = z[aj].dot(z[ai].conj().T)  # dot(z[aj].T)

        return out


class CPUMatChunk(MatProd):
    """Loop over a small set of sub-matrix products which collectively contain all nont-redundant pairs."""

    def compute(self, z: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nfeed, Nant, Nax, Nsrc).
        out
            Output array, shaped as (Nfeed, Nfeed, Npairs).
        """
        z = z.reshape((self.nant, self.nfeed, -1))

        mat_product = np.zeros(
            (self.nant, self.nant, self.nfeed, self.nfeed), dtype=z.dtype
        )

        # Chris 12/20/23: instead we will use matsets
        for j in range(self.nfeed):
            for k in range(self.nfeed):
                for i, (ai, aj) in enumerate(self.matsets):
                    AI, AJ = np.meshgrid(ai, aj)
                    mat_product[AI, AJ, j, k] = z[ai[:], j].conj().dot(z[aj[:], k].T).T

        # Now, we need to identify the non-redundant pairs and put them into the final output array
        for j in range(self.nfeed):
            for k in range(self.nfeed):
                for i, (ai, aj) in enumerate(self.antpairs):
                    out[i, j, k] = mat_product[ai, aj, j, k]

        return out
