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
        v.shape = (self.nfeed, self.nant, self.nfeed, self.nant)
        v = v.transpose((0, 2, 1, 3))  # transpose always returns a view

        if self.all_pairs:
            out[:] = v.reshape((self.nfeed, self.nfeed, self.nant * self.nant))
        else:
            out[:] = v[:, :, self.ant1_idx, self.ant2_idx]

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
        z = z.reshape((self.nfeed, self.nant, -1))

        for i, (ai, aj) in enumerate(self.antpairs):
            out[:, :, i] = z[:, ai].conj().dot(z[:, aj].T)

        return out
