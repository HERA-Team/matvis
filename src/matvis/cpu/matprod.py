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


class CPUMatBlock(MatProd):
    """Compute a set of rectangular antenna-block sub-matrix products.

    Instead of one full ``Nant x Nant`` product (:class:`CPUMatMul`) or a Python
    loop over individual pairs (:class:`CPUVectorDot`), this computes one smaller
    ``(len(rows)*Nfeed) x (len(cols)*Nfeed)`` product per ``antenna_blocks`` entry
    and gathers just the requested ``antpairs`` out of each block directly into
    the output -- no ``(Nant, Nant, Nfeed, Nfeed)`` intermediate is ever built.
    Useful when the requested ``antpairs`` are concentrated within a modest
    number of antenna groupings (e.g. redundant-baseline dedup, or an
    array with a compact core), trading a small amount of "wasted" computation
    on non-requested pairs inside a block for either fewer total FLOPs than the
    full product, or fewer/bigger BLAS calls than :class:`CPUVectorDot`.

    See :mod:`matvis.redundancy` for helpers that build ``antenna_blocks`` lists.
    """

    supports_blocks = True

    def compute(self, z: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nfeed*Nant, Nax*Nsrc).
        out
            Output array, shaped as (Npairs, Nfeed, Nfeed).
        """
        z = z.reshape((self.nant, self.nfeed, -1))

        for rows, cols, local_rows, local_cols, slots in self._block_plan:
            zr = z[rows].reshape(len(rows) * self.nfeed, -1)
            zc = z[cols].reshape(len(cols) * self.nfeed, -1)

            block = zr.conj().dot(zc.T)
            block.shape = (len(rows), self.nfeed, len(cols), self.nfeed)
            block = block.transpose((0, 2, 3, 1))  # -> (rows, cols, nfeed_j, nfeed_i)

            out[slots] = block[local_rows, local_cols]

        return out
