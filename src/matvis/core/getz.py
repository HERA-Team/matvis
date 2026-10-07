"""Core abstract class for obtaining the Z matrix."""

import numpy as np

try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False


class ZMatrixCalc:
    r"""
    Class for computing the Z matrix.

    The Z matrix is defined as:

    .. math::

            Z = A^* I \exp(tau)

    where A is the beam, I is the square root of the flux, and tau is the phase.
    The beam enters conjugated so that the source sum ``conj(Z_i) Z_j^T``
    computed by the matprod classes is ``A_i C A_j^H`` times the fringe term for
    feeds (p, q) of antennas (i, j), with A indexed [feed, sky component]. This
    is pyuvsim's convention.

    Parameters
    ----------
    antenna_order
        Optional length-``Nant`` permutation: row ``p`` of the returned ``Z`` is
        built for antenna ``antenna_order[p]`` rather than for antenna ``p``.
        Relabelling the antenna axis costs nothing here (it only changes which
        row of ``exptau`` and which beam each output row reads), but it lets the
        block-decomposed matprod classes slice their operands straight out of
        ``Z`` instead of gathering them -- see
        :func:`~matvis.redundancy.contiguity_order`. Whatever is passed here
        must also be passed to the matprod class, or the visibilities will be
        attributed to the wrong antennas.
    """

    def __init__(
        self,
        nant: int,
        nfeed: int,
        nax: int,
        nsrc: int,
        ctype,
        gpu: bool = False,
        antenna_order: np.ndarray | None = None,
    ):
        self.antenna_order = (
            None if antenna_order is None else np.asarray(antenna_order)
        )
        if self.antenna_order is not None and self.antenna_order.shape != (nant,):
            raise ValueError(
                f"antenna_order must have shape ({nant},), got "
                f"{self.antenna_order.shape}"
            )
        self.nant = nant
        self.nfeed = nfeed
        self.nax = nax
        self.nsrc = nsrc
        self.ctype = ctype

        self.gpu = gpu
        if gpu and not HAVE_CUDA:
            raise ImportError("You need to install the [gpu] extra to use gpu!")

        self.xp = cp if self.gpu else np

    def setup(self):
        """Perform any necessary setup steps.

        Accepts no inputs and returns nothing.
        """
        self.z = self.xp.full(
            (self.nfeed * self.nant, self.nax * self.nsrc),
            self.ctype(0.0),
            dtype=self.ctype,
        )

    def __call__(
        self,
        sqrt_flux: np.ndarray | None,
        beam: np.ndarray,
        exptau: np.ndarray,
        beam_idx: np.ndarray | None,
        m_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the Z matrix.

        Z = conj(A) * sqrt_flux * exp(tau), or conj(A @ M) * exp(tau).

        Parameters
        ----------
        sqrt_flux
            Square root of the flux. Shape=(Nsrcs,). May be None when m_matrix is given.
        beam
            Beam. Shape=(Nbeams, Nfeed, Nax, Nsrcs).
        exptau
            Complex exponential of the delay (i.e. exp(+2π*i*nu*D.X/c)).
            Shape=(Nant, Nsrcs).
        beam_idx
            The beam indices, i.e. the beam index that each antenna corresponds to.

        m_matrix
            Optional sky factor of shape (2, 2, Nsrcs), with C = M M†.
            When provided, sqrt_flux is ignored.

        Returns
        -------
        Z
            The Z matrix. Shape=(Nfeed*Nant, Nax*Nsrcs).
        """
        if m_matrix is not None:
            xp = self.xp
            src_ant = self.antenna_order
            rowbeam = xp.arange(self.nant) if beam_idx is None else xp.asarray(beam_idx)
            if src_ant is not None:
                rowbeam = rowbeam[src_ant]
            bm = beam if beam.shape[0] == 1 else beam[rowbeam]
            phase = exptau if src_ant is None else exptau[src_ant]
            # MatProd sums conj(Z_i) Z_j: conjugate the entire Jones-sky product,
            # including complex M, without changing the shared phase buffer.
            self.z = self.z.reshape(self.nant, self.nfeed, self.nax, self.nsrc)
            self.z[:] = (
                bm[:, :, :1, :] * m_matrix[None, None, 0]
                + bm[:, :, 1:2, :] * m_matrix[None, None, 1]
            ).conj()
            self.z *= phase[:, None, None, :]
            self.z = self.z.reshape(self.nant * self.nfeed, self.nax * self.nsrc)
            return self.z

        exptau *= sqrt_flux

        self.z = self.z.reshape(self.nant, self.nfeed, self.nax, self.nsrc)

        # Row p of z belongs to antenna src_ant[p]; the identity unless the
        # caller asked for a different antenna order (see the class docstring).
        src_ant = self.antenna_order

        # z is built as conj(conj(exptau) * A), which equals conj(A) * exptau
        # without allocating a conjugated copy of the beam.
        exptau_conj = exptau.conj()
        for fd in range(self.nfeed):
            for ax in range(self.nax):
                self.z[:, fd, ax, :] = (
                    exptau_conj if src_ant is None else exptau_conj[src_ant]
                )

        if beam.shape[0] == 1 or (beam_idx is None and src_ant is None):
            # A single shared beam broadcasts over the antenna axis; and with no
            # beam_idx and no reordering, `beam` is already one-per-antenna in
            # row order. Either way a plain broadcast multiply is correct.
            self.z *= beam
        else:
            # Which beam each *row* of z wants. Without beam_idx there is one
            # beam per antenna, so the beam index is just the antenna index.
            rowbeam = np.arange(self.nant) if beam_idx is None else beam_idx
            if src_ant is not None:
                rowbeam = rowbeam[src_ant]
            # Since rowbeam is an array of integers, using it as an index into beam
            # is "fancy indexing", which causes a memory copy. To avoid this, we loop
            # over the indices. While this might be a bit slower, it avoids the memory
            # copy and thus is more memory efficient.
            for ant, bmidx in enumerate(rowbeam):
                self.z[ant] *= beam[bmidx]

        self.xp.conj(self.z, out=self.z)

        # Here we expand the beam to all ants (from its beams), then broadcast to
        # the shape of exptau, so we end up with shape (Nant, Nfeed, Nax, Nsources)
        self.z = self.z.reshape(self.nant * self.nfeed, self.nax * self.nsrc)

        return self.z
