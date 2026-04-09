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

            Z = A I \exp(tau)

    where A is the beam, I is the square root of the flux, and tau is the phase.
    """

    def __init__(
        self, nant: int, nfeed: int, nax: int, nsrc: int, ctype, gpu: bool = False
    ):
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
        sqrt_flux: np.ndarray,
        beam: np.ndarray,
        exptau: np.ndarray,
        beam_idx: np.ndarray | None,
        m_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the Z matrix.

        When m_matrix is None (default), computes the unpolarized case:
            Z = A * sqrt_flux * exp(tau)

        When m_matrix is provided, computes the full polarized case:
            Z[ant, fd, ax, src] = Σ_k beam[ant, fd, k, src] * exp(tau) * M[k, ax, src]

        Parameters
        ----------
        sqrt_flux
            Square root of the flux. Shape=(Nsrcs,). Ignored when m_matrix is provided.
        beam
            Beam. Shape=(Nbeams, Nfeed, Nax, Nsrcs).
        exptau
            Complex exponential of the delay (i.e. exp(-2π*i*nu*D.X)).
            Shape=(Nant, Nsrcs).
        beam_idx
            The beam indices, i.e. the beam index that each antenna corresponds to.
        m_matrix
            The M matrix from coherency decomposition C = M @ M†.
            Shape=(2, 2, Nsrcs). If None, uses the existing unpolarized path.

        Returns
        -------
        Z
            The Z matrix. Shape=(Nfeed*Nant, Nax*Nsrcs).
        """
        if m_matrix is not None:
            # Full polarized path: Z[ant, fd, ax, src] = Σ_k bm[ant, fd, k, src] * exptau[ant, src] * M[k, ax, src]
            # IMPORTANT: Do NOT mutate exptau here — callers may reuse the same
            # exptau buffer across multiple calls (e.g. sign-split).
            #
            # Manually unrolled k-contraction (k=0,1) with broadcasting.
            # This avoids Python loops and is ~1.6x faster than the loop version.
            xp = self.xp
            bm = beam[beam_idx] if beam_idx is not None else beam

            # bm[:,:,k,:] → (Nant, Nfeed, Nsrc), add ax dim → (Nant, Nfeed, 1, Nsrc)
            # m_matrix[k,:,:] → (2, Nsrc), add ant/feed dims → (1, 1, 2, Nsrc)
            b0 = bm[:, :, :1, :]
            b1 = bm[:, :, 1:2, :]
            m0 = m_matrix[xp.newaxis, xp.newaxis, 0, :, :]
            m1 = m_matrix[xp.newaxis, xp.newaxis, 1, :, :]

            # Z = (b0*M0 + b1*M1) * exptau — single vectorized expression
            self.z.shape = (self.nant, self.nfeed, self.nax, self.nsrc)
            self.z[:] = (b0 * m0 + b1 * m1) * exptau[:, xp.newaxis, xp.newaxis, :]
            self.z.shape = (self.nant * self.nfeed, self.nax * self.nsrc)
        else:
            # Existing unpolarized path — unchanged.
            exptau *= sqrt_flux

            self.z.shape = (self.nant, self.nfeed, self.nax, self.nsrc)

            for fd in range(self.nfeed):
                for ax in range(self.nax):
                    self.z[:, fd, ax, :] = exptau

            if beam_idx is None:
                self.z *= beam
            else:
                self.z *= beam[beam_idx]

            # Here we expand the beam to all ants (from its beams), then broadcast to
            # the shape of exptau, so we end up with shape (Nant, Nfeed, Nax, Nsources)
            self.z.shape = (self.nant * self.nfeed, self.nax * self.nsrc)

        if self.gpu:
            cp.cuda.Device().synchronize()

        return self.z
