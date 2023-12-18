"""Core abstract class for obtaining the Z matrix."""
import numpy as np


class ZMatrixCalc:
    r"""
    Class for computing the Z matrix.

    The Z matrix is defined as:

    .. math::

            Z = A I \exp(tau)

    where A is the beam, I is the square root of the flux, and tau is the phase.
    """

    def __init__(self, nant: int, nfeed: int, nax: int, nsrc: int, ctype):
        self.nant = nant
        self.nfeed = nfeed
        self.nax = nax
        self.nsrc = nsrc
        self.ctype = ctype

    def setup(self):
        """Perform any necessary setup steps.

        Accepts no inputs and returns nothing.
        """
        self.z = np.zeros(
            (self.nfeed * self.nant, self.nax * self.nsrc), dtype=self.ctype
        )

    def compute(
        self,
        sqrt_flux: np.ndarray,
        beam: np.ndarray,
        exptau: np.ndarray,
        beam_idx: np.ndarray,
    ) -> np.ndarray:
        """Compute the Z matrix.

        Z = A * I * exp(tau)

        Parameters
        ----------
        sqrt_flux
            Square root of the flux. Shape=(Nsrcs,).
        beam
            Beam. Shape=(Nbeams, Nfeed, Nax, Nsrcs).
        exptau
            Complex exponential of the delay (i.e. exp(-2π*i*nu*D.X)).
            Shape=(Nant, Nsrcs).
        beam_idx
            The beam indices, i.e. the beam index that each antenna corresponds to.

        Returns
        -------
        Z
            The Z matrix. Shape=(Nfeed*Nant, Nax*Nsrcs).
        """
        exptau *= sqrt_flux

        self.z.shape = (self.nant, self.nfeed, self.nax, self.nsrc)

        for fd in range(self.nfeed):
            for ax in range(self.nax):
                self.z[:, fd, ax, :] = exptau

        self.z *= beam[beam_idx]
        # self.z *= sqrt_flux

        # Here we expand the beam to all ants (from its beams), then broadcast to
        # the shape of exptau, so we end up with shape (Nant, Nfeed, Nax, Nsources)
        #        v = beam[beam_idx] * exptau[:, None, None, :]
        #       nfeed, nant, nax, nsrcs = v.shape
        self.z.shape = (self.nant * self.nfeed, self.nax * self.nsrc)


#        return v.reshape((nant * nfeed, nax * nsrcs))  # reform into matrix
