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
            Beam. Shape=(Nfeed, Nbeams, Nax, Nsrcs).
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

        # Here we expand the beam to all ants (from its beams), then broadcast to
        # the shape of exptau, so we end up with shape (Nfeed, Nant, Nax, Nsources)
        v = beam[:, beam_idx] * exptau[None, :, None, :]
        nfeed, nant, nax, nsrcs = v.shape
        return v.reshape((nfeed * nant, nax * nsrcs))  # reform into matrix
