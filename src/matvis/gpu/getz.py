"""GPU implementation for obtaining the Z matrix."""
import cupy as cp


class GPUZMatrixCalc:
    """GPU implementation for obtaining the Z matrix."""

    def compute(
        self,
        sqrt_flux: cp.ndarray,
        beam: cp.ndarray,
        exptau: cp.ndarray,
        beam_idx: cp.ndarray,
    ) -> cp.ndarray:
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
        nfeed, _, nax, _ = beam.shape
        nant, nsrc = exptau.shape

        Z = cp.empty(shape=(nfeed, nant, nax, nsrc), dtype=exptau.dtype)
        cp.multiply(exptau[None, :, None, :], beam[:, beam_idx], out=Z)
        Z *= sqrt_flux
        return Z.reshape((nfeed * nant, nax * nsrc))
