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
        _, nfeed, nax, _ = beam.shape
        nant, nsrc = exptau.shape

        exptau *= sqrt_flux
        beam = beam[beam_idx]
        beam *= exptau[:, None, None, :]
        out = beam.reshape((nant * nfeed, nax * nsrc))

        cp.cuda.Device().synchronize()
        return out
