"""GPU implementation for obtaining the Z matrix."""
import cupy as cp

from ..core.getz import ZMatrixCalc


class GPUZMatrixCalc(ZMatrixCalc):
    """GPU implementation for obtaining the Z matrix."""

    def setup(self):
        """Perform any necessary setup steps.

        Accepts no inputs and returns nothing.
        """
        self.z = cp.zeros(
            (self.nfeed * self.nant, self.nax * self.nsrc), dtype=self.ctype
        )

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
        cp.cuda.Device().synchronize()
