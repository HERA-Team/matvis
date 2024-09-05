"""Core abstract class for computing exp(tau)."""

import numpy as np
from astropy.constants import c as speed_of_light

from .._utils import get_dtypes

try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False


class TauCalculator:
    """Core abstract class for computing exp(tau)."""

    def __init__(
        self,
        antpos: np.ndarray,
        freq: float,
        nsrc: int,
        precision: int = 1,
        gpu: bool = False,
    ):
        self.rtype, self.ctype = get_dtypes(precision)
        self.nant = len(antpos)
        self.nsrc = nsrc
        ang_freq = self.rtype(2.0 * np.pi * freq)
        self.antpos = antpos.astype(self.rtype) * ang_freq * 1j / speed_of_light.value

        self.gpu = gpu
        if gpu and not HAVE_CUDA:
            raise ImportError("You need to install the [gpu] extra to use gpu!")

    def setup(self):
        """Perform any necessary setup steps.

        Accepts no inputs and returns nothing.
        """
        self._xp = cp if self.gpu else np
        self.exptau = self._xp.full(
            (self.nant, self.nsrc), self.rtype(0.0), dtype=self.ctype
        )
        self.antpos = self._xp.asarray(self.antpos)

    def __call__(self, crdtop: np.ndarray) -> np.ndarray:
        """Compute the complex exponential of the delay.

        exp(-2π*i*nu*D.X)

        Parameters
        ----------
        antpos
            Antenna positions. Shape=(Nant, 3).
        crdtop
            Topocentric coordinates. Shape=(3, Nsrcs).

        Returns
        -------
        exptau
            The complex exponential of the delay. Shape=(Nant, Nsrcs).
        """
        self._xp.matmul(self.antpos, crdtop, out=self.exptau)
        self._xp.exp(self.exptau, out=self.exptau)

        if self.gpu:
            cp.cuda.Device().synchronize()

        return self.exptau
