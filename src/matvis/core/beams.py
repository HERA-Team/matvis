"""Functions for working with beams."""
import numpy as np
from abc import ABC, abstractmethod
from pyuvdata import UVBeam
from typing import Any


def _wrangle_beams(
    beam_idx: np.ndarray | None,
    beam_list: list[UVBeam],
    polarized: bool,
    nant: int,
    freq: float,
) -> tuple[list[UVBeam], int, np.ndarray]:
    """Perform all the operations and checks on the input beams.

    Checks that the beam indices match the number of antennas, pre-interpolates to the
    given frequency, and checks that the beam type is appropriate for the given
    polarization

    Parameters
    ----------
    beam_idx
        Index of the beam to use for each antenna.
    beam_list
        List of unique beams.
    polarized
        Whether to use beam polarization
    nant
        Number of antennas
    freq
        Frequency to interpolate beam to.
    """
    # Get the number of unique beams
    nbeam = len(beam_list)

    # Check the beam indices
    if beam_idx is None and nbeam not in (1, nant):
        raise ValueError(
            "If number of beams provided is not 1 or nant, beam_idx must be provided."
        )
    if beam_idx is not None:
        assert beam_idx.shape == (nant,), "beam_idx must be length nant"
        assert all(
            0 <= i < nbeam for i in beam_idx
        ), "beam_idx contains indices greater than the number of beams"

    for bm in beam_list:
        if isinstance(bm, UVBeam) and not bm.future_array_shapes:
            bm.use_future_array_shapes()

    # make sure we interpolate to the right frequency first.
    beam_list = [
        bm.interp(freq_array=np.array([freq]), new_object=True, run_check=False)
        if isinstance(bm, UVBeam)
        else bm
        for bm in beam_list
    ]

    if polarized and any(b.beam_type != "efield" for b in beam_list):
        raise ValueError("beam type must be efield if using polarized=True")
    elif not polarized and any(
        (
            b.beam_type != "power"
            or getattr(b, "Npols", 1) > 1
            or b.polarization_array[0] not in [-5, -6]
        )
        for b in beam_list
    ):
        raise ValueError(
            "beam type must be power and have only one pol (either xx or yy) if polarized=False"
        )

    return beam_list, nbeam, beam_idx


class BeamInterpolator(ABC):
    """Base class for beam-interpolation methods.

    This class has a set __init__ method but provides two abstract methods:
    setup and interp. The setup method is called once -- outside the main time loop
    of the simulation -- and the interp method is called once per time step.

    Parameters
    ----------
    beam_list
        List of unique beams.
    beam_idx
        Index of the beam to use for each antenna.
    polarized
        Whether to use beam polarization
    nant
        Number of antennas
    freq
        Frequency to interpolate beam to.
    spline_opts
        A dictionary of options to send to the spline interpolation method.
    precision
        The precision of the data (1 or 2).
    """

    def __init__(
        self,
        beam_list: list[UVBeam],
        beam_idx: np.ndarray,
        polarized: bool,
        nant: int,
        freq: float,
        nsrc: int,
        spline_opts: dict | None = None,
        precision: int = 1,
    ):
        self.polarized = polarized
        self.beam_list, self.nbeam, self.beam_idx = _wrangle_beams(
            beam_idx=beam_idx,
            beam_list=beam_list,
            polarized=polarized,
            nant=nant,
            freq=freq,
        )
        self.polarized = polarized
        self.nant = nant
        self.freq = freq
        self.spline_opts = spline_opts or {}

        if self.polarized:
            self.nfeed = 2
            self.nax = 2
        else:
            self.nfeed = 1
            self.nax = 1

        if precision == 1:
            self.complex_dtype = np.complex64
            self.real_dtype = np.float32
        elif precision == 2:
            self.complex_dtype = np.complex128
            self.real_dtype = np.float64

        self.nsrc = nsrc

    def setup(self):  # noqa: B027
        """Perform any necessary setup steps.

        Accepts no inputs and returns nothing.
        """
        self.interpolated_beam = np.zeros(
            (self.nbeam, self.nfeed, self.nax, self.nsrc), dtype=self.complex_dtype
        )

    @abstractmethod
    def interp(self, tx: np.ndarray, ty: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Perform the beam interpolation.

        This method must return an array of shape ``(nbeam, nfeed, nax, nsrcs_up)``.
        """
        pass

    def __call__(
        self, tx: np.ndarray, ty: np.ndarray, check: bool = True
    ) -> np.ndarray:
        """Perform the beam interpolation.

        Parameters
        ----------
        tx, ty
            Coordinates to evaluate the beam at, in sin-projection.
        check
            Whether to check the output for invalid values.

        Returns
        -------
        out
            The interpolated beam values, shape (nbeam, nfeed, nax, nsrc)
        """
        self.interp(tx, ty, self.interpolated_beam)
        if check:
            # Check for invalid beam values
            sm = self.interpolated_beam.sum()
            if np.isinf(sm) or np.isnan(sm):
                raise ValueError("Beam interpolation resulted in an invalid value")

        return self.interpolated_beam
