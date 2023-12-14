"""GPU beam interpolation routines."""
import cupy as cp
import itertools
import numpy as np
from cupyx.scipy import ndimage
from pyuvdata import UVBeam

from .. import coordinates
from .._uvbeam_to_raw import uvbeam_to_azza_grid
from ..core.beams import BeamInterpolator
from ..cpu.beams import UVBeamInterpolator


class GPUBeamInterpolator(BeamInterpolator):
    """Interpolate a UVBeam object on the GPU.

    This uses cupy.ndimage.map_coordinates to perform the interpolation.
    """

    def setup(self):
        """Set up the interpolator.

        Decides if the beam_list is a list of UVBeam objects or AnalyticBeam objects,
        and dispatches accordingly.
        """
        self.use_interp = isinstance(self.beam_list[0], UVBeam)
        if self.use_interp and not all(isinstance(b, UVBeam) for b in self.beam_list):
            raise ValueError(
                "GPUBeamInterpolator only supports beam_lists with either all UVBeam or all AnalyticBeam objects."
            )

        if self.use_interp:
            # We need to make sure that each beam "raw" data is on the same grid.
            # There is no advantage to using any other resolution but the native raw
            # resolution, which is what is returned by default. This may not be the case
            # if we were to use higher-order splines in the initial interpolation from
            # UVBeam. Eg. if "cubic" interpolation was shown to be better than linear,
            # we might want to do cubic interpolation with pyuvbeam onto a much higher-res
            # grid, then use linear interpolation on the GPU with that high-res grid.
            # We can explore this later...
            d0, self.daz, self.dza = uvbeam_to_azza_grid(self.beam_list[0])
            naz = 2 * np.pi / self.daz + 1
            assert np.isclose(int(naz), naz)

            self.beam_data = cp.zeros(
                (self.nbeam,) + d0.shape,
                dtype=self.complex_dtype if self.polarized else self.real_dtype,
            )
            self.beam_data[0] = cp.asarray(d0)

            if len(self.beam_list) > 1:
                for i, b in enumerate(self.beam_list[1:]):
                    self.beam_data[i + 1].set(
                        uvbeam_to_azza_grid(b, naz=int(naz), dza=self.dza)[0]
                    )
            #                    self.beam_data[i+1] = uvbeam_to_azza_grid(b, naz=int(naz), dza=self.dza)[0]

            self.beam_idx = cp.asarray(self.beam_idx, dtype=np.uint)

        else:
            # If doing simply analytic beams, just use the UVBeamInterpolator
            self._eval = UVBeamInterpolator.interp

    def interp(self, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
        """Evaluate the beam on the GPU.

        This function will either interpolate the beam to the given coordinates tx, ty,
        or evaluate the beam there if it is an analytic beam.

        Parameters
        ----------
        tx, ty
            Coordinates to evaluate the beam at, in sin-projection.
        """
        return (
            self._interp(tx, ty)
            if self.use_interp
            else cp.asarray(self._eval(self, tx.get(), ty.get()))
        )

    def _interp(
        self,
        tx: np.ndarray,
        ty: np.ndarray,
    ):
        """Perform the beam interpolation, choosing between CPU and GPU as necessary."""
        az, za = coordinates.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")

        return gpu_beam_interpolation(self.beam_data, self.daz, self.dza, az, za)


def gpu_beam_interpolation(
    beam: np.ndarray | cp.ndarray,
    daz: float,
    dza: float,
    az: np.ndarray | cp.ndarray,
    za: np.ndarray | cp.ndarray,
):
    """
    Interpolate beam values from a regular az/za grid using GPU.

    Parameters
    ----------
    beam
        The beam values. The shape of this array should be
        ``(nbeam, nax, nfeed, nza, naz)``. This is the axis ordering returned by
        UVBeam.interp. This array can either be real or complex. Either way, the output
        is complex.
    daz, dza
        The grid sizes in azimuth and zenith-angle respectively.
    az, za
        The azimuth and zenith-angle values of the sources to which to interpolate.
        These should be  1D arrays. They are not treated as a "grid".

    Returns
    -------
    beam_at_src
        The beam interpolated at the sources. The shape of the array is
        ``(nbeam, nfeed, nax, nsrc)``. The array is always complex (at single or
        double precision, depending on the input).
    """
    beam = cp.asarray(beam)
    az = cp.asarray(az)
    za = cp.asarray(za)

    # Get precision from the beam object.
    if beam.dtype in (np.dtype("float32"), np.dtype("complex64")):
        ctype = np.dtype("complex64")
    elif beam.dtype in (np.dtype("float64"), np.dtype("complex128")):
        ctype = np.dtype("complex128")
    else:
        raise ValueError(
            f"Got {beam.dtype} as the dtype for beam, which is unrecognized"
        )

    complex_beam = beam.dtype.name.startswith("complex")
    az /= daz
    za /= dza

    nbeam, nax, nfeed, nza, naz = beam.shape
    nsrc = len(az)

    beam_at_src = cp.zeros((nbeam, nfeed, nax, nsrc), dtype=beam.dtype)

    for bm, fd, ax in itertools.product(range(nbeam), range(nfeed), range(nax)):
        ndimage.map_coordinates(
            beam[bm, ax, fd],
            cp.asarray([za, az]),
            order=1,
            output=beam_at_src[bm, fd, ax],
            mode="nearest",  # controls the end-point behavior, no-op
        )

    if not complex_beam:  # power beam
        cp.sqrt(beam_at_src, out=beam_at_src)
        beam_at_src = beam_at_src.astype(ctype)

    cp.cuda.Device().synchronize()
    return beam_at_src
