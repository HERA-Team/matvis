"""GPU beam interpolation routines."""

import itertools

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage
from pyuvdata import UVBeam

from .. import coordinates
from ..core.beams import BeamInterpolator
from ..cpu.beams import UVBeamInterpolator

# One fused bilinear-interpolation kernel evaluating every (beam, feed, axis)
# plane for every source in a single launch. The previous implementation made
# nbeam*nfeed*nax separate map_coordinates launches per chunk (1400 launches
# for a 350-antenna array with per-antenna beams), which left the GPU idle
# most of the time waiting on the host to issue work.
#
# Grid: x covers sources, y covers planes. Out-of-range coordinates clamp to
# the grid edge. Input layout (nbeam, nax, nfeed, nza, naz) [UVBeam order],
# output layout (nbeam, nfeed, nax, nsrc) [matvis order].
_BILINEAR_MODULE = cp.RawModule(
    code=r"""
#include <cupy/complex.cuh>

template<typename R, typename T>
__device__ void bilinear_all_planes(
    const T* __restrict__ beam,
    const R* __restrict__ az,
    const R* __restrict__ za,
    const R* __restrict__ daz,
    const R* __restrict__ dza,
    const R* __restrict__ azmin,
    const int nfeed,
    const int nax,
    const long nza,
    const long naz,
    const long nsrc,
    T* __restrict__ out)
{
    const long s = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (s >= nsrc) return;
    const int p = blockIdx.y;
    const int bm = p / (nax * nfeed);
    const int r = p % (nax * nfeed);
    const int ax = r / nfeed;
    const int fd = r % nfeed;

    R x = za[s] / dza[bm];
    R y = (az[s] - azmin[bm]) / daz[bm];

    long x0 = (long)floor(x);
    long y0 = (long)floor(y);
    R fx = x - x0;
    R fy = y - y0;
    if (x0 < 0) { x0 = 0; fx = 0; }
    if (x0 > nza - 2) { x0 = nza - 2; fx = 1; }
    if (y0 < 0) { y0 = 0; fy = 0; }
    if (y0 > naz - 2) { y0 = naz - 2; fy = 1; }

    const T* b = beam + ((((long)bm * nax + ax) * nfeed + fd) * nza + x0) * naz + y0;
    T v = b[0] * ((1 - fx) * (1 - fy))
        + b[1] * ((1 - fx) * fy)
        + b[naz] * (fx * (1 - fy))
        + b[naz + 1] * (fx * fy);

    out[(((long)bm * nfeed + fd) * nax + ax) * nsrc + s] = v;
}

extern "C" {
__global__ void bilinear_c64(
    const complex<float>* beam, const float* az, const float* za,
    const float* daz, const float* dza, const float* azmin,
    int nfeed, int nax, long nza, long naz, long nsrc, complex<float>* out)
{ bilinear_all_planes<float, complex<float> >(beam, az, za, daz, dza, azmin, nfeed, nax, nza, naz, nsrc, out); }

__global__ void bilinear_c128(
    const complex<double>* beam, const double* az, const double* za,
    const double* daz, const double* dza, const double* azmin,
    int nfeed, int nax, long nza, long naz, long nsrc, complex<double>* out)
{ bilinear_all_planes<double, complex<double> >(beam, az, za, daz, dza, azmin, nfeed, nax, nza, naz, nsrc, out); }

__global__ void bilinear_f32(
    const float* beam, const float* az, const float* za,
    const float* daz, const float* dza, const float* azmin,
    int nfeed, int nax, long nza, long naz, long nsrc, float* out)
{ bilinear_all_planes<float, float>(beam, az, za, daz, dza, azmin, nfeed, nax, nza, naz, nsrc, out); }

__global__ void bilinear_f64(
    const double* beam, const double* az, const double* za,
    const double* daz, const double* dza, const double* azmin,
    int nfeed, int nax, long nza, long naz, long nsrc, double* out)
{ bilinear_all_planes<double, double>(beam, az, za, daz, dza, azmin, nfeed, nax, nza, naz, nsrc, out); }
}
"""
)

_BILINEAR_KERNELS = {
    np.dtype("complex64"): ("bilinear_c64", np.float32),
    np.dtype("complex128"): ("bilinear_c128", np.float64),
    np.dtype("float32"): ("bilinear_f32", np.float32),
    np.dtype("float64"): ("bilinear_f64", np.float64),
}


def prepare_for_map_coords(uvbeam: UVBeam):
    """Obtain coordinates for doing map_coordinates interpolation from a UVBeam."""
    d0, az, za = uvbeam._prepare_coordinate_data(uvbeam.data_array)
    d0 = d0[:, :, 0]  # only one frequency
    return d0, np.diff(az)[0], np.diff(za)[0], az.min()


class GPUBeamInterpolator(BeamInterpolator):
    """Interpolate a UVBeam object on the GPU.

    This uses cupy.ndimage.map_coordinates to perform the interpolation.
    """

    def setup(self):
        """Set up the interpolator.

        Decides if the beam_list is a list of UVBeam objects or AnalyticBeam objects,
        and dispatches accordingly.
        """
        self.use_interp = self.beam_list[0]._isuvbeam
        if self.use_interp and not all(b._isuvbeam for b in self.beam_list):
            raise ValueError(
                "GPUBeamInterpolator only supports beam_lists with either all UVBeam or all AnalyticBeam objects."
            )

        if self.beam_idx is not None:
            self.beam_idx = cp.asarray(self.beam_idx, dtype=np.uint)

        if self.use_interp:
            # We need to make sure that each beam "raw" data is on the same grid.
            # There is no advantage to using any other resolution but the native raw
            # resolution, which is what is returned by default. This may not be the case
            # if we were to use higher-order splines in the initial interpolation from
            # UVBeam. Eg. if "cubic" interpolation was shown to be better than linear,
            # we might want to do cubic interpolation with pyuvbeam onto a much higher-res
            # grid, then use linear interpolation on the GPU with that high-res grid.
            # We can explore this later...
            if any(bm.beam.pixel_coordinate_system != "az_za" for bm in self.beam_list):
                raise ValueError('pixel coordinate system must be "az_za"')

            self.daz = np.zeros(len(self.beam_list))
            self.dza = np.zeros(len(self.beam_list))
            self.azmin = np.zeros(len(self.beam_list))

            d0, self.daz[0], self.dza[0], self.azmin[0] = prepare_for_map_coords(
                self.beam_list[0].beam
            )

            self.beam_data = cp.zeros(
                (self.nbeam,) + d0.shape,
                dtype=self.complex_dtype if self.polarized else self.real_dtype,
            )
            self.beam_data[0].set(d0.astype(self.beam_data.dtype, copy=False))

            if len(self.beam_list) > 1:
                for i, b in enumerate(self.beam_list[1:]):
                    d, self.daz[i + 1], self.dza[i + 1], self.azmin[i + 1] = (
                        prepare_for_map_coords(b.beam)
                    )
                    self.beam_data[i + 1].set(
                        d.astype(self.beam_data.dtype, copy=False)
                    )
        else:
            # If doing simply analytic beams, just use the UVBeamInterpolator
            self._eval = UVBeamInterpolator.interp
            self._np_beam = np.zeros(
                (self.nbeam, self.nfeed, self.nax, self.nsrc), dtype=self.complex_dtype
            )

        self.interpolated_beam = cp.zeros(
            (self.nbeam, self.nfeed, self.nax, self.nsrc), dtype=self.complex_dtype
        )

    def interp(self, tx: cp.ndarray, ty: cp.ndarray, out: cp.ndarray) -> np.ndarray:
        """Evaluate the beam on the GPU.

        This function will either interpolate the beam to the given coordinates tx, ty,
        or evaluate the beam there if it is an analytic beam.

        Parameters
        ----------
        tx, ty
            Coordinates to evaluate the beam at, in sin-projection.
        """
        if self.use_interp:
            self._interp(tx, ty, out)
        else:
            self._eval(self, cp.asnumpy(tx), cp.asnumpy(ty), self._np_beam)
            out.set(self._np_beam)

    def _interp(
        self,
        tx: cp.ndarray,
        ty: cp.ndarray,
        out: cp.ndarray,
    ):
        """Perform the beam interpolation, choosing between CPU and GPU as necessary."""
        az, za = coordinates.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")

        # Set all the elements
        self.interpolated_beam[..., len(az) :] = 0.0

        gpu_beam_interpolation(
            self.beam_data,
            self.daz,
            self.dza,
            self.azmin,
            az,
            za,
            beam_at_src=out,
            **self.spline_opts,
        )


def gpu_beam_interpolation(
    beam: np.ndarray | cp.ndarray,
    daz: np.ndarray,
    dza: np.ndarray,
    azmin: np.ndarray,
    az: np.ndarray | cp.ndarray,
    za: np.ndarray | cp.ndarray,
    beam_at_src: cp.ndarray | None = None,
    order: int = 1,
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

    nbeam, nax, nfeed, nza, naz = beam.shape
    nsrc = len(az)

    if beam_at_src is None:
        beam_at_src = cp.zeros((nbeam, nfeed, nax, nsrc), dtype=beam.dtype)
    else:
        assert beam_at_src.shape == (nbeam, nfeed, nax, nsrc)

    if order == 1:
        # Single fused launch over all (beam, feed, axis) planes and sources.
        # The kernel writes values of the beam's own dtype; when the caller
        # supplies a complex output buffer for a real (power) beam, go through
        # a real-valued scratch array and cast on copy (as map_coordinates
        # would have done element-wise).
        target = (
            beam_at_src
            if beam_at_src.dtype == beam.dtype
            else cp.empty((nbeam, nfeed, nax, nsrc), dtype=beam.dtype)
        )
        kernel_name, rdtype = _BILINEAR_KERNELS[beam.dtype]
        kern = _BILINEAR_MODULE.get_function(kernel_name)
        az = cp.ascontiguousarray(az, dtype=rdtype)
        za = cp.ascontiguousarray(za, dtype=rdtype)
        daz = cp.asarray(daz, dtype=rdtype)
        dza = cp.asarray(dza, dtype=rdtype)
        azmin = cp.asarray(azmin, dtype=rdtype)
        assert beam._c_contiguous and target._c_contiguous
        block = 128
        grid = ((nsrc + block - 1) // block, nbeam * nfeed * nax)
        kern(
            grid,
            (block,),
            (
                beam,
                az,
                za,
                daz,
                dza,
                azmin,
                np.int32(nfeed),
                np.int32(nax),
                np.int64(nza),
                np.int64(naz),
                np.int64(nsrc),
                target,
            ),
        )
        if target is not beam_at_src:
            if not complex_beam:
                cp.sqrt(target, out=target)
            beam_at_src[:] = target
            return beam_at_src
    else:
        for bm in range(nbeam):
            coords = cp.asarray([za / dza[bm], (az - azmin[bm]) / daz[bm]])
            for fd, ax in itertools.product(range(nfeed), range(nax)):
                ndimage.map_coordinates(
                    beam[bm, ax, fd],
                    coords,
                    order=order,
                    output=beam_at_src[bm, fd, ax],
                )

    if not complex_beam:  # power beam
        cp.sqrt(beam_at_src, out=beam_at_src)
        beam_at_src = beam_at_src.astype(ctype)
    return beam_at_src
