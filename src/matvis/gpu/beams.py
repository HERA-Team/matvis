"""GPU beam interpolation routines."""

import itertools
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage
from pyuvdata import UVBeam

from .. import coordinates
from ..core.beams import BeamInterpolator
from ..cpu.beams import UVBeamInterpolator

KERNELS_PATH = Path(__file__).parent / "kernels"

# See kernels/beam_interp.cu for the kernel source and layout notes.
_INTERP_MODULE = cp.RawModule(code=(KERNELS_PATH / "beam_interp.cu").read_text())

# Kernel-name suffix and matching coordinate dtype for each supported beam dtype.
_KERNEL_DTYPES = {
    np.dtype("complex64"): ("c64", np.float32),
    np.dtype("complex128"): ("c128", np.float64),
    np.dtype("float32"): ("f32", np.float32),
    np.dtype("float64"): ("f64", np.float64),
}

# Interpolation orders served by a dedicated fused kernel, rather than by the
# generic per-plane map_coordinates fallback.
_KERNEL_ORDERS = {1: "bilinear", 3: "bicubic"}

# Coefficient nodes the cubic stencil reaches beyond each edge of the grid.
# Must match CUBIC_HALO in kernels/beam_interp.cu.
_CUBIC_HALO = 1

# Edge-replication padding applied before the spline prefilter. This is not an
# independently chosen convergence bound -- it is the value scipy itself uses
# (``scipy.ndimage._interpolation._prepad_for_spline_filter`` for mode="nearest"),
# adopted so that matvis agrees with scipy to floating-point precision rather
# than merely converging towards the same answer. Truncating the filter's
# boundary condition at 12 nodes leaves a ~1e-7 relative error against the exact
# nearest-boundary spline, which both libraries share and which is well below
# single-precision resolution.
_SPLINE_PAD = 12


def prepare_for_map_coords(uvbeam: UVBeam) -> tuple[np.ndarray, float, float, float]:
    """Obtain coordinates for doing map_coordinates interpolation from a UVBeam.

    Returns
    -------
    array
        The beam data array in the shape defined by UVBeam, but without a frequency
        axis. For a power beam, shape (1, Npols, Nza, Naz). For Efield
        beam (Naxes_vec, Nfeeds, Nza, Naz).
    float
        The regular grid spacing in azimuth for the beam data.
    float
        The regular grid spacing in zenith angle for the beam data.
    float
        The minimum azimuth of the beam data.
    """
    d0, az, za = uvbeam._prepare_coordinate_data(uvbeam.data_array)
    d0 = d0[:, :, 0]  # only one frequency
    return d0, np.diff(az)[0], np.diff(za)[0], az.min()


@dataclass(frozen=True)
class BeamCoefficients:
    """B-spline coefficients for a beam, as returned by :func:`prefilter_beam`.

    Wrapping the array rather than returning it bare makes "have these values
    been prefiltered?" a fact about the *type* instead of something the caller
    has to track and assert correctly. Both ways of getting it wrong --
    prefiltering twice, or forgetting to prefilter at all -- would otherwise
    produce a plausible-looking but subtly over-smoothed beam rather than an
    error, since a coefficient array is not distinguishable from a beam grid by
    inspection.

    Attributes
    ----------
    coeffs
        The spline coefficients, shape ``(nbeam, nax, nfeed, nza + 2 * halo,
        naz + 2 * halo)``, in the dtype of the beam they came from.
    order
        The spline order the coefficients were computed for.
    halo
        Coefficient nodes carried beyond each edge of the underlying grid.
    """

    coeffs: cp.ndarray
    order: int = 3
    halo: int = _CUBIC_HALO

    @property
    def grid_shape(self) -> tuple[int, int]:
        """The ``(nza, naz)`` shape of the underlying beam grid, without the halo."""
        nza, naz = self.coeffs.shape[-2:]
        return nza - 2 * self.halo, naz - 2 * self.halo


def prefilter_beam(beam: np.ndarray | cp.ndarray, order: int = 3) -> BeamCoefficients:
    """Convert gridded beam values into B-spline coefficients for cubic interpolation.

    Cubic interpolation is a two-step process: the grid is first converted into
    the coefficients of a cubic B-spline that passes exactly through the grid
    values (this function), and those coefficients are then combined with the
    B-spline basis at each source position (the ``bicubic`` CUDA kernel).
    Skipping this step and applying the basis directly to the grid values would
    *smooth* the beam by the (1, 4, 1)/6 kernel rather than interpolate it.

    The prefilter is a sequential recursion along each grid axis, so it is much
    less GPU-friendly than the interpolation itself -- but it depends only on
    the beam, so matvis runs it once during setup rather than once per source
    chunk.

    Parameters
    ----------
    beam
        Gridded beam values, shape ``(nbeam, nax, nfeed, nza, naz)``. Real or
        complex; the output has the same dtype.
    order
        Spline order. Only ``3`` is supported: order 1 needs no prefilter, and
        the other orders go through :func:`cupyx.scipy.ndimage.map_coordinates`,
        which prefilters internally.

    Returns
    -------
    BeamCoefficients
        Spline coefficients, shape ``(nbeam, nax, nfeed, nza + 2, naz + 2)``.
        The extra node on each side of both grid axes is the halo that the
        four-point cubic stencil reaches into at the edges of the grid. Pass
        this straight to :func:`gpu_beam_interpolation` in place of the beam.

    Notes
    -----
    The boundary treatment reproduces ``scipy.ndimage.map_coordinates(...,
    order=3, mode="nearest")``: the grid is padded with 12 nodes of edge
    replication before filtering, and the result trimmed back to a one-node
    halo. The recursion is run in double precision whatever the beam's dtype,
    since it is a one-off setup cost and is the least numerically forgiving
    step in the pipeline.
    """
    if order != 3:
        raise ValueError(f"prefilter_beam only supports order=3, got {order}")

    beam = cp.asarray(beam)
    work = np.dtype("complex128" if beam.dtype.kind == "c" else "float64")
    # Pad (and later trim back) only the two grid axes of a single beam's block.
    pad = ((0, 0),) * (beam.ndim - 3) + ((_SPLINE_PAD, _SPLINE_PAD),) * 2
    keep = slice(_SPLINE_PAD - _CUBIC_HALO, _CUBIC_HALO - _SPLINE_PAD)

    nza, naz = beam.shape[-2:]
    out = cp.empty(
        beam.shape[:-2] + (nza + 2 * _CUBIC_HALO, naz + 2 * _CUBIC_HALO), beam.dtype
    )

    # One beam at a time: the padded double-precision working copy is the
    # largest array involved, and doing all beams at once would need a
    # multi-gigabyte transient allocation for a production-sized beam list.
    for i, block in enumerate(beam):
        coeff = cp.pad(block, pad, mode="edge")
        for axis in (-2, -1):
            coeff = ndimage.spline_filter1d(
                coeff, order=order, axis=axis, output=work, mode="nearest"
            )
        out[i] = coeff[..., keep, keep]

    return BeamCoefficients(out, order=order, halo=_CUBIC_HALO)


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
        self.spline_order = self.spline_opts.get("order", 1)
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

            if self.spline_order == 3:
                # Cubic interpolation consumes B-spline coefficients, which
                # depend only on the beam. Pay for them once, here, rather than
                # once per source chunk in the time/frequency loop. From here on
                # beam_data is a BeamCoefficients, which gpu_beam_interpolation
                # accepts in place of a raw grid.
                self.beam_data = prefilter_beam(self.beam_data)
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
            power_beam=not self.polarized,
            **self.spline_opts,
        )


def gpu_beam_interpolation(
    beam: np.ndarray | cp.ndarray | BeamCoefficients,
    daz: np.ndarray,
    dza: np.ndarray,
    azmin: np.ndarray,
    az: np.ndarray | cp.ndarray,
    za: np.ndarray | cp.ndarray,
    beam_at_src: cp.ndarray | None = None,
    order: int = 1,
    power_beam: bool | None = None,
):
    """
    Interpolate beam values from a regular az/za grid using GPU.

    Parameters
    ----------
    beam
        The beam values. The shape of this array should be
        ``(nbeam, nax, nfeed, nza, naz)``. This is the axis ordering returned by
        UVBeam.interp. This array can either be real or complex. Either way, the output
        is complex. For ``order=3`` this may instead be the
        :class:`BeamCoefficients` returned by :func:`prefilter_beam`, which
        avoids re-running the prefilter on every call; passing raw values
        prefilters them internally.
    daz, dza
        The grid sizes in azimuth and zenith-angle respectively.
    az, za
        The azimuth and zenith-angle values of the sources to which to interpolate.
        These should be  1D arrays. They are not treated as a "grid".
    order
        Spline order to interpolate with. Orders 1 (bilinear) and 3 (bicubic)
        are served by dedicated fused CUDA kernels, which clamp out-of-range
        coordinates to the edge of the beam grid. Any other order falls back to
        a per-plane :func:`cupyx.scipy.ndimage.map_coordinates` loop, which is
        substantially slower and uses map_coordinates' own boundary handling.
    power_beam
        Whether the provided ``beam`` is in power units or E-field units. If not
        provided, then it is inferred based on whether the provided ``beam`` is real- or
        complex-valued. Failing to set ``power_beam=True`` and providing a power beam
        with cross-polarized components will result in the interpolation routine
        treating the beam as if it were an E-field beam instead of a power beam (i.e.,
        no square root will be taken after interpolation).

    Returns
    -------
    beam_at_src
        The beam interpolated at the sources. The shape of the array is
        ``(nbeam, nfeed, nax, nsrc)``. The array is always complex (at single or
        double precision, depending on the input).
    """
    # Unwrap prefiltered coefficients, remembering the grid dims they describe
    # (the stored array's trailing axes include the halo).
    grid_shape = None
    if isinstance(beam, BeamCoefficients):
        if beam.order != order:
            raise ValueError(
                f"beam holds order-{beam.order} spline coefficients, but "
                f"order={order} was requested"
            )
        grid_shape = beam.grid_shape
        beam = beam.coeffs

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

    complex_beam = (
        not power_beam
        if power_beam is not None
        else beam.dtype.name.startswith("complex")
    )

    nbeam, nax, nfeed, nza, naz = beam.shape
    if grid_shape is not None:
        nza, naz = grid_shape
    elif order == 3:
        beam = prefilter_beam(beam, order=order).coeffs
    nsrc = len(az)

    if np.iscomplexobj(beam) and nax == 1:
        raise ValueError(
            "The beam is complex valued but has only one Efield axis. Are you sure this isn't a power beam?"
        )

    if beam_at_src is None:
        beam_at_src = cp.zeros((nbeam, nfeed, nax, nsrc), dtype=beam.dtype)
    else:
        assert beam_at_src.shape == (nbeam, nfeed, nax, nsrc)

    if order in _KERNEL_ORDERS:
        # Use the custom beam interpolation kernel. If provided a power beam
        # and a complex output buffer, cast interpolated beam to complex on
        # copy.
        target = (
            beam_at_src
            if beam_at_src.dtype == beam.dtype
            else cp.empty((nbeam, nfeed, nax, nsrc), dtype=beam.dtype)
        )
        suffix, rdtype = _KERNEL_DTYPES[beam.dtype]
        kern = _INTERP_MODULE.get_function(f"{_KERNEL_ORDERS[order]}_{suffix}")
        az = cp.ascontiguousarray(az, dtype=rdtype)
        za = cp.ascontiguousarray(za, dtype=rdtype)
        daz = cp.asarray(daz, dtype=rdtype)
        dza = cp.asarray(dza, dtype=rdtype)
        azmin = cp.asarray(azmin, dtype=rdtype)
        assert beam._c_contiguous and target._c_contiguous
        # 128 is a conventional warp-multiple default, not empirically
        # tuned for this kernel/shape.
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
