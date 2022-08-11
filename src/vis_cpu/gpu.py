"""GPU implementation of the simulator."""
from __future__ import annotations

import logging
import numpy as np
import warnings
from astropy.constants import c as speed_of_light
from pathlib import Path
from pyuvdata import UVBeam
from typing import Callable, Optional, Sequence

from . import conversions
from ._uvbeam_to_raw import uvbeam_to_azza_grid
from .cpu import _evaluate_beam_cpu, _validate_inputs, _wrangle_beams, vis_cpu

logger = logging.getLogger(__name__)

# This enables us to put in profile decorators that will be no-ops if no profiling
# library is being used.
try:
    profile
except NameError:
    from ._utils import no_op

    profile = no_op

try:
    import pycuda.autoinit
    from jinja2 import Template
    from pycuda import compiler
    from pycuda import cumath as cm
    from pycuda import driver, gpuarray
    from skcuda.cublas import (
        cublasCgemm,
        cublasCreate,
        cublasDestroy,
        cublasDgemm,
        cublasSetStream,
        cublasSgemm,
        cublasZgemm,
    )

    HAVE_CUDA = True

except ImportError:
    HAVE_CUDA = False
    Template = no_op


ONE_OVER_C = 1.0 / speed_of_light.value

templates = Path(__file__).parent / "gpu_src"

with open(templates / "measurement_equation.cu") as fl:
    MeasEqTemplate = Template(fl.read())

with open(templates / "beam_interpolation.cu") as fl:
    BeamInterpTemplate = Template(fl.read())

TYPE_MAP = {
    np.float32: "float",
    np.float64: "double",
    np.complex64: "cuFloatComplex",
    np.complex128: "cuDoubleComplex",
    np.dtype("float64"): "double",
}


def _logdebug(xgpu: gpuarray.GPUArray, name: str):
    """Log an array shape and first 40 elements as a debug statement.

    We put this in an if statement because the xgpu.get() statement takes a long
    time for large arrays, and we don't want to do it at all if we're not going to end
    up logging it.
    """
    if logger.getEffectiveLevel() <= logging.DEBUG:  # pragma: no cover
        # For comparison to cpu
        xcpu = xgpu.get().flatten()
        if xcpu.size > 40:
            xcpu = xcpu[:40]

        logger.debug(
            f"GPU: {name} | {xgpu.shape}: {xcpu}",
        )


@profile
def vis_gpu(
    *,
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam | Callable] | None,
    polarized: bool = False,
    beam_idx: np.ndarray | None = None,
    nthreads: int = 1024,
    max_memory: int = 2**29,
    min_chunks: int = 1,
    precision: int = 1,
    beam_spline_opts: dict | None = None,
) -> np.ndarray:
    """GPU implementation of the visibility simulator."""
    if not HAVE_CUDA:
        raise ImportError("You need to install the [gpu] extra to use this function!")

    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, eq2tops, crd_eq, I_sky
    )

    if beam_spline_opts:
        warnings.warn(
            "You have passed beam_spline_opts, but these are not used in GPU."
        )

    nsrc = len(I_sky)

    if precision == 1:
        real_dtype, complex_dtype = np.float32, np.complex64
        cublas_real_mm = cublasSgemm
        cublas_complex_mm = cublasCgemm
    else:
        real_dtype, complex_dtype = np.float64, np.complex128
        cublas_real_mm = cublasDgemm
        cublas_complex_mm = cublasZgemm

    DTYPE, CDTYPE = TYPE_MAP[real_dtype], TYPE_MAP[complex_dtype]

    # apply scalars so 1j*tau*freq is the correct exponent
    ang_freq = 2 * freq * np.pi

    # ensure data types
    antpos = antpos.astype(real_dtype)
    eq2tops = eq2tops.astype(real_dtype)
    crd_eq = crd_eq.astype(real_dtype)
    Isqrt = np.sqrt(0.5 * I_sky).astype(real_dtype)

    chunk = max(
        min(nsrc, min_chunks),
        2 ** int(np.ceil(np.log2(float(nant * nsrc) / max_memory / 2))),
    )
    npixc = nsrc // chunk

    beam_list, nbeam, beam_idx = _wrangle_beams(
        beam_idx=beam_idx,
        beam_list=beam_list,
        polarized=polarized,
        nant=nant,
        freq=freq,
    )

    # Ways to block up threads for sending to GPU calculations. "Meas" is for the
    # measurement equation function, and "prod" is for the inner-product calculation.
    meas_block = (
        max(1, nthreads // (nant * nax * nfeed)),
        min(nthreads, nant * nax),
        min(nthreads, nfeed),
    )

    logger.info(
        f"Using {np.prod(meas_block)} threads in total for measurement equation."
    )
    logger.info(f"Using a shared-memory buffer of size {5*meas_block[0]}.")

    use_uvbeam = isinstance(beam_list[0], UVBeam)
    if use_uvbeam and not all(isinstance(b, UVBeam) for b in beam_list):
        raise ValueError(
            "vis_gpu only support beam_lists with either all UVBeam or all AnalyticBeam objects."
        )

    cuda_params = {
        "NANT": nant,
        "NAX": nax,
        "NFEED": nfeed,
        "NBEAM": nbeam,
        "BLOCK_PX": meas_block[0],
        "DTYPE": DTYPE,
        "CDTYPE": CDTYPE,
        "f": "f" if precision == 1 else "",
    }

    if use_uvbeam:
        # We need to make sure that each beam "raw" data is on the same grid.
        # There is no advantage to using any other resolution but the native raw
        # resolution, which is what is returned by default. This may not be the case
        # if we were to use higher-order splines in the initial interpolation from
        # UVBeam. Eg. if "cubic" interpolation was shown to be better than linear,
        # we might want to do cubic interpolation with pyuvbeam onto a much higher-res
        # grid, then use linear interpolation on the GPU with that high-res grid.
        # We can explore this later...
        d0, daz, dza = uvbeam_to_azza_grid(beam_list[0])
        naz = 2 * np.pi / daz + 1
        assert np.isclose(int(naz), naz)

        raw_beam_data = [d0]
        if len(beam_list) > 1:
            raw_beam_data.extend(
                uvbeam_to_azza_grid(b, naz=int(naz), dza=dza)[0] for b in beam_list[1:]
            )
    else:
        daz, dza = None, None

    # Setup the GPU code and arrays
    meas_eq_code = MeasEqTemplate.render(**cuda_params)

    if use_uvbeam:
        beam_interp_code = BeamInterpTemplate.render(
            **{
                **cuda_params,
                **{
                    "NBEAM": nbeam,
                    "BEAM_N_AZ": raw_beam_data[0].shape[-1],
                    "BEAM_N_ZA": raw_beam_data[0].shape[-2],
                    "DAZ": daz,
                    "DZA": dza,
                },
            }
        )
        beam_interp_module = compiler.SourceModule(beam_interp_code)
        beam_interp = beam_interp_module.get_function("InterpolateBeamAltAz")
    else:
        beam_interp = None

    meas_eq_module = compiler.SourceModule(meas_eq_code)
    meas_eq = meas_eq_module.get_function("MeasEq")
    # vis_inner_product = meas_eq_module.get_function("VisInnerProduct")

    logger.info(
        f"""
        Measurement Equation Kernel Properties:
            SHARED: {meas_eq.shared_size_bytes}
            LOCAL: {meas_eq.local_size_bytes}
            REGISTERS: {meas_eq.num_regs}
            MAX_THREADS_PER_BLOCK: {meas_eq.max_threads_per_block}
        """
    )

    # bm_texref = gpu_module.get_texref("bm_tex")
    h = cublasCreate()  # handle for managing cublas

    # define GPU buffers and transfer initial values
    # never changes, transpose happens in copy so cuda bm_tex is (BEAM_PX,BEAM_PX,NANT)
    # bm_texref.set_array(numpy3d_to_array(beams))
    antpos_gpu = gpuarray.to_gpu(antpos)  # never changes, set to -2*pi*antpos/c
    beam_idx = gpuarray.to_gpu(beam_idx.astype(np.uint))
    Isqrt_gpu = gpuarray.empty(shape=(npixc,), dtype=real_dtype)

    # Send the regular-grid beam data to the GPU. This has dimensions (Nbeam, Nax, Nfeed, Nza, Nza)
    # Note that Nbeam is not in general equal to Nant (we can have multiple antennas with
    # the same beam).
    if use_uvbeam:
        beam_data_gpu = gpuarray.to_gpu(
            np.array(raw_beam_data, dtype=complex_dtype if polarized else real_dtype),
        )
    else:
        beam_data_gpu = None

    # will be set on GPU by bm_interp
    crd_eq_gpu = gpuarray.empty(shape=(3, npixc), dtype=real_dtype)
    # sent from CPU each time
    eq2top_gpu = gpuarray.empty(shape=(3, 3), dtype=real_dtype)
    # will be set on GPU
    crdtop_gpu = gpuarray.empty(shape=(3, npixc), dtype=real_dtype)
    # will be set on GPU
    vis_gpus = [
        gpuarray.empty(shape=(nfeed * nant, nfeed * nant), dtype=complex_dtype)
        for _ in range(chunk)
    ]

    # output CPU buffers for downloading answers
    vis_cpus = [
        np.empty(shape=(nfeed * nant, nfeed * nant), dtype=complex_dtype)
        for _ in range(chunk)
    ]
    streams = [driver.Stream() for _ in range(chunk)]
    event_order = [
        "start",
        "upload",
        "eq2top",
        "tau",
        "meas_eq",
        "vis",
        "end",
    ]

    if use_uvbeam:
        event_order.insert(4, "interpolation")

    vis = np.empty((ntimes, nfeed * nant, nfeed * nant), dtype=complex_dtype)

    logger.info("Running With %s chunks: ", chunk)

    for t in range(ntimes):
        eq2top_gpu.set(eq2tops[t])  # defines sky orientation for this time step
        events = [{e: driver.Event() for e in event_order} for _ in range(chunk)]

        for c in range(chunk + 2):
            cc = c - 1
            ccc = c - 2
            if 0 <= ccc < chunk:
                stream = streams[ccc]
                vis_gpus[ccc].get(ary=vis_cpus[ccc], stream=stream)
                events[ccc]["end"].record(stream)
            if 0 <= cc < chunk:
                stream = streams[cc]
                cublasSetStream(h, stream.handle)

                # cublas arrays are in Fortran order, so P=M*N is actually
                # peformed as P.T = N.T * M.T
                cublas_real_mm(  # compute crdtop = dot(eq2top,crd_eq)
                    h,
                    "n",
                    "n",
                    npixc,
                    3,
                    3,
                    1.0,
                    crd_eq_gpu.gpudata,
                    npixc,
                    eq2top_gpu.gpudata,
                    3,
                    0.0,
                    crdtop_gpu.gpudata,
                    npixc,
                )
                events[cc]["eq2top"].record(stream)

                tx, ty, tz = crdtop_gpu.get_async(stream=stream)
                above_horizon = tz > 0
                tx = tx[above_horizon]
                ty = ty[above_horizon]
                nsrcs_up = len(tx)
                crdtop_lim_gpu = gpuarray.to_gpu_async(
                    crdtop_gpu.get_async(stream=stream)[:, above_horizon].copy(),
                    stream=stream,
                )

                if nsrcs_up < 1:
                    continue

                tau_gpu = gpuarray.empty(shape=(nant, nsrcs_up), dtype=real_dtype)

                cublas_real_mm(  # compute tau = dot(antpos,crdtop) / speed_of_light
                    h,
                    "n",
                    "n",
                    nsrcs_up,
                    nant,
                    3,
                    ONE_OVER_C,
                    crdtop_lim_gpu.gpudata,
                    nsrcs_up,
                    antpos_gpu.gpudata,
                    3,
                    0.0,
                    tau_gpu.gpudata,
                    nsrcs_up,
                )
                events[cc]["tau"].record(stream)

                # Need to do this in polar coordinates, NOT (l,m), at least for
                # polarized beams. This is because at zenith, the Efield components are
                # discontinuous (in power they are continuous). When interpolating the
                # E-field components, you need to treat the zenith point differently
                # depending on which "side" of zenith you're on. This is doable in polar
                # coordinates, but not in Cartesian coordinates.
                A_gpu = do_beam_interpolation(
                    freq,
                    beam_list,
                    polarized,
                    nthreads,
                    nax,
                    nfeed,
                    complex_dtype,
                    nbeam,
                    use_uvbeam,
                    daz,
                    dza,
                    beam_interp,
                    beam_data_gpu,
                    events,
                    cc,
                    stream,
                    tx,
                    ty,
                    nsrcs_up,
                )

                v_gpu = gpuarray.empty(
                    shape=(nfeed * nant, nax * nsrcs_up), dtype=complex_dtype
                )
                Isqrt_lim_gpu = gpuarray.to_gpu_async(
                    Isqrt_gpu.get()[above_horizon].copy(), stream=stream
                )

                # blocks of threads are mapped to (pixels,ants,freqs)

                grid = (
                    int(np.ceil(nsrcs_up / float(meas_block[0]))),
                    int(np.ceil(nax * nant / float(meas_block[1]))),
                    int(np.ceil(nfeed / float(meas_block[2]))),
                )

                logger.info(f"Measurement Eq. Grid Size: {grid}")

                _logdebug(A_gpu, "Beam")

                # compute v = A * sqrtI * exp(1j*tau*freq)
                meas_eq(
                    A_gpu,
                    Isqrt_lim_gpu,
                    tau_gpu,
                    ang_freq,
                    np.uint(nsrcs_up),
                    beam_idx,
                    v_gpu,
                    grid=grid,
                    block=meas_block,
                    stream=stream,
                )
                events[cc]["meas_eq"].record(stream)

                _logdebug(v_gpu, "vant")

                # compute vis = dot(v, v.T)
                # We want to take an outer product over feeds/antennas, contract over
                # E-field components, and integrate over the sky.
                # Remember cublas is in fortran order...
                # v_gpu is (nfeed * nant, nax * nsrcs_up)
                cublas_complex_mm(
                    h,
                    "c",  # conjugate transpose for first (remember fortran order)
                    "n",  # no transpose for second.
                    nfeed * nant,
                    nfeed * nant,
                    nax * nsrcs_up,
                    1.0,
                    v_gpu.gpudata,
                    nax * nsrcs_up,
                    v_gpu.gpudata,
                    nax * nsrcs_up,
                    0.0,
                    vis_gpus[cc].gpudata,
                    nfeed * nant,
                )

                _logdebug(vis_gpus[cc], "Vis")

                events[cc]["vis"].record(stream)

            if c < chunk:
                # This is the first thing that happens in the loop over chunks.

                stream = streams[c]
                events[c]["start"].record(stream)
                crd_eq_gpu.set_async(
                    crd_eq[:, c * npixc : (c + 1) * npixc], stream=stream
                )
                Isqrt_gpu.set_async(Isqrt[c * npixc : (c + 1) * npixc], stream=stream)
                events[c]["upload"].record(stream)

        events[chunk - 1]["end"].synchronize()
        vis[t] = sum(vis_cpus)

    # teardown GPU configuration
    cublasDestroy(h)
    vis = vis.conj().reshape((ntimes, nfeed, nant, nfeed, nant))
    return vis.transpose((0, 1, 3, 2, 4)) if polarized else vis[:, 0, :, 0, :]


def do_beam_interpolation(
    freq,
    beam_list,
    polarized,
    nthreads,
    nax,
    nfeed,
    complex_dtype,
    nbeam,
    use_uvbeam,
    daz,
    dza,
    beam_interp,
    beam_data_gpu,
    events,
    cc,
    stream,
    tx,
    ty,
    nsrcs_up,
):
    """Perform the beam interpolation, choosing between CPU and GPU as necessary."""
    if use_uvbeam:  # perform interpolation on GPU
        az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")

        A_gpu = gpu_beam_interpolation(
            beam_data_gpu,
            daz,
            dza,
            az,
            za,
            gpu_func=beam_interp,
            nthreads=nthreads,
            stream=stream,
            return_on_cpu=False,
        )
        events[cc]["interpolation"].record(stream)
    else:
        A_gpu = gpuarray.empty(shape=(nax, nfeed, nbeam, nsrcs_up), dtype=complex_dtype)
        A_s = np.zeros((nax, nfeed, nbeam, nsrcs_up), dtype=complex_dtype)

        _evaluate_beam_cpu(
            A_s,
            beam_list,
            tx,
            ty,
            polarized,
            freq,
        )
        A_gpu.set(A_s)
    return A_gpu


def gpu_beam_interpolation(
    beam: np.ndarray | gpuarray.GPUArray,
    daz: float,
    dza: float,
    az: np.ndarray,
    za: np.ndarray,
    gpu_func: Callable | None = None,
    nthreads: int = 1024,
    stream=None,
    return_on_cpu: bool = True,
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
    gpu_func
        The callable, compiled GPU function to use. This is generated dynamically
        if not given, but can be given to avoid recompilation.
    nthreads
        The number of threads to use.
    stream
        An option GPU stream to write to.
    return_on_cpu
        Whether to return the result as CPU memory, or a GPUArray.

    Returns
    -------
    beam_at_src
        The beam interpolated at the sources. The shape of the array is
        ``(nbeam, nax, nfeed, nsrc)``. The array is always complex (at single or
        double precision, depending on the input).
    """
    # Get precision from the beam object.
    if beam.dtype in (np.dtype("float32"), np.dtype("complex64")):
        rtype, ctype = np.dtype("float32"), np.dtype("complex64")
    elif beam.dtype in (np.dtype("float64"), np.dtype("complex128")):
        rtype, ctype = np.dtype("float64"), np.dtype("complex128")
    else:
        raise ValueError(
            f"Got {beam.dtype} as the dtype for beam, which is unrecognized"
        )

    complex_beam = beam.dtype.name.startswith("complex")

    # Make everything the correct type.
    daz = rtype.type(daz)
    dza = rtype.type(dza)
    az = az.astype(rtype, copy=False)
    za = za.astype(rtype, copy=False)

    if not isinstance(beam, gpuarray.GPUArray):
        if stream is not None:
            beam = gpuarray.to_gpu_async(beam, stream=stream)
        else:
            beam = gpuarray.to_gpu(beam)

    nbeam, nax, nfeed, nza, naz = beam.shape
    nsrc = len(az)

    if gpu_func is None:
        beam_interp_code = BeamInterpTemplate.render(
            **{
                "DTYPE": TYPE_MAP[rtype],
                "NBEAM": nbeam,
                "BEAM_N_AZ": naz,
                "BEAM_N_ZA": nza,
                "DAZ": daz,
                "DZA": dza,
                "NAX": nax,
                "NFEED": nfeed,
            }
        )
        beam_interp_module = compiler.SourceModule(beam_interp_code)
        gpu_func = beam_interp_module.get_function("InterpolateBeamAltAz")

    block = (
        max(1, nthreads // nbeam // (nax * nfeed)),
        min(nthreads, nbeam),
        nax * nfeed,
    )
    grid = (
        int(np.ceil(nsrc / float(block[0]))),
        int(np.ceil(nbeam / float(block[1]))),
        int(np.ceil(nax * nfeed / float(block[2]))),
    )

    az_gpu = gpuarray.to_gpu_async(az, stream=stream)
    za_gpu = gpuarray.to_gpu_async(za, stream=stream)
    nsrc_uint = np.uint(nsrc)

    def fnc(beam_in, beam_out):
        gpu_func(
            az_gpu,
            za_gpu,
            beam_in,
            nsrc_uint,
            beam_out,
            block=block,
            grid=grid,
            stream=stream,
        )

    if complex_beam:
        beam_at_src_rl = gpuarray.empty(shape=(nax, nfeed, nbeam, nsrc), dtype=rtype)
        beam_at_src_im = gpuarray.empty(shape=(nax, nfeed, nbeam, nsrc), dtype=rtype)

        fnc(beam.real, beam_at_src_rl)
        fnc(beam.imag, beam_at_src_im)
        beam_at_src = beam_at_src_rl + 1j * beam_at_src_im
    else:
        beam_at_src = gpuarray.empty(shape=(nax, nfeed, nbeam, nsrc), dtype=rtype)

        fnc(beam, beam_at_src)
        cm.sqrt(beam_at_src, out=beam_at_src)
        beam_at_src = beam_at_src.astype(ctype)

    if return_on_cpu:
        return beam_at_src.get_async(stream=stream)
    else:
        return beam_at_src


vis_gpu.__doc__ += vis_cpu.__doc__
