"""GPU implementation of the simulator."""
import logging
import numpy as np
from astropy.constants import c as speed_of_light
from jinja2 import Template
from pathlib import Path
from pyuvdata import UVBeam
from typing import Callable, Optional, Sequence

try:
    import pycuda.autoinit
    from pycuda import compiler, driver, gpuarray
    from skcuda.cublas import (
        cublasCreate,
        cublasDestroy,
        cublasDgemm,
        cublasSetStream,
        cublasSgemm,
    )

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False

from .vis_cpu import _evaluate_beam_cpu, _validate_inputs, _wrangle_beams, vis_cpu

logger = logging.getLogger(__name__)

ONE_OVER_C = 1.0 / speed_of_light.value


templates = Path(__file__).parent / "gpu_src"

with open(templates / "measurement_equation.c", "r") as fl:
    MeasEqTemplate = Template(fl.read())

with open(templates / "beam_interpolation.c", "r") as fl:
    BeamInterpTemplate = Template(fl.read())


def _numpy3d_to_array(np_array):
    """Copy a 3D (d,h,w) numpy array into a 3D pycuda array.

    Array can be used to set a texture.  (For some reason, gpuarrays can't
    be used to do that directly).  A transpose happens implicitly; the CUDA
    array has dim (w,h,d).
    """
    d, h, w = np_array.shape
    descr = driver.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    if np_array.dtype == np.float64:
        descr.format = driver.array_format.SIGNED_INT32
        descr.num_channels = 2
    else:
        descr.format = driver.dtype_to_array_format(np_array.dtype)
        descr.num_channels = 1
    descr.flags = 0
    device_array = driver.Array(descr)
    copy = driver.Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d
    copy()
    return device_array


NTHREADS = 512  # make 512 for smaller GPUs
MAX_MEMORY = 2**29  # floats (4B each)
MIN_CHUNK = 1


def vis_gpu(
    *,
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    beam_list: Optional[Sequence[UVBeam | Callable]],
    polarized: bool = False,
    beam_idx: Optional[np.ndarray] = None,
    nthreads: int = NTHREADS,
    max_memory: int = MAX_MEMORY,
    precision: int = 1,
) -> np.ndarray:
    """GPU implementation of the visibility simulator."""
    if not HAVE_CUDA:
        raise ImportError("You need to install the [gpu] extra to use this function!")

    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, eq2tops, crd_eq, I_sky
    )

    nsrc = len(I_sky)

    if precision == 1:
        real_dtype, complex_dtype = np.float32, np.complex64
        DTYPE, CDTYPE = "float", "cuFloatComplex"
        cublas_real_mm = cublasSgemm
    else:
        real_dtype, complex_dtype = np.float64, np.complex128
        DTYPE, CDTYPE = "double", "cuDoubleComplex"
        cublas_real_mm = cublasDgemm

    # apply scalars so 1j*tau*freq is the correct exponent
    ang_freq = 2 * freq * np.pi

    # ensure data types
    antpos = antpos.astype(real_dtype)
    eq2tops = eq2tops.astype(real_dtype)
    crd_eq = crd_eq.astype(real_dtype)
    Isqrt = np.sqrt(0.5 * I_sky).astype(real_dtype)

    chunk = max(
        min(nsrc, MIN_CHUNK),
        2 ** int(np.ceil(np.log2(float(nant * nsrc) / max_memory / 2))),
    )
    npixc = nsrc // chunk

    # blocks of threads are mapped to (pixels,ants,freqs)
    block = (max(1, nthreads // nant), min(nthreads, nant), 1)
    grid = (
        int(np.ceil(npixc / float(block[0]))),
        int(np.ceil(nant * nax * nfeed / float(block[1]))),
    )

    prod_grid = (
        int(np.ceil(nant * nfeed / float(block[0]))),
        int(np.ceil(nant * nfeed / float(block[1]))),
    )

    beam_list, nbeam, beam_idx = _wrangle_beams(
        beam_idx=beam_idx,
        beam_list=beam_list,
        polarized=polarized,
        nant=nant,
        freq=freq,
        nax=1,  # update if we can do polarization
        nfeed=1,  # update if we can do polarization
        interp=True,  # update later if we can.
    )

    cuda_params = {
        "NANT": nant,
        "NPIX": npixc,
        "NAX": nax,
        "NFEED": nfeed,
        "INTERP_FUNC": "",  # put in `interp_func` later
        "BLOCK_PX": block[0],
        "DTYPE": DTYPE,
        "CDTYPE": CDTYPE,
        "f": "f" if precision == 1 else "",
    }

    # Choose to use single or double precision CUDA code
    meas_eq_code = MeasEqTemplate.render(**cuda_params)
    # beam_interp_code = BeamInterpTemplate.render(**cuda_params)

    meas_eq_module = compiler.SourceModule(meas_eq_code)
    # bm_interp = gpu_module.get_function("InterpolateBeam")
    meas_eq = meas_eq_module.get_function("MeasEq")
    vis_inner_product = meas_eq_module.get_function("VisInnerProduct")

    # bm_texref = gpu_module.get_texref("bm_tex")
    h = cublasCreate()  # handle for managing cublas

    # define GPU buffers and transfer initial values
    # never changes, transpose happens in copy so cuda bm_tex is (BEAM_PX,BEAM_PX,NANT)
    # bm_texref.set_array(numpy3d_to_array(beams))
    antpos_gpu = gpuarray.to_gpu(antpos)  # never changes, set to -2*pi*antpos/c
    Isqrt_gpu = gpuarray.empty(shape=(npixc,), dtype=real_dtype)

    # will be set on GPU by bm_interp
    crd_eq_gpu = gpuarray.empty(shape=(3, npixc), dtype=real_dtype)
    # sent from CPU each time
    eq2top_gpu = gpuarray.empty(shape=(3, 3), dtype=real_dtype)
    # will be set on GPU
    crdtop_gpu = gpuarray.empty(shape=(3, npixc), dtype=real_dtype)
    # will be set on GPU
    vis_gpus = [
        gpuarray.empty(shape=(nfeed, nfeed, nant, nant), dtype=complex_dtype)
        for _ in range(chunk)
    ]

    # output CPU buffers for downloading answers
    vis_cpus = [
        np.empty(shape=(nfeed, nfeed, nant, nant), dtype=complex_dtype)
        for _ in range(chunk)
    ]
    streams = [driver.Stream() for _ in range(chunk)]
    event_order = (
        "start",
        "upload",
        "eq2top",
        "tau",
        #        "interpolate",
        "meas_eq",
        "vis",
        "end",
    )
    vis = np.empty((ntimes, nfeed, nfeed, nant, nant), dtype=complex_dtype)

    logging.info("Running With %s chunks: ", chunk)

    for t in range(ntimes):
        eq2top_gpu.set(eq2tops[t])  # defines sky orientation for this time step
        events = [{e: driver.Event() for e in event_order} for _ in range(chunk)]

        for c in range(chunk + 2):
            cc = c - 1
            ccc = c - 2
            if 0 <= ccc < chunk:
                stream = streams[ccc]
                vis_gpus[ccc].get_async(ary=vis_cpus[ccc], stream=stream)
                events[ccc]["end"].record(stream)
            if 0 <= cc < chunk:
                stream = streams[cc]
                cublasSetStream(h, stream.handle)
                ## compute crdtop = dot(eq2top,crd_eq)
                # cublas arrays are in Fortran order, so P=M*N is actually
                # peformed as P.T = N.T * M.T
                cublas_real_mm(
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

                tx, ty, tz = crdtop_gpu.get()
                above_horizon = tz > 0
                tx = tx[above_horizon]
                ty = ty[above_horizon]
                nsrcs_up = len(tx)
                crdtop_lim_gpu = gpuarray.to_gpu(
                    crdtop_gpu.get()[:, above_horizon].copy()
                )  # empty(shape=(3, nsrcs_up), dtype=real_dtype)

                if nsrcs_up < 1:
                    continue

                tau_gpu = gpuarray.empty(shape=(nant, nsrcs_up), dtype=real_dtype)

                ## compute tau = dot(antpos,crdtop) / speed_of_light
                cublas_real_mm(
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

                ## interpolate bm_tex at specified topocentric coords, store interpolation in A
                ## threads are parallelized across pixel axis
                ## Need to do this in polar coordinates or BAD THINGS HAPPEN

                A_gpu = gpuarray.empty(
                    shape=(nax, nfeed, nant, nsrcs_up), dtype=complex_dtype
                )

                A_s = _evaluate_beam_cpu(
                    beam_list,
                    tx,
                    ty,
                    polarized,
                    nbeam,
                    nax,
                    nfeed,
                    freq,
                    np.uint(nsrcs_up),
                    complex_dtype,
                )

                # bm_interp(crdtop_gpu, A_gpu, grid=grid, block=block, stream=stream)
                # events[cc]["interpolate"].record(stream)

                v_gpu = gpuarray.empty(
                    shape=(nax, nfeed, nant, nsrcs_up), dtype=complex_dtype
                )
                Isqrt_lim_gpu = gpuarray.to_gpu(Isqrt_gpu.get()[above_horizon].copy())

                # compute v = A * sqrtI * exp(1j*tau*freq)
                A_gpu.set(A_s)
                meas_eq(
                    A_gpu,
                    Isqrt_lim_gpu,
                    tau_gpu,
                    real_dtype(ang_freq),
                    np.uint(nsrcs_up),
                    v_gpu,
                    grid=grid,
                    block=block,
                    stream=stream,
                )
                events[cc]["meas_eq"].record(stream)

                # compute vis = dot(v, v.T)
                # We want to take an outer product over feeds/antennas, contract over
                # E-field components, and integrate over the sky.
                vis_inner_product(
                    v_gpu.gpudata,
                    np.uint(nsrcs_up),
                    vis_gpus[cc].gpudata,
                    grid=prod_grid,
                    block=block,
                )

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
    return vis if polarized else vis[:, 0, 0, :, :]


vis_gpu.__doc__ += vis_cpu.__doc__
