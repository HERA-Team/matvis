"""JAX implementation of the visibility simulator."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import logging
import numpy as np
import psutil
import time
import warnings
from collections.abc import Sequence
from pyuvdata import UVBeam
from typing import Callable

from . import conversions
from ._uvbeam_to_raw import uvbeam_to_azza_grid
from .cpu import (
    _evaluate_beam_cpu,
    _log_progress,
    _validate_inputs,
    _wrangle_beams,
    vis_cpu,
)
from .gpu import _get_required_chunks

logger = logging.getLogger(__name__)


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
    use_redundancy: bool = False,
) -> np.ndarray:
    """JAX implementation of the visibility simulator."""
    pr = psutil.Process()
    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, eq2tops, crd_eq, I_sky
    )

    if beam_spline_opts:
        warnings.warn(
            "You have passed beam_spline_opts, but these are not used in JAX.",
            stacklevel=1,
        )

    nsrc = len(I_sky)

    if precision == 1:
        real_dtype, complex_dtype = np.float32, np.complex64
    else:
        real_dtype, complex_dtype = np.float64, np.complex128

    # apply scalars so 1j*tau*freq is the correct exponent
    ang_freq = 2 * freq * np.pi

    # ensure data types
    antpos = jax.device_put(antpos.astype(real_dtype))
    eq2tops = eq2tops.astype(real_dtype)
    crd_eq = jax.device_put(crd_eq.astype(real_dtype))
    Isqrt = jnp.sqrt(0.5 * I_sky.astype(real_dtype))

    beam_list, nbeam, beam_idx = _wrangle_beams(
        beam_idx=beam_idx,
        beam_list=beam_list,
        polarized=polarized,
        nant=nant,
        freq=freq,
    )

    total_beam_pix = sum(
        beam.data_array.shape[-2] * beam.data_array.shape[-1]
        for beam in beam_list
        if hasattr(beam, "data_array")
    )

    nchunks = min(
        max(
            min_chunks,
            _get_required_chunks(
                nax, nfeed, nant, nsrc, nbeam, total_beam_pix, precision
            ),
        ),
        nsrc,
    )

    npixc = nsrc // nchunks

    use_uvbeam = isinstance(beam_list[0], UVBeam)
    if use_uvbeam and not all(isinstance(b, UVBeam) for b in beam_list):
        raise ValueError(
            "vis_jax only support beam_lists with either all UVBeam or all AnalyticBeam objects."
        )

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

    # Send the regular-grid beam data to the GPU. This has dimensions
    # (Nbeam, Nax, Nfeed, Nza, Nza)
    # Note that Nbeam is not in general equal to Nant (we can have multiple antennas with
    # the same beam).
    if use_uvbeam:
        beam_data_gpu = jax.device_put(
            np.array(raw_beam_data.astype(complex_dtype if polarized else real_dtype)),
        )
    else:
        beam_data_gpu = None

    # will be set on GPU by bm_interp
    crd_eq_gpu = jnp.empty(shape=(3, npixc), dtype=real_dtype)
    # sent from CPU each time
    eq2top_gpu = jnp.empty(shape=(3, 3), dtype=real_dtype)
    # will be set on GPU
    vis_chunks = [
        jnp.empty(shape=(nfeed * nant, nfeed * nant), dtype=complex_dtype)
        for _ in range(nchunks)
    ]

    vis = np.full((ntimes, nfeed * nant, nfeed * nant), 0.0, dtype=complex_dtype)

    logger.info(f"Running With {nchunks} chunks")

    report_chunk = ntimes // 100 + 1
    pr = psutil.Process()
    tstart = time.time()
    mlast = pr.memory_info().rss
    plast = tstart

    for t in range(ntimes):
        eq2top_gpu = jax.device_put(
            eq2tops[t]
        )  # defines sky orientation for this time step

        for c in range(nchunks):
            crd_eq_gpu = jax.device_put(crd_eq[:, c * npixc : (c + 1) * npixc])
            Isqrt_gpu = Isqrt[c * npixc : (c + 1) * npixc]

            crdtop = jnp.dot(eq2top_gpu, crd_eq_gpu)

            # tx, ty, tz = crdtop_gpu
            above_horizon = crdtop[:, 2] > 0
            tx = crdtop[above_horizon:, 0]
            ty = crdtop[above_horizon, 1]
            nsrcs_up = tx.size

            if nsrcs_up < 1:
                continue

            crdtop = crdtop[:, above_horizon]

            # Need to do this in polar coordinates, NOT (l,m), at least for
            # polarized beams. This is because at zenith, the Efield components are
            # discontinuous (in power they are continuous). When interpolating the
            # E-field components, you need to treat the zenith point differently
            # depending on which "side" of zenith you're on. This is doable in polar
            # coordinates, but not in Cartesian coordinates.
            A_gpu = do_beam_interpolation(
                freq,
                polarized,
                nax,
                nfeed,
                complex_dtype,
                use_uvbeam,
                daz,
                dza,
                tx,
                ty,
                beam_list=beam_list,
                beam_data_gpu=beam_data_gpu,
            )

            tau = -ang_freq * 1j * jnp.dot(antpos, crdtop)
            Z = jnp.einsum("l,ijkl,jl->ijkl", Isqrt_gpu, A_gpu, jnp.exp(tau))
            vis_chunks[c] = jnp.dot(Z, Z.T.conj())

        vis[t] = jax.device_get(jnp.sum(vis_chunks, axis=0))

        if not (t % report_chunk or t == ntimes - 1):
            plast, mlast = _log_progress(tstart, plast, t + 1, ntimes, pr, mlast)

    vis = vis.conj().reshape((ntimes, nfeed, nant, nfeed, nant))
    return vis.transpose((0, 1, 3, 2, 4)) if polarized else vis[:, 0, :, 0, :]


def do_beam_interpolation(
    freq,
    polarized,
    nax,
    nfeed,
    complex_dtype,
    use_uvbeam,
    daz,
    dza,
    tx,
    ty,
    beam_list=None,
    beam_data_gpu=None,
):
    """Perform the beam interpolation, choosing between CPU and GPU as necessary."""
    if use_uvbeam:  # perform interpolation on GPU
        az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")
        return jax_beam_interpolation(beam_data_gpu, daz, dza, az, za)
    else:
        A_s = np.zeros((nax, nfeed, len(beam_list), len(tx)), dtype=complex_dtype)

        _evaluate_beam_cpu(
            A_s,
            beam_list,
            tx,
            ty,
            polarized,
            freq,
        )
        return jax.device_put(A_s)


def jax_beam_interpolation(
    beam,
    daz: float,
    dza: float,
    az: np.ndarray,
    za: np.ndarray,
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
        ``(nbeam, nax, nfeed, nsrc)``. The array is always complex (at single or
        double precision, depending on the input).
    """
    complex_beam = beam.dtype.name.startswith("complex")

    az_pixel_coords = az / daz
    za_pixel_coords = za / dza

    # maybe v thing here.
    def fnc(beam_in):
        jsp.ndimage.map_coordinates(
            beam_in,
            jnp.array([az_pixel_coords, za_pixel_coords]),
            order=1,
            mode="wrap",  # TODO: check if this is correct
        )

    return fnc(beam) if complex_beam else jnp.sqrt(fnc(beam))
