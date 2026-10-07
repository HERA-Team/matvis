"""Simple example wrapper for basic usage of matvis."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AnalyticBeam
from pyuvdata.beam_interface import BeamInterface

from . import HAVE_GPU, cpu
from .core.beams import prepare_beam_unpolarized

if HAVE_GPU:
    from . import gpu

logger = logging.getLogger(__name__)


def simulate_vis(
    ants: dict[int, np.ndarray],
    fluxes: np.ndarray | None = None,
    ra: np.ndarray | None = None,
    dec: np.ndarray | None = None,
    freqs: np.ndarray | None = None,
    times: Time | None = None,
    beams: list[AnalyticBeam | UVBeam | BeamInterface] | None = None,
    telescope_loc: EarthLocation | None = None,
    polarized: bool | None = None,
    precision: Literal[1, 2] = 1,
    use_feed: Literal["x", "y"] = "x",
    use_gpu: bool = False,
    beam_spline_opts: dict | None = None,
    beam_idx: np.ndarray | None = None,
    antpairs: np.ndarray | list[tuple[int, int]] | None = None,
    antenna_blocks: list[tuple[np.ndarray, np.ndarray]] | None = None,
    source_buffer: float = 1.0,
    coord_method: Literal[
        "CoordinateRotationAstropy",
        "CoordinateRotationERFA",
        "GPUCoordinateRotationERFA",
    ] = "CoordinateRotationERFA",
    coord_method_params: dict | None = None,
    matprod_method: Literal[
        "MatMul",
        "VectorDot",
        "MatBlock",
        "CPUMatMul",
        "GPUMatMul",
        "CPUVectorDot",
        "GPUVectorDot",
        "CPUMatBlock",
        "GPUMatBlock",
    ] = "MatMul",
    *,
    stokes: np.ndarray | None = None,
    raise_on_negative_flux: bool | None = None,
    **backend_kwargs,
):
    r"""
    Run a basic simulation using ``matvis``.

    This wrapper handles the necessary coordinate conversions etc.

    Parameters
    ----------
    ants : dict
        Dictionary of antenna positions. The keys are the antenna names
        (integers) and the values are the Cartesian x,y,z positions of the
        antennas (in meters) relative to the array center. The order of the
        keys defines the antenna indices used by beam_idx, antpairs and output.
    fluxes : array_like, optional
        Stokes I fluxes, shape (NSRCS, NFREQS). Exactly one of fluxes or stokes
        must be provided.
    ra, dec : array_like
        Arrays of source RA and Dec positions in radians. RA goes from [0, 2 pi]
        and Dec from [-pi/2, +pi/2].
    freqs : array_like
        Frequency channels for the simulation, in Hz.
    times : astropy.Time instance
        Times of the observation (can be an array of times).
    beams : list of ``UVBeam``, ``AnalyticBeam`` or ``BeamInterface`` objects
        Beam objects to use for each antenna.
    telescope_loc
        An EarthLocation object representing the center of the array.
    polarized : bool, optional
        If True, use polarized beams and calculate all available linearly-
        polarized visibilities, e.g. V_nn, V_ne, V_en, V_ee. If left as
        ``None`` (default), inferred from ``stokes``: True when ``stokes``
        is given, False when ``fluxes`` is given. Passing ``polarized=False``
        together with ``stokes`` raises ``ValueError``, since a Stokes-Q/U/V
        sky cannot be represented by a single feed.
    precision : int, optional
        Which precision setting to use for :func:`~matvis`. If set to ``1``,
        uses the (``np.float32``, ``np.complex64``) dtypes. If set to ``2``,
        uses the (``np.float64``, ``np.complex128``) dtypes.
    use_feed
        Either 'x' or 'y'. Intended for ``polarized=False``, but currently not
        passed on: unpolarized simulations always use the 'x' feed (see Returns).
    use_gpu : bool, optional
        Whether to use the GPU for simulation.
    beam_spline_opts : dict, optional
        Options for interpolating gridded (``UVBeam``) beams. Passed through to
        :func:`scipy.ndimage.map_coordinates` on the CPU backend, and to its GPU
        equivalent on the GPU backend. Keys left out fall back to
        :data:`~matvis.core.beams.DEFAULT_SPLINE_OPTS`, currently
        ``{"order": 3, "mode": "nearest"}``.

        On the GPU backend only ``order`` 1 (bilinear) and 3 (bicubic) have
        dedicated fused kernels. Any other order falls back to a
        per-(beam, feed, axis) ``map_coordinates`` loop that issues a separate
        kernel launch for every combination -- hundreds per source chunk at
        production scale -- and is *much* slower than either fused kernel. Orders
        0, 2, 4 and 5 are supported for completeness, not for production use.

        See :doc:`/beam_interpolation` for how to choose an order, and for the
        behaviour at the edges of the beam grid.
    beam_idx
        An array of integers, of the same length as ``ants``. Each entry is for an
        antenna of the same index, and its value should be the index of the beam in
        the beam list that corresponds to the antenna.
    antpairs
        Pairs of antenna *indices* (positions in ``ants``, not antenna names) to
        calculate visibilities for, as an integer array of shape ``(Npairs, 2)``.
        Both orders of a pair may be requested. If None, all ``Nants**2``
        ordered pairs are calculated.
    antenna_blocks
        Advanced/optional. A list of ``(row_antenna_idx, col_antenna_idx)``
        integer-array tuples defining rectangular sub-matrix blocks to compute
        instead of the full antenna x antenna product; only used when
        ``matprod_method`` is ``"MatBlock"``/``"CPUMatBlock"``/``"GPUMatBlock"``.
        See :mod:`matvis.redundancy` for helpers to build one, and
        :class:`~matvis.cpu.matprod.CPUMatBlock` for the full rationale.
    source_buffer : float, optional
        The fraction of the total number of sources to use when allocating memory
        for the sources above horizon. For large numbers of sources, a fraction of
        ~0.55 should be sufficient.
    coord_method
        The method to use to transform coordinates from the equatorial to horizontal
        frame. The default is "CoordinateRotationERFA". Use
        "CoordinateRotationAstropy" for direct Astropy coordinate transforms.
    coord_method_params
        Parameters particular to the coordinate rotation method of choice. For example,
        for the CoordinateRotationERFA (and GPU version of the same) method, there
        is the parameter ``update_bcrs_every``, which should be a time in seconds, for
        which larger values speed up the computation.
    matprod_method
        The method to use for the final matrix multiplication. Default is 'MatMul',
        which simply uses matrix multiplication over the two full matrices. Currently,
        the other option is `VectorDot`, which uses a loop over the antenna pairs,
        computing the sum over sources as a vector dot product, which can be faster for
        large arrays where `antpairs` is small (possibly from high redundancy). You
        should run a performance test before changing this. If not CPU/GPU prefix is
        specified, it will be added automatically based on the value of `use_gpu`.
    stokes : array_like, optional
        Full Stokes parameters of shape (4, NSRCS, NFREQS) with [I, Q, U, V].
        Enables polarized sky model support via eigendecomposition of the
        coherency matrix. Setting ``stokes`` automatically enables
        ``polarized=True``; passing ``polarized=False`` alongside is an
        error. Exactly one of ``fluxes`` or ``stokes`` must be provided.
    raise_on_negative_flux : bool, optional
        How to handle sources with a negative coherency eigenvalue. If
        ``None`` (default), the choice depends on the sky-model mode:
        ``True`` when ``fluxes`` is given (an unpolarized sky with
        negative flux is almost always a bug) and ``False`` when
        ``stokes`` is given (EoR-like models can legitimately have
        negative Stokes I). Set explicitly to override.

    Returns
    -------
    vis : np.ndarray
        Complex visibilities in the units of ``fluxes``, ``complex64`` if
        ``precision=1`` and ``complex128`` if ``precision=2``. The shape is
        ``(Nfreqs, Ntimes, Npairs, Nfeeds, Nfeeds)`` if ``polarized`` is True and
        ``(Nfreqs, Ntimes, Npairs)`` otherwise.

        The pair axis follows ``antpairs``. If ``antpairs`` is None, it holds all
        ``Nants**2`` ordered pairs, with pair ``(i, j)`` at index
        ``i * Nants + j``. In both cases ``i`` and ``j`` are indices into
        ``ants``, not antenna names.

        With ``polarized=True``, pair ``(i, j)`` follows pyuvsim's convention:

        .. math::

            V_{ij} = \sum_{\rm sources} A_i \, C \, A_j^\dagger \,
            \exp\left(2 \pi i \nu \, (\mathbf{x}_j - \mathbf{x}_i) \cdot
            \hat{\mathbf{s}} / c\right),

        where :math:`A_i` is antenna ``i``'s Jones matrix in the source
        direction :math:`\hat{\mathbf{s}}`, indexed [feed, sky component],
        :math:`C` is the source coherency (:math:`I/2` times the identity for an
        unpolarized sky), and :math:`\mathbf{x}` are the
        positions in ``ants``. So ``vis[..., p, q]`` correlates feed ``p`` of
        antenna ``i`` with the conjugate of feed ``q`` of antenna ``j``, with
        feeds ordered as in the beams' ``feed_array``. In pyuvdata's naming this
        is the polarization ``feed_array[p] + feed_array[q]``: for example,
        ``"xy"`` is x on antenna ``i`` and y on antenna ``j``. The baseline
        vector (pyuvdata's uvw) is :math:`\mathbf{x}_j - \mathbf{x}_i`, and
        :math:`V_{ji} = V_{ij}^\dagger`.

        With ``polarized=False``, each pair holds

        .. math::

            V_{ij} = \sum_{\rm sources} \sqrt{P_i P_j} \, \frac{I}{2} \,
            \exp\left(2 \pi i \nu \, (\mathbf{x}_j - \mathbf{x}_i) \cdot
            \hat{\mathbf{s}} / c\right),

        where :math:`P` is the beam's power response for the 'x' feed, or the
        beam's own polarization if it is a single-polarization power beam. When
        all antennas share a beam, this equals the polarized ``"xx"`` result.

    See Also
    --------
    matvis.matvis_to_uvdata : Put this output into a ``UVData`` object.
    """
    required = {
        "ra": ra,
        "dec": dec,
        "freqs": freqs,
        "times": times,
        "beams": beams,
        "telescope_loc": telescope_loc,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise TypeError(f"Missing required arguments: {', '.join(missing)}")

    if use_gpu:
        if not HAVE_GPU:
            raise ImportError("You cannot use GPU without installing GPU-dependencies!")

        import cupy as cp

        device = cp.cuda.Device()
        attrs = device.attributes
        attrs = {str(k): v for k, v in attrs.items()}
        string = "\n\t".join(f"{k}: {v}" for k, v in attrs.items())
        logger.debug(f"""
            Your GPU has the following attributes:
            \t{string}
            """)

    fnc = gpu.simulate if use_gpu else cpu.simulate

    if (fluxes is None) == (stokes is None):
        raise ValueError("Provide exactly one of `fluxes` or `stokes` to simulate_vis.")

    if polarized is None:
        polarized = stokes is not None
    elif not polarized and stokes is not None:
        raise ValueError(
            "polarized=False is incompatible with stokes=... — "
            "stokes input implies polarized=True. "
            "Either omit `polarized` or set polarized=True."
        )

    if stokes is not None:
        assert stokes.shape == (
            4,
            ra.size,
            freqs.size,
        ), "The `stokes` array must have shape (4, NSRCS, NFREQS)."
    else:
        assert fluxes.shape == (
            ra.size,
            freqs.size,
        ), "The `fluxes` array must have shape (NSRCS, NFREQS)."

    if raise_on_negative_flux is None:
        raise_on_negative_flux = stokes is None

    # Determine precision
    complex_dtype = np.complex64 if precision == 1 else np.complex128

    # Get polarization information from beams
    if polarized:
        nfeeds = getattr(beams[0], "Nfeeds", 2)

    # Antenna x,y,z positions
    antpos = np.array([ants[k] for k in ants.keys()])
    nants = antpos.shape[0]

    skycoords = SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame="icrs")

    npairs = len(antpairs) if antpairs is not None else nants * nants
    if polarized:
        vis = np.zeros(
            (freqs.size, times.size, npairs, nfeeds, nfeeds), dtype=complex_dtype
        )
    else:
        vis = np.zeros((freqs.size, times.size, npairs), dtype=complex_dtype)

    if matprod_method in ["MatMul", "VectorDot", "MatBlock"]:
        matprod_method = f"GPU{matprod_method}" if use_gpu else f"CPU{matprod_method}"

    # Loop over frequencies and call matvis_cpu/gpu
    for i, freq in enumerate(freqs):
        if stokes is not None:
            per_freq_kwargs = {"stokes": stokes[:, :, i]}
        else:
            per_freq_kwargs = {"I_sky": fluxes[:, i]}
        vis[i] = fnc(
            antpos=antpos,
            freq=freq,
            times=times,
            skycoords=skycoords,
            telescope_loc=telescope_loc,
            beam_list=beams,
            precision=precision,
            polarized=polarized,
            beam_spline_opts=beam_spline_opts,
            beam_idx=beam_idx,
            antpairs=antpairs,
            antenna_blocks=antenna_blocks,
            source_buffer=source_buffer,
            matprod_method=matprod_method,
            coord_method=coord_method,
            coord_method_params=coord_method_params,
            raise_on_negative_flux=raise_on_negative_flux,
            **per_freq_kwargs,
            **backend_kwargs,
        )
    return vis
