==================
Beam Interpolation
==================

When a simulation uses a gridded beam (a ``UVBeam`` in ``az_za`` coordinates,
rather than an analytic beam), ``matvis`` has to evaluate that beam at the
position of every source, at every time step. This page describes the
interpolation schemes available for that step and how to choose between them.

Choosing an interpolation order
===============================

The scheme is selected with the ``order`` key of ``beam_spline_opts``:

.. code-block:: python

   from matvis import simulate_vis

   vis = simulate_vis(
       ...,
       beams=beams,
       beam_spline_opts={"order": 3},   # bicubic; the default is 1 (bilinear)
   )

Two orders have dedicated fused CUDA kernels on the GPU backend:

.. list-table::
   :header-rows: 1

   * - ``order``
     - Scheme
     - Notes
   * - 1 (default)
     - Bilinear
     - Cheapest. Four grid points per source.
   * - 3
     - Bicubic B-spline
     - Sixteen grid points per source, plus a one-off prefilter of the beam
       grid at setup. Matches ``scipy.ndimage.map_coordinates(order=3)``.

Any other order (0, 2, 4, 5) still works, but falls back to a generic
per-beam, per-feed, per-axis :func:`cupyx.scipy.ndimage.map_coordinates` loop
that issues a separate GPU launch for every plane. At production scale that is
hundreds of launches per source chunk and is *much* slower than either of the
fused kernels — orders 2, 4 and 5 are supported for completeness, not for
production use.

On the CPU backend, ``beam_spline_opts`` is passed straight through to
:meth:`pyuvdata.UVBeam.interp`, which uses ``scipy.ndimage.map_coordinates``
for all orders.

When cubic is worth it
======================

Cubic interpolation matters when the beam grid is coarse compared to the
structure in the beam. As a concrete measure, the bundled HERA dipole beam was
decimated and then interpolated back onto the native grid nodes that had been
thrown away, so the interpolated values could be compared against the values
actually there:

.. list-table::
   :header-rows: 1

   * - Grid spacing
     - Linear RMS error
     - Cubic RMS error
     - Improvement
   * - 4°
     - :math:`2.6 \times 10^{-3}`
     - :math:`4.5 \times 10^{-4}`
     - 5.8x
   * - 6°
     - :math:`5.8 \times 10^{-3}`
     - :math:`1.3 \times 10^{-3}`
     - 4.5x

(Errors are relative to the peak beam value.) Cubic buys roughly half an order
of magnitude in RMS accuracy for the same grid, or equivalently lets a coarser
grid reach the same accuracy — which matters because the raw beam grids are
themselves a significant memory cost when simulating many unique beams (see
:doc:`performance`).

The gain is much smaller in the *maximum* error than in the RMS: the worst
errors occur where the beam has genuine small-scale structure that neither
scheme can recover from the available samples. Cubic is not a substitute for
adequately sampling the beam in the first place.

Interpolating in the beam's own units is also worth keeping in mind: for a
power beam, ``matvis`` interpolates the power and takes the square root
afterwards (both orders do this, and so does the CPU backend). A cubic spline
can overshoot below zero near a deep null in a power beam, which produces
``nan`` after the square root; linear interpolation cannot. If you see ``nan``
appear only at order 3, this is the likely cause.

.. _interp-cost:

Cost
====

Cubic interpolation is roughly **1.8x** the GPU time of linear for the
interpolation stage itself, which takes the stage from ~12% to ~19% of total
GPU time at the production-slice configuration, for a ~9% increase in overall
runtime. See :ref:`interpolation-order` on the Performance page for the
measured numbers and the configurations they were taken at.

That ratio depends on how the sources in a chunk are distributed: the 4 × 4
neighbourhoods of nearby sources overlap in cache, so a chunk covering a small
patch of sky costs relatively less than one scattered over the whole sky. A
synthetic worst case — every source at an independent uniformly random
position — costs 2.6x rather than 1.8x.

The one-off costs at setup are small: the prefilter (below) takes ~0.4 s for
350 unique beams on a 180 × 360 grid — about a fifth of one integration — and
the coefficient array is 1.7% larger than the beam grid it replaces.

How the cubic path works
========================

Cubic interpolation is not simply "the bilinear kernel with more taps". A
cubic B-spline that passes exactly *through* the grid values has coefficients
that differ from those values, and computing them is a sequential recursion
over each grid axis — a poor fit for a GPU, and hopeless to redo for every
source chunk.

``matvis`` therefore splits the work in two:

1. **Prefilter** (once, during setup). :func:`matvis.gpu.beams.prefilter_beam`
   converts the beam grid into B-spline coefficients, returned as a
   :class:`~matvis.gpu.beams.BeamCoefficients`. This depends only on the beam,
   so it is paid once per simulation rather than once per source chunk. The
   result carries a one-node halo on each side of both grid axes, which is
   what the four-point stencil reaches into at the edges of the grid.
2. **Evaluate** (per source chunk). The ``bicubic`` CUDA kernel combines the
   coefficients in a 4 × 4 neighbourhood with the B-spline basis, for every
   (beam, feed, axis, source) combination in a single launch — the same fused
   structure as the bilinear kernel.

Both steps live in :mod:`matvis.gpu.beams`; the kernels are in
``src/matvis/gpu/kernels/beam_interp.cu``.

Calling the interpolator directly
---------------------------------

:func:`matvis.gpu.beams.gpu_beam_interpolation` can be used on its own. If you
call it repeatedly against the same beam, prefilter once and hand it the
resulting :class:`~matvis.gpu.beams.BeamCoefficients` each time:

.. code-block:: python

   from matvis.gpu.beams import gpu_beam_interpolation, prefilter_beam

   coeffs = prefilter_beam(beam)          # once
   for az, za in source_chunks:           # many times
       out = gpu_beam_interpolation(coeffs, daz, dza, azmin, az, za, order=3)

Passing the raw beam grid instead is still correct — the function prefilters
internally — but repeats that work on every call. The two are not confusable:
``prefilter_beam`` returns a wrapper type rather than a bare array, so
accidentally prefiltering twice (or not at all) raises instead of quietly
returning an over-smoothed beam.

.. _interp-boundaries:

Behaviour at the edges of the grid
==================================

Both fused kernels **clamp** out-of-range coordinates to the edge of the beam
grid: a source at a zenith angle beyond the last grid node gets the value at
that last node, rather than an extrapolated or zeroed value. This keeps the
two orders consistent with each other, and keeps sources just past the edge of
a beam's support pinned to the horizon value instead of falling off a cliff.

For any coordinate *inside* the grid, the cubic kernel reproduces
``scipy.ndimage.map_coordinates(..., order=3)`` to floating-point precision,
whatever ``mode`` scipy is given — the modes only differ outside the grid.
Outside it, the kernels differ from scipy deliberately:

- scipy's ``mode="constant"`` (its default, and what the CPU backend uses
  unless told otherwise) returns 0 outside the grid.
- scipy's ``mode="nearest"`` interpolates through a 12-node edge-replicated
  pad, so it rings slightly just outside the grid before saturating.
- ``matvis``'s fused kernels clamp immediately.

This only matters if your sources actually fall outside the beam grid. Note
that a beam whose azimuth grid stops short of 360° does *not* wrap: a source in
the gap is clamped to the last azimuth node rather than interpolated around to
the first. Sample azimuth over the full circle if that matters for your
simulation.
