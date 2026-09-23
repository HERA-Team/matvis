=========
Changelog
=========

Dev
===

Fixed
-----

- Source chunking was planned from ``Device().mem_info[0]``, i.e. free device
  memory as the *driver* sees it. cupy keeps freed blocks in its own pool
  rather than returning them, so every ``gpu.simulate`` call after the first in
  a process saw a fraction of the card free and chunked far more finely than
  necessary -- at the production slice, 100 chunks instead of the 30 requested.
  Because ``simulate_vis`` calls the backend once per channel, a
  multi-frequency run could use a different chunk size for each channel.
  Availability is now computed as driver-free plus the pool's free blocks.

Performance
-----------

- GPU source chunks now accumulate straight into a single device visibility
  buffer (via ``beta=1`` in ``cherk``) instead of each chunk filling its own
  buffer that is summed at the end of the integration. The Hermitian mirror
  kernel runs once per integration rather than per chunk, the transpose into
  output ordering happens on the device, and the result is staged through a
  pinned host buffer. ``sum_chunks`` drops from 13.9 ms to 1.0 ms per
  integration at 350 antennas / 30 chunks / single precision; the ``beta=1``
  accumulation costs the matrix product ~0.15 ms per chunk, so the net saving
  is ~8 ms per integration (~0.4% of wall time on an RTX A2000). Device
  memory held for visibility buffers no longer scales with the chunk count
  (118 MB → 8 MB in that configuration); this only occasionally changes the
  auto-chunking decision at 350 antennas (e.g. 24 → 22 chunks with 2 GB
  free), but the buffers previously grew with the very chunk count they
  helped determine, and that term dominates for larger arrays.
- Major GPU hot-path overhaul (~7.7x faster per chunk at 350 antennas / 350
  beams / polarized / single precision; see the new "Performance" docs page):

  - Matrix product now uses the cuBLAS Hermitian rank-k routine
    (``cherk``/``zherk``, half the FLOPs) with ``cgemm3m`` for general
    products, bound directly from ``libcublas``.
  - Beam interpolation for gridded beams is a single fused bilinear kernel
    over all (beam, feed, axis) combos instead of one ``map_coordinates``
    launch per combo.
  - The Z matrix is computed in one fused kernel (previously several
    broadcast passes plus a Python loop over antennas).
  - The GPU loop runs on a single stream with no device synchronization,
    keeping the GPU ~95% utilized.
  - The phase-factor matmul no longer silently runs in complex128 when
    single precision is requested (this also removes a large hidden
    temporary array that could cause out-of-memory errors).

Changed
-------

- **Beam interpolation defaults are now shared by both backends**, in
  ``matvis.core.beams.DEFAULT_SPLINE_OPTS`` (``{"order": 3, "mode":
  "nearest"}``). Anything a caller leaves out of ``beam_spline_opts`` is taken
  from there. Two backend disagreements are resolved:

  - **Default order.** The GPU backend defaulted to ``order=1`` (bilinear)
    while the CPU backend defaulted to ``order=3``, having inherited it from
    ``scipy.ndimage.map_coordinates`` (and, before the switch to that routine,
    from ``RectBivariateSpline``'s ``kx=ky=3``). A simulation that did not set
    ``beam_spline_opts`` therefore used a different interpolant depending on
    the backend. Both now default to cubic. **GPU simulations of gridded
    beams that do not set** ``beam_spline_opts`` **will change**: more
    accurate, and ~9% slower overall at the production slice. Pass
    ``beam_spline_opts={"order": 1}`` to restore the previous GPU behaviour.
  - **Boundary mode**, now ``"mirror"`` on both backends. Since ``matvis``
    drops sources below the horizon before interpolating, it never evaluates a
    beam outside its grid, so the mode matters for one reason only: for
    ``order >= 2`` it selects the B-spline prefilter, and so changes
    interpolated values *inside* the grid within a few nodes of an edge. It is
    therefore chosen for accuracy just inside the edges. ``"mirror"`` is by far
    the best fit at the zenith pole — an edge every ``az_za`` beam has, and
    where the beam is brightest — measuring ~250x more accurate there than
    ``"nearest"`` on the bundled HERA dipole beam, at the cost of being ~2x
    worse at a horizon-truncated edge. The GPU's order-3 prefilter previously
    imposed ``"nearest"`` (12 nodes of edge replication); it now imposes mirror
    symmetry directly, which also drops that padding approximation and so
    matches scipy exactly rather than to ~1e-7. Asking the fused kernels for a
    different mode now raises instead of being silently ignored.

    The CPU backend's in-grid results are unchanged by this: scipy's default
    ``mode="constant"``, which it previously inherited, shares the ``"mirror"``
    prefilter. Pinning the mode makes that agreement explicit rather than
    coincidental.

Fixed
-----

- Documentation: the Beam Interpolation page claimed that scipy's ``mode``
  affects only coordinates outside the beam grid. It does not for
  ``order >= 2`` — it selects the B-spline prefilter, and so changes
  interpolated values inside the grid near an edge. The page also no longer
  justifies the boundary treatment by what happens to sub-horizon sources
  (``matvis`` never evaluates one), and now documents the O(h) error that every
  symmetric boundary mode produces in the outermost grid cell, which matters
  only for beams truncated at the horizon.
- GPU: a source chunk skipped because it had no sources above the horizon no
  longer contributes the *previous* integration's visibilities. Previously
  each chunk kept its own buffer which was only overwritten when the chunk
  was actually computed, but was summed unconditionally.
- Better handling of errors when GPUs are present but currently unavailable for some
  reason.
- Single-precision GPU simulations with gridded (``UVBeam``) beams no longer
  crash on a dtype mismatch when uploading beam data.
- GPU buffer sizes now respect the coordinate rotator's ``nsrc_alloc`` (which
  ignores ``source_buffer`` for chunks of fewer than 1000 sources),
  preventing shape-mismatch errors in small simulations.

Infrastructure
--------------

- ``matvis profile`` writes machine-readable ``summary-stats-*.json``
  (including per-stage CUDA-event timings with ``--gpu-event-timing``), and
  the GPU loop is annotated with NVTX ranges for ``nsys``. Canonical
  benchmark configs and roofline micro-benchmarks live in ``profiling/``.
- ``matvis profile`` reports ``sum_chunks`` as its own stage, both in the
  line-profiler table and as ``derived.sum_chunks_per_integration``, which is
  measured after an explicit stream drain so it excludes time spent waiting
  on the queued chunk pipeline.
- ``matvis profile`` frees the warmup simulation's device memory before the
  timed run, and warns when auto-chunking used more chunks than ``--nchunks``
  requested (recorded as ``nchunks_used`` in the JSON). Previously the warmup's
  retained buffers could make the timed run see only a fraction of the card
  free and silently pick a much larger chunk count, changing the workload
  being measured.
- The profiling harness is robust to one-time costs and host noise: an
  untimed warmup simulation runs first (``--no-warmup`` to disable),
  per-integration wall times are recorded individually, CUDA-event stage
  timings report medians as well as means and standard deviations, and a ``derived``
  block in the JSON separates steady-state wall time, GPU-only time, and host overhead
  per integration.

Tests
-----

- Correct formation of SkyModel for ``pyradiosky>=0.3.0`` in tests.
- Re-enabled the CPU-vs-GPU parity test suite, which had been silently
  skipped since the move away from ``pycuda`` (it still guarded on
  ``importorskip("pycuda")``); extended it to single precision.
- New ``tests/test_precision_gpu.py`` validating single- against double-precision
  results end-to-end on both backends.

Version 1.0.1
=============

Fixed
-----

- When getting the raw beam data for GPU, there was a check for whether the beam covers
  the whole sky which didn't always pass when it should have. This has been fixed.

Performance
-----------

- Added the ability to stop checks on whether the beam interpolates to inf/nan.

Version 1.0.0
=============

Version 1.0 is a major update that brings the GPU implementation up to the same API
as the CPU implementation. It also *removes* support for (l,m)-grid beams.

Removed
-------

- Support for ``bm_pix`` and ``use_pixel_beams`` (in both CPU and GPU implementations).
  Now, using a ``UVBeam`` object will automatically use interpolation on the gridded
  underlying data (which is typically in az/za). This can be done directly using
  methods in ``UVBeam``, or via new GPU methods. If you input an ``AnalyticBeam``, the
  beam will instead just merely be evaluated.

Added
-----

- Polarization support for GPU implementation.

Changed
-------

- Faster performance if using ``beam_list`` and the frequency is not in the ``freq_array``.
  (interpolation done before the loop).
- Factor of ~10x speed-up of ``vis_cpu`` due to changing the final ``einsum`` into a
  matrix product.
- **BREAKING CHANGE:** the output from the CPU and GPU implementations has changed
  shape: it is now ``(Ntimes, Nfeed, Nfeed, Nant, Nant)`` (and without the feed axes
  for non-polarized data).

Internals
---------

- ``vis_cpu`` and ``vis_gpu`` *modules* renamed to ``cpu`` and ``gpu`` respectively, to
  avoid some problems with name clashes.
- New more comprehensive tests comparing the GPU and CPU implementations against
  each other and against pyuvsim.
- New tests of documentation notebooks to ensure they're up to date.

Documentation
-------------

- Updated sphinx them to Furo.
- More complete Module Reference documentation.
- Updated tutorial to match the new API.
- Added a new "Understanding the Algorithm" page (with math!)

Version 0.4.3
=============

Changed
-------

- Call ``UVBeam.interp`` with ``reuse_spline=True`` and ``check_azza_domain=False`` for
  significantly faster performance when using ``beam_list``.

Version 0.4.2
=============

Fixed
-----

- The visibility integral, calculated with the call to ``einsum``, has been fixed.
  It now takes an outer product over feeds, sums over E-field components, and performs
  the integral over the sky.

Version 0.4.0
=============

Changed
-------

- Enhanced performance by allowing unique beams only to be passed (no breaking API
  change).
- Enhanced performance of ``vis_cpu`` by only using sources above horizon, and changing
  some array multiplication strategies (factor of ~3).

Version 0.2.3
=============

Fixed
-----

- Fix issue with spurious beam normalization when a pixel beam
  interpolation grid is generated from a UVBeam object
- Fix bug where the imaginary part of complex pixel beams was
  being discarded
- Fix bug that was causing polarized calculations to fail with
  ``simulate_vis``
- CI paths fixed so coverage reports are linked properly

Added
-----

- New units tests

Version 0.2.2
=============

Fixed
-----

- Fix issue with complex primary beams being cast to real

Version 0.2.1
=============

Fixed
-----

- Make IPython import optional.

Version 0.2.0
=============

Changed
-------

- ``lm_to_az_za`` --> ``enu_to_az_za`` and added ``orientation`` parameter. Significant
  increase in documentation of this and related coordinate functions.
- Refactoring of construction of spline within main CPU routine to its own function:
  ``construct_pixel_beam_spline``.

Added
-----

- ``eci_to_enu_matrix`` function
- ``enu_to_eci_matrix`` function
- ``point_source_crd_eq`` function
- ``equatorial_to_eci_coords`` function
- ``uvbeam_to_lm`` function
- New ``plotting`` module with ``animate_source_map`` function.
- Ability to do **polarization**! (Only in ``vis_cpu`` for now, not GPU).
- New ``wrapper`` module with ``simulate_vis`` function that makes it easier to simulate
  over an array of frequencies and source positions in standard RA/DEC (i.e. it does
  the frequency loop, and calculates the rotation matrices for you). It is an *example*
  wrapper for the core engine.
- Many more unit tests.

Version 0.1.2
=============

Fixed
-----

- Installation of gpu extras fixed.

Version 0.1.1
=============

Fixed
-----

- Fix import logic for GPU.

Version 0.1.0
=============

- Port out of hera_sim.
