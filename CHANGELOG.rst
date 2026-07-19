=========
Changelog
=========

Dev
===

Performance
-----------

- Major GPU hot-path overhaul (~7.7x faster per chunk at 350 antennas / 350
  beams / polarized / single precision; see the new "Performance" docs page):

  - Matrix product now uses the cuBLAS Hermitian rank-k routine
    (``cherk``/``zherk``, half the FLOPs) with ``cgemm3m`` for general
    products, bound directly from ``libcublas``.
  - Beam interpolation for gridded beams is a single fused bilinear kernel
    over all (beam, feed, axis) planes instead of one ``map_coordinates``
    launch per plane.
  - The Z matrix is computed in one fused kernel (previously several
    broadcast passes plus a Python loop over antennas).
  - The GPU loop runs on a single stream with no device synchronization,
    keeping the GPU ~95% utilized.
  - The phase-factor matmul no longer silently runs in complex128 when
    single precision is requested (this also removes a large hidden
    temporary that could cause out-of-memory errors).

Fixed
-----

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
- The profiling harness is robust to one-time costs and host noise: an
  untimed warmup simulation runs first (``--no-warmup`` to disable),
  per-integration wall times are recorded individually, CUDA-event stage
  timings report medians as well as means, and a ``derived`` block in the
  JSON separates steady-state wall time, GPU-only time, and host overhead
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
