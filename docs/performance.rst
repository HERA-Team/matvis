===========
Performance
===========

This page describes how ``matvis`` performance scales with simulation size,
gives measured rule-of-thumb numbers for estimating run times, and records a
changelog of changes that significantly affected performance.

Unless noted otherwise, all statements refer to the GPU implementation with
the recommended production settings: **single precision**, polarized (2 feeds
× 2 E-field axes), gridded (``UVBeam``) beams with linear interpolation, and
the ERFA coordinate method.

Where the time goes
===================

For each time and frequency, ``matvis`` performs five stages (see
:doc:`understanding_the_algorithm`). Their costs scale as:

.. list-table::
   :header-rows: 1

   * - Stage
     - Scaling (per time, per frequency)
     - Share at HERA scale [1]_
   * - Coordinate rotation
     - :math:`N_{\rm src}`
     - few %
   * - Horizon cut / chunk selection
     - :math:`N_{\rm src}`
     - few %
   * - Beam interpolation
     - :math:`N_{\rm beam} N_{\rm feed} N_{\rm ax} N_{\rm src}`
     - ~15%
   * - Phase factor + Z matrix
     - :math:`N_{\rm ant} N_{\rm feed} N_{\rm ax} N_{\rm src}`
     - ~12%
   * - Matrix product :math:`V = Z Z^\dagger`
     - :math:`(N_{\rm ant} N_{\rm feed})^2 N_{\rm ax} N_{\rm src}`
     - ~70%
   * - Total
     - × :math:`N_{\rm times} \times N_{\rm freq}`
     -

.. [1] Measured with per-chunk CUDA events at
   :math:`N_{\rm ant} = N_{\rm beam} = 350`, polarized, single precision,
   on an RTX A2000 (Ampere) laptop GPU. The matrix product uses the cuBLAS
   Hermitian rank-k routine (``cherk``) and runs at the library's roofline,
   so the ~70% share is a hard floor rather than overhead.

Because the matrix product dominates for large arrays, total time is
approximately **linear in the number of sources and quadratic in the number
of antennas** (the cross-over to :math:`N_{\rm ant}^2` domination happens
around 100–200 antennas). The number of *distinct* beams only affects the
beam-interpolation share, so simulating 350 unique beams costs only ~15%
more than one shared beam.

Rules of thumb
==============

Measured cost per integration (one time sample, one frequency) at the
canonical production-slice configuration — 350 antennas, 350 beams,
polarized, fp32, 10⁶ sources — via ``profiling/run-canonical.sh``:

.. list-table::
   :header-rows: 1

   * - Hardware
     - GPU time / integration
     - Wall time / integration
   * - RTX A2000 laptop (Ampere, 95 W class)
     - 2.0 s
     - 2.1 s
   * - GeForce GTX Titan X (Maxwell, 2015 workstation card)
     - TBD [2]_
     - ≲1.8 s [2]_
   * - Quadro RTX 5000 (Turing, 16 GB workstation card)
     - TBD [2]_
     - ~1.8 s [2]_

**GPU time** (``derived.gpu_time_per_integration``: median per-chunk CUDA
events × chunk count) measures the card and transfers between machines with
the same GPU. **Wall time** (``derived.steady_wall_per_integration``: median
per-integration wall time, excluding the first integration) adds host-side
work — coordinate rotation, Python dispatch — and so also depends on the
machine's CPU; the difference between the columns is the host overhead on
the benchmark machine.

.. [2] Measured before the profiling harness gained its warmup pass and
   median-based statistics; wall values are steady-state estimates from
   per-integration logs, and GPU-time medians were not recorded. To be
   re-measured.

Scale this linearly in :math:`N_{\rm src}` and quadratically in
:math:`N_{\rm ant}` (above ~200 antennas). Frequencies are embarrassingly
parallel and are typically run as separate jobs.

For a GEMM-dominated estimate on other hardware:

.. math::

   t_{\rm gemm} \approx \frac{8 \,(N_{\rm feed} N_{\rm ant})^2 \, N_{\rm ax} N_{\rm src}^{\rm alloc}}{R}

where :math:`R` is the achieved complex-GEMM rate of your GPU. ``matvis``
computes :math:`V = ZZ^\dagger` with ``cherk``, which does half the work of a
general GEMM; on the A2000 the *effective* :math:`R` is ~5.2 TFLOPS (fp32).
Measure :math:`R` for your own GPU and problem shape with
``profiling/gemm_experiments.py``.

.. important::

   :math:`N_{\rm src}^{\rm alloc}` above is the *allocated* number of sources
   per chunk, not the number above the horizon: padded buffer entries go
   through the GEMM too. The ``source_buffer`` parameter therefore multiplies
   the dominant cost directly. If your sky is roughly uniform (half below the
   horizon at any time), ``source_buffer=0.6`` is nearly a 2x saving over the
   default ``1.0``.

GEMM strategy: hardware dependence
-----------------------------------

``matvis`` computes the matrix product with the cuBLAS Hermitian rank-k
routine (``cherk``/``zherk``, half the FLOPs of a general GEMM) and, for the
redundant-baseline ``GPUVectorDot`` path, with ``cgemm3m`` (the Gauss 3M
algorithm, ~25% fewer real multiplies). Both are bound directly from
``libcublas`` since cupy doesn't expose them. **How much they help is
architecture-dependent** — measured at :math:`M=700, K=10^5` (350 antennas,
polarized, complex64):

.. list-table::
   :header-rows: 1

   * - GPU
     - cgemm (baseline)
     - cgemm3m
     - cherk
   * - RTX A2000 (Ampere)
     - 213 ms
     - 100 ms (2.1x)
     - 75 ms (2.8x)
   * - GeForce GTX Titan X (Maxwell)
     - 72 ms
     - 107 ms (0.7x — *slower*)
     - 71 ms (~1.0x — no measurable gain)
   * - Quadro RTX 5000 (Turing)
     - 80 ms
     - 51 ms (1.6x — *fastest here*)
     - 80 ms (~1.0x — no measurable gain)

Three data points, three different outcomes: both alternatives help
substantially (Ampere), neither helps (Maxwell), or only ``cgemm3m`` helps —
and by enough to matter (Turing: ``cherk`` is within noise of plain
``cgemm``, but ``cgemm3m`` is ~1.6x faster than either). Because ``matvis``'s
primary matrix-product path (``GPUMatMul``) always uses ``cherk``, the Turing
result is a real, currently-unrealized gain: matprod is 66% of chunk time in
the production-slice benchmark above, so switching to ``cgemm3m`` there would
cut chunk time by roughly 20–25%. ``cherk`` is never *worse* than ``cgemm``
in any of the three measurements, so it remains a safe default, but it is
clearly not always the *fastest* choice. There is currently no runtime
auto-selection between strategies (tracked in
`issue #136 <https://github.com/HERA-Team/matvis/issues/136>`_); until then,
check both with ``profiling/gemm_experiments.py`` before assuming ``cherk``
is optimal on a new GPU generation.

Precision
=========

Single precision is the recommended production mode: it is validated against
double precision in ``tests/test_precision_gpu.py`` (agreement to :math:`10^{-5}`
of the peak visibility at test scale), uses half the memory, and is at least
2x faster even on data-centre GPUs with strong fp64 (V100/A100). On
consumer/workstation GPUs, fp64 arithmetic runs at 1/32 of fp32 throughput,
so double precision there is 10-30x slower end-to-end.

Memory and chunking
===================

Device memory is dominated by the per-chunk :math:`Z` matrix and interpolated
beam array, each of size
:math:`N_{\rm ant/beam} N_{\rm feed} N_{\rm ax} N_{\rm src}^{\rm alloc}`
complex values, plus the raw beam grids
(:math:`N_{\rm beam} N_{\rm feed} N_{\rm ax} N_{\rm pix}`). Sources are
automatically chunked to fit free GPU memory (see ``min_chunks`` and
``memory_buffer``); chunking is cheap as long as chunks stay :math:`\gtrsim
10^4` sources, so large problems run fine on small GPUs.

Benchmarking your own configuration
===================================

The ``matvis profile`` CLI runs a synthetic simulation of any size and
writes a machine-readable ``summary-stats-*.json``::

    matvis profile -a 350 -b 350 -s 1000000 -t 4 --gpu \
        --interpolated-beam --single-precision --gpu-event-timing \
        --coord-method CoordinateRotationERFA -o outdir

The harness is designed so its headline numbers are robust out of the box:

- An untimed **warmup** simulation runs first (disable with ``--no-warmup``),
  so one-time costs — cupy kernel compilation, cuBLAS workspace allocation,
  ERFA/IERS cache loads — are paid before any timing starts. Without it, the
  first integration can be several times slower than steady state and
  contaminate every average.
- Per-integration wall times are recorded individually, and the headline
  ``derived.steady_wall_per_integration`` is the **median excluding the
  first integration**.
- Per-chunk CUDA-event stage timings (``--gpu-event-timing``) keep all
  samples and report **medians** alongside means;
  ``derived.gpu_time_per_integration`` is the median per-chunk total times
  the chunk count.

The three ``derived`` values (steady wall, GPU time, host overhead) are the
ones to quote and compare — they are what the Rules of Thumb table reports.

The ``profiling/`` directory in the repository contains canonical benchmark
configurations, GEMM/interpolation "speed of light" micro-benchmarks, and an
``nsys`` recipe (the GPU loop is annotated with NVTX ranges). See
``profiling/README.md``.

.. warning::

   The ``stages`` table in the JSON output comes from ``line_profiler``
   timing individual Python lines, but the GPU loop is asynchronous: a line
   can appear expensive simply because it's where the host next blocks on
   already-queued GPU work (especially "Coordinate Rotation", which shares
   its bucket with the horizon-cut's blocking sync — see issue #133). Use it
   only as a rough indicator for the CPU backend; for the GPU backend use
   the ``derived`` and ``run_stats.event_timing_ms`` values.

Performance changelog
=====================

Changes that significantly altered performance, newest first:

.. list-table::
   :header-rows: 1

   * - Version / PR
     - Change
     - Measured impact
   * - `PR #130 <https://github.com/HERA-Team/matvis/pull/130>`_ (July 2026)
     - GPU hot-path overhaul: Hermitian rank-k (``cherk``) matrix product and
       ``cgemm3m`` bound directly from cuBLAS; single fused bilinear
       beam-interpolation kernel (replacing ~1400 ``map_coordinates`` launches
       per chunk at 350 beams); fused Z-matrix kernel; single compute stream
       with no device syncs in the loop; fixed a silent complex128 promotion
       in the phase-factor matmul (which also caused OOMs); fixed
       single-precision gridded-beam support.
     - 7.7x per-chunk GPU time (505 → 65 ms), 7.4x steady-state wall time at
       350 antennas / 350 beams / polarized / fp32; GPU utilization ~35% →
       ~95% (RTX A2000).
   * - v1.3.0 (Dec 2023)
     - Complete architectural rewrite from PyCUDA + hand-written CUDA kernels
       to cupy, making the code far easier to maintain and extend.
     - Introduced host-side overheads (kernel-launch storms, per-chunk
       synchronization, Python loops) and a hidden double-precision phase
       matmul that the July 2026 overhaul removed; between these releases,
       GPU performance was substantially below the figures published in the
       ``matvis`` paper.
   * - pre-v1.3 (paper implementation)
     - Original PyCUDA implementation with fused measurement-equation kernel;
       basis of the performance results in
       `Kittiwisit et al. (2025) <https://doi.org/10.1093/rasti/rzaf001>`_
       (Fig. 6, V100).
     - Reference point: ~100x GPU speed-up over the CPU implementation at
       :math:`N_{\rm ant}=256`, :math:`N_{\rm src} \approx 5\times10^5`.
