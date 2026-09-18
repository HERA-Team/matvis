===========
Performance
===========

This page describes how ``matvis`` performance scales with simulation size,
gives measured rule-of-thumb numbers for estimating run times, and records a
changelog of changes that significantly affected performance.

Unless noted otherwise, all statements refer to the GPU implementation with
the following settings: **single precision**, polarized (2 feeds
× 2 E-field axes), gridded (``UVBeam``) beams with linear interpolation, and
the ERFA coordinate method at its default ``update_bcrs_every = 0``.

.. note::

   That default is the *exact*, most expensive setting: the light-deflection
   and aberration corrections, which are ~90% of the cost of a coordinate
   rotation, are recomputed at every integration. Earlier versions of this page
   claimed the benchmarks used a large ``update_bcrs_every`` instead; they never
   did, because ``matvis profile`` had no way to set it. It does now
   (``--update-bcrs-every``), but the numbers below are all at the exact
   setting. Loosening it to ~180 s is therefore a speed-up relative to what is
   reported here, not the other way around -- and at production scale on a GPU
   it is worth well under 1% of an integration, because coordinate rotation is
   not where the time goes (see the table below).

The simulations reported here were run with the ``matvis profile`` script,
documented at :doc:`cli`. This script outputs a JSON file with profiling
information in it, also documented at :doc:`cli`. Below we make
reference to some of the data in this JSON output (e.g.
``derived.gpu_time_per_integration``).

Throughout, we reference a "production-slice". By this we refer to a simulation
with 350 antennas (each with unique beams) and one million sources, simulated for
just one time and one channel (i.e. "production" scale refers to the large array
and number of sources, while the "slice" refers to the single time/frequency).
This simulation size is large enough that overheads are relatively negligible.


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

.. [1] Measured with per-chunk CUDA events for a "production-slice" on an RTX A2000
   (Ampere) laptop GPU. The matrix product uses the cuBLAS
   Hermitian rank-k routine (``cherk``) and was tested to run at the library's
   roofline (i.e. the theoretical maximum),
   so the ~70% share is a minimum for this setup, with minimal overhead.

Because the matrix product dominates for large arrays, total time is
approximately **linear in the number of sources and quadratic in the number
of antennas** (the cross-over to :math:`N_{\rm ant}^2` domination happens
around 100–200 antennas). The number of *distinct* beams only affects the
beam-interpolation share, so simulating 350 unique beams costs only ~15%
more than one shared beam.

Rules of thumb for the matrix product phase
===========================================

The matrix product phase is the dominant phase for interferometers of a realistic size
(~100 antennas or more). Here we list the measured and theoretical cost of this phase
per integration (one time sample, one frequency) at the
canonical production-slice configuration via ``profiling/run-canonical.sh`` for
some GPUs that were available (if you have your own GPU and check the peformance,
please report it to us so we can add it here)!

The theoretical minimum here is given by the number of floating point operations required
divided by the *advertised* performance (TFLOPS) of the card, :math:`P_{\rm theo}`. That is,

.. math::

  t_{\rm min} ({\rm sec}) = \frac{4 \,(N_{\rm feed} N_{\rm ant})^2 \, N_{\rm ax} N_{\rm src}^{\rm alloc}}{P_{\rm theo}} = \frac{3.92}{P_{\rm theo}}.

In this equation we have set :math:`N_{\rm feed}=N_{\rm ax}=2`, :math:`N_{\rm ant}=350`,
and :math:`N_{\rm src}^{\rm alloc}=10^6` (i.e. the production slice settings), and the
factor of four accounts for the data being complex valued.
Note that this theoretical minimum assumes that the matrix-multiply uses CHERK, with
half the operations of a standard GEMM, and also assumes single precision is being used.

In practice, the matrix multiply operation on a given card will not achieve the theoretical
FLOPS of the card. Let the actual throughput of the card for the GEMM operation be
:math:`R`. Then the time taken for a production slice is

.. math::

   t_{\rm gemm} \approx \frac{8 \,(N_{\rm feed} N_{\rm ant})^2 \, N_{\rm ax} N_{\rm src}^{\rm alloc}}{R}.

In this equation, the potential savings that come from using CHERK insead of the general
GEMM (theoretically up to a factor of two) are absorbed into the performance, :math:`R`.
That is, :math:`R` is the *effective* achieved FLOPS for a GEMM operation of this
shape/scale on a given GPU (and therefore could be higher, up to a factor of two, than
the advertised FLOPS of the card). You can measure :math:`R` for your own GPU with our
provided ``profiling/gemm_experiments.py`` script.

.. important::

   :math:`N_{\rm src}^{\rm alloc}` above is the *allocated* number of sources
   per chunk, not the number above the horizon: padded buffer entries go
   through the GEMM too. The ``source_buffer`` parameter therefore multiplies
   the dominant cost directly. If your sky is roughly uniform (half below the
   horizon at any time), ``source_buffer=0.6`` is nearly a 2x saving over the
   default ``1.0``. These results use the default source buffer.


.. list-table::
   :header-rows: 1

   * - Hardware
     - GPU time / integration
     - Wall time / integration
     - :math:`P_{\rm theo}` (TFLOPS)
     - :math:`t_{\rm min}`
     - Efficiency
   * - RTX A2000 laptop (Ampere, 95 W class)
     - 2.34 s
     - 2.37 s
     - 8
     - 0.49 s
     - 21%
   * - GeForce GTX Titan X (Maxwell, 2015 workstation card)
     - 1.6 s
     - 1.7 s
     - 6.6
     - 0.59 s
     - 37%
   * - Quadro RTX 5000 (Turing, 16 GB workstation card)
     - 1.8 s
     - 1.8 s
     - 11.2
     - 0.35 s
     - 19%
   * - Tesla V100-SXM2-32GB (Volta, data-centre)
     - 0.8 s
     - 0.8 s
     - 15.7
     - 0.25 s
     - 31%

**GPU time** (``derived.gpu_time_per_integration``: per-integration sum of
per-chunk CUDA event totals, median over integrations excluding the first)
measures the time spent computing on the GPU (and transferring data to/from
the GPU).
**Wall time** (``derived.steady_wall_per_integration``: median
per-integration wall time, excluding the first integration) adds host-side
work — coordinate rotation, Python dispatch — and so also depends on the
machine's CPU; the difference between the columns is the host overhead on
the benchmark machine (2% or less on all four machines above).
**Efficiency**: defined as the theoretical minimum time divided by the measured time,
as a percentage.

.. note::

   These efficiencies (19-37%) are well below 100%. This does not necessarily
   reflect that ``matvis`` is ineffeciently calling the matrix multiply routine (CHERK),
   but is more likely a reflection that CHERK (under the given matrix shape conditions)
   cannot achieve the theoretical peak performance of the card.
   For the A2000 row in particular, the ~70% matrix-product share quoted in
   `Where the time goes`_ was itself measured with ``cherk`` already running
   at *the library's roofline* [1]_ — cuBLAS could not do any better for this
   problem shape on that card, so the entire 100%→25% shortfall happens
   inside cuBLAS, not in the code calling it. Two effects are known to pull
   achieved throughput for complex GEMM/CHERK below a card's advertised
   (real, fp32) peak: complex-valued kernels generally reach a lower
   fraction of peak than a real SGEMM of the same size, and vendor-advertised
   TFLOPS are boost-clock figures rarely sustained under continuous load
   (especially on the 95 W laptop A2000). GPU time also includes device
   data transfer (see above), which further widens the gap from the
   compute-only theoretical minimum, though we have not separately measured
   how much of the gap this accounts for.


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
   * - Tesla V100-SXM2-32GB (Volta)
     - 36 ms
     - 21 ms (1.7x — *fastest here*)
     - 37 ms (~1.0x — no measurable gain)

As you can see, on the cards measured here, there is a significant difference between
the different matrix-multiply strategies, and which one is fastest is dependent on the
GPU. ``cherk`` is never *worse* than ``cgemm`` in any of
the four measurements, so it remains a safe default, but on the two most
modern architectures measured it captures none of the available speedup.
There is currently no runtime auto-selection between strategies (tracked in
`issue #136 <https://github.com/HERA-Team/matvis/issues/136>`_); until then, check
both with ``profiling/gemm_experiments.py`` before assuming ``cherk`` is optimal.

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

The raw beam-grid term doesn't scale with chunk size (it's the same whether
you have 1 chunk or 100), while the :math:`Z`/interpolated-beam terms scale
with :math:`N_{\rm src}^{\rm alloc}`, i.e. with the chunk size. For
production-scale runs (:math:`N_{\rm pix} \sim 6.5 \times 10^4` for
degree-scale beam sampling), the raw beam grids can dominate total memory
when chunks are small, but become a negligible fraction once chunks are
large enough that the chunk-scaled terms take over. Worth checking
explicitly if you're tuning ``min_chunks``/``memory_buffer`` on a
memory-constrained GPU with many unique beams.

Benchmarking your own configuration
===================================

The ``matvis profile`` CLI runs a synthetic simulation of any size and
writes a machine-readable ``summary-stats-*.json``::

    matvis profile -a 350 -b 350 -s 1000000 -t 4 --gpu \
        --interpolated-beam --single-precision --gpu-event-timing \
        --coord-method CoordinateRotationERFA -o outdir

The script is designed so its headline numbers are robust out of the box:

- An untimed **warmup** simulation runs first (disable with ``--no-warmup``),
  so one-time costs — cupy kernel compilation, cuBLAS workspace allocation,
  ERFA/IERS cache loads — are paid before any timing starts. Without it, the
  first integration can be several times slower than steady state and
  contaminate every average.
- Per-integration wall times are recorded individually, and the headline
  ``derived.steady_wall_per_integration`` is the **median excluding the
  first integration**.
- Per-chunk CUDA-event stage timings (``--gpu-event-timing``) keep all
  samples and report per-stage **medians** alongside means;
  ``derived.gpu_time_per_integration`` sums each integration's actual chunk
  totals and takes the **median across integrations, excluding the first**
  — the same warmup-robust treatment as the wall time above.

The three ``derived`` values (steady wall, GPU time, host overhead) are the
ones to quote and compare — they are what the Rules of Thumb table reports.

The ``profiling/`` directory in the repository contains canonical benchmark
configurations, GEMM/interpolation roofline micro-benchmarks (i.e. measures
of performance compared to the theoretical maximum), and an
``nsys`` recipe (the GPU loop is annotated with NVTX ranges). See
``profiling/README.md``.

.. warning::

   The ``stages`` table in the JSON output comes from ``line_profiler``
   timing individual Python lines, but the GPU loop is asynchronous: a line
   can appear expensive simply because it's where the host next blocks on
   already-queued GPU work (especially "Coordinate Rotation", which shares
   its bucket with the horizon-cut's blocking sync — see
   `issue #133 <https://github.com/HERA-Team/matvis/issues/133>`_). Use it
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
     - GPU hot-path overhaul:

       - Matrix product uses the cuBLAS Hermitian rank-k routine (``cherk``)
         and ``cgemm3m``, bound directly from cuBLAS.
       - Beam interpolation: replaced per-beam, per-feed, per-polarization
         ``map_coordinates`` calls (~1400 separate GPU launches per chunk at
         350 beams) with a single fused kernel launch that covers all of
         them at once ("fused" = combined into one GPU launch instead of
         many).
       - Z-matrix construction is also a single fused kernel.
       - The per-time, per-chunk simulation loop runs on a single compute
         stream with no device syncs.
       - Fixed a silent complex128 promotion in the phase-factor matmul
         (which also caused OOMs).
       - Fixed single-precision gridded-beam support.
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
