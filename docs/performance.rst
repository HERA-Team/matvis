===========
Performance
===========

This page describes how ``matvis`` performance scales with simulation size,
gives measured rule-of-thumb numbers for estimating run times, and records a
changelog of changes that significantly affected performance.

Unless noted otherwise, all statements refer to the GPU implementation with
the following settings: **single precision**, polarized (2 feeds
× 2 E-field axes), gridded (``UVBeam``) beams with linear interpolation
explicitly selected via ``beam_spline_opts={"order": 1}`` (the default is
cubic; see `Beam interpolation order`_ for its cost), and
the ERFA coordinate method with a large value set for ``update_bcrs_every`` so
that it doesn't dominate the runs.

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

.. _interpolation-order:

Beam interpolation order
========================

Bicubic interpolation is the default (see :doc:`beam_interpolation`); it reads
16 grid points per source instead of 4. The numbers everywhere else on this
page use linear interpolation, selected explicitly with
``beam_spline_opts={"order": 1}``, so that the rest of the page isolates the
other stages:

.. list-table::
   :header-rows: 1

   * - Configuration
     - Beam stage (linear)
     - Beam stage (cubic)
     - Beam share of GPU time
     - Total GPU time
   * - production-slice (350 ants/beams, :math:`10^6` sources, 30 chunks)
     - 8.2 ms/chunk
     - 14.8 ms/chunk (1.8x)
     - 12% → 19%
     - +9%
   * - dev (64 ants/beams, :math:`2\times10^5` sources)
     - 6.8 ms/chunk
     - 11.5 ms/chunk (1.7x)
     - 12% → 17%
     - +14%

Measured on an RTX A2000 laptop GPU with ``--gpu-event-timing``, polarized,
single precision, 180 × 360 beam grid; medians of four runs per order for the
production slice. The total is quoted as the beam-stage *delta* over the linear
chunk total (+6.6 ms on ~68 ms), because the matrix product's own run-to-run
jitter is larger than the effect being measured and swamps a direct
before/after comparison of totals.

.. note::

   Per-chunk stage timings are only comparable between runs at the same chunk
   size, and absolute values drift with the GPU's clock and thermal state — on
   a laptop card they moved by up to 20% between sessions. Use the ``tau`` and
   ``z`` stages as a control: they are unaffected by the interpolation order,
   so a pair of runs whose ``tau``/``z`` agree is a valid comparison. On that
   basis the 1.8x stage cost and the 12% → 19% share reproduced across two
   independent sessions (1.80x and 1.75x) even as the absolute milliseconds
   moved.

Reproduce with::

    matvis profile -a 350 -b 350 -s 1000000 -t 4 --nchunks 30 --gpu \
        --interpolated-beam --single-precision --gpu-event-timing \
        --coord-method CoordinateRotationERFA -f 1 --spline-order 3 \
        -o profiling/results

The 1.8x on the stage is much less than the 4x increase in grid points read,
because the stage is bound by the coefficient loads, and the 4 × 4
neighbourhoods of neighbouring sources overlap heavily in cache. The total-run
penalty is smaller again (~9%), because the matrix product still dominates —
so the *relative* cost of cubic falls as the array grows, and rises as the
source count per antenna falls.

Two one-off setup costs come with ``order=3``, both small:

- The spline **prefilter** (see :doc:`beam_interpolation`) takes ~0.4 s for 350
  unique beams on a 180 × 360 grid — under a fifth of a single integration,
  and it does not scale with the number of times, frequencies or sources.
- The coefficient array carries a one-node halo on each grid axis, making it
  1.7% larger than the beam grid it replaces (692 → 704 MiB at 350 beams).
  Negligible against the per-chunk terms discussed under `Memory and
  chunking`_. Beams are prefiltered one at a time as they reach the device, so
  setup never holds the raw grids and the coefficients simultaneously (peak
  720 MiB rather than 1408 MiB at 350 beams).

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

- ``derived.sum_chunks_per_integration`` measures the once-per-integration
  readout (completing the Hermitian matrix, reordering it, and copying it to
  the host). It is timed *after* an explicit stream drain, so unlike the
  line-profiler and NVTX views of the same call it excludes time spent
  waiting on the queued chunk pipeline.
- ``nchunks_used`` records what auto-chunking actually settled on.
  ``--nchunks`` is only a *minimum*: when device memory is tight the run can
  silently use many more chunks, which changes the per-chunk problem size and
  makes stage timings incomparable. The profiler prints a warning when this
  happens, and frees the warmup run's device buffers beforehand so the timed
  run sees the whole card.

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

   "Sum Chunks" is the clearest example of how badly this can mislead. Its
   ``stages`` entry once read ~73 ms per integration, but essentially all of
   that was the host waiting on the integration's queued chunk pipeline, plus
   the first integration's one-off allocations skewing a 4-sample mean. The
   ``derived.sum_chunks_per_integration`` value — measured after an explicit
   stream drain, and reported as a median excluding the first integration —
   put the true cost at 13.9 ms.

Performance changelog
=====================

Changes that significantly altered performance, newest first:

.. list-table::
   :header-rows: 1

   * - Version / PR
     - Change
     - Measured impact
   * - Bicubic beam interpolation (Sept 2026)
     - Added a fused bicubic-B-spline CUDA kernel for gridded beams
       (``beam_spline_opts={"order": 3}``), alongside a one-off spline
       prefilter at setup. Previously, any order other than 1 fell back to a
       per-(beam, feed, axis) ``map_coordinates`` loop. The GPU default order
       also moved from 1 to 3, matching what the CPU backend already did.
     - Beam-interpolation stage 1.8x slower than linear (12% → 19% of GPU
       time), ~+9% total runtime at the production slice — versus hundreds of
       kernel launches per chunk on the old fallback path. ~6x lower RMS
       interpolation error at 4° beam sampling.
   * - `issue #132 <https://github.com/HERA-Team/matvis/issues/132>`_
       (Sept 2026)
     - GPU chunk accumulation and visibility readout:

       - Source chunks accumulate directly into one device buffer via the
         ``beta=1`` argument of ``cherk``, instead of each chunk filling its
         own buffer that is summed at the end of the integration.
       - The Hermitian mirror kernel runs once per integration rather than
         once per chunk.
       - The transpose into output ordering happens on the device, and the
         result is staged through a pinned host buffer.
     - ``sum_chunks`` 13.9 → 1.0 ms per integration (14x) at 350 antennas /
       30 chunks / fp32 on an RTX A2000. The ``beta=1`` accumulation costs
       the matrix product ~0.15 ms per chunk, so the *net* saving is ~8 ms
       per integration — about 0.4% of a 2.0 s integration on that GPU, and
       an estimated ~0.6% on a V100-class card (the device-side parts of the
       old readout scale with memory bandwidth, but the PCIe copy and the
       host-side transpose it removed do not). Device memory for the
       visibility buffers is now independent of the chunk count
       (118 MB → 8 MB here); that shifts the auto-chunking decision only
       occasionally at this size (24 → 22 chunks with 2 GB free), but the
       term grows as :math:`N_{\rm chunk} (N_{\rm ant} N_{\rm feed})^2` and
       dominates for larger arrays. Also fixes a correctness bug: a chunk skipped because nothing in it
       was above the horizon used to contribute the *previous* integration's
       visibilities.
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
