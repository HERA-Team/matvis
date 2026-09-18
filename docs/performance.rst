===========
Performance
===========

This page describes how ``matvis`` performance scales with simulation size,
gives measured rule-of-thumb numbers for estimating run times, and records a
changelog of changes that significantly affected performance.

Unless noted otherwise, all statements refer to the GPU implementation with
the following settings: **single precision**, polarized (2 feeds
× 2 E-field axes), gridded (``UVBeam``) beams with linear interpolation, and
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

Block-decomposed products on redundant arrays
=============================================

A *redundant* array measures far fewer distinct baselines than it has antenna
pairs, so the full :math:`N_{\rm ant} \times N_{\rm ant}` product computes many
visibilities that are, by construction, copies of one another. The
``MatBlock`` matprod methods compute a handful of rectangular antenna-index
sub-matrix products instead, gathering just the requested ``antpairs`` out of
them (see :doc:`understanding_the_algorithm` for the mechanism, and
:func:`~matvis.redundancy.find_dense_blocks` for building the decomposition).

.. important::

   **This only helps if the array is redundant.** The saving comes entirely
   from asking for fewer visibilities than :math:`N_{\rm ant}^2`; the
   decomposition cannot create redundancy that isn't there. In particular, if
   every antenna has its own beam, no two antenna pairs give the same
   visibility, every pair is wanted, and the best possible decomposition is
   the full product itself — with the extra costs measured below on top. The
   control measurement at the end of this section shows this case running
   **1.4x slower** than plain ``MatMul``. Use ``MatBlock`` only when
   ``len(antpairs)`` is a small fraction of :math:`N_{\rm ant}^2`.

Measured speedup
----------------

Benchmark configuration: ``matvis hera-profile -a 11`` — a HERA-like split-core
hex layout with **320 antennas**, i.e. 102 400 antenna pairs but only **1 501
unique baselines** (a redundancy factor of 68) — with 196 608 sources
(``--nside 128``), 6 source chunks (~32.8k sources per chunk, matching the
canonical production-slice chunk size), one shared beam, gridded/interpolated,
polarized, single precision, ``CoordinateRotationERFA``, 6 integrations, on an
**RTX A2000 laptop GPU**. "Area" is :math:`\sum_b N^b_{\rm row} N^b_{\rm col}`
summed over blocks, the quantity the FLOP count is proportional to; the full
product's area is :math:`N_{\rm ant}^2 = 102\,400`. "Matrix product" is the
``matprod`` CUDA-event median per chunk; "Wall" is
``derived.steady_wall_per_integration``.

.. list-table::
   :header-rows: 1

   * - Method
     - Area (FLOP proxy)
     - Predicted from area
     - Matrix product
     - Wall / integration
     - Actual speedup
   * - ``MatMul`` (full product)
     - 102 400
     - 1.0x
     - 39.0 ms
     - 0.299 s
     - 1.00x
   * - ``MatBlock``, ``max_blocks=1``
     - 33 176
     - 3.1x
     - 25.8 ms
     - 0.218 s
     - 1.37x
   * - ``MatBlock``, ``max_blocks=2``
     - 7 623
     - 13.4x
     - 15.0 ms
     - 0.154 s
     - 1.94x
   * - ``MatBlock``, ``max_blocks=3``
     - 3 527
     - 29.0x
     - 12.2 ms
     - 0.140 s
     - 2.14x
   * - ``MatBlock``, ``max_blocks=4``
     - 2 707
     - 37.8x
     - **11.4 ms**
     - **0.130 s**
     - **2.31x**
   * - ``MatBlock``, ``max_blocks=6``
     - 1 920
     - 53.3x
     - 13.5 ms
     - 0.145 s
     - 2.06x
   * - ``MatBlock``, ``max_blocks=8``
     - 1 714
     - 59.7x
     - 13.7 ms
     - 0.148 s
     - 2.02x
   * - ``MatBlock``, ``max_blocks=12``
     - 1 542
     - 66.4x
     - 20.4 ms
     - 0.186 s
     - 1.61x
   * - ``VectorDot`` (one GEMM per baseline)
     - 1 501
     - 68.2x
     - 211.5 ms
     - 1.343 s
     - 0.22x (*4.5x slower*)

Building the decomposition is a one-off setup cost of 24 ms (1 block) to 140 ms
(12 blocks) at this array size — negligible against any real simulation, but it
is paid per ``simulate_vis`` call, so build it once and reuse it if you are
calling in a loop.

The headline is that the block decomposition **is** a real win —
2.3x end-to-end, 3.4x on the matrix product itself — but it realizes only about
9 per cent of the 37.8x that the FLOP count alone predicts, and the best
``max_blocks`` is **not** the one that minimizes FLOPs. Cutting past four blocks
keeps reducing the area and starts making things slower again.

Why the FLOP ratio isn't achievable
-----------------------------------

The full product is compute-bound; the block products are not. Each block reads
:math:`(N^b_{\rm row} + N^b_{\rm col})` antennas' worth of the :math:`Z` matrix
to produce an :math:`N^b_{\rm row} \times N^b_{\rm col}` result, and because
:math:`N_{\rm src}` is enormous compared to any block edge, these are extremely
"skinny" GEMMs (at ``max_blocks=4`` the blocks are 3x319, 10x38, 27x46 and
64x2 antennas, i.e. :math:`M` as small as 6 rows against :math:`K = 65\,536`).
Their cost is set by streaming :math:`Z`, not by arithmetic — and *more blocks
stream more of it*, because antennas get re-read by every block they appear in:

.. list-table::
   :header-rows: 1

   * - ``max_blocks``
     - 1
     - 2
     - 3
     - 4
     - 6
     - 8
     - 12
   * - Area (arithmetic)
     - 33 176
     - 7 623
     - 3 527
     - 2 707
     - 1 920
     - 1 714
     - 1 542
   * - :math:`\sum_b (N^b_{\rm row} + N^b_{\rm col})` (traffic)
     - 423
     - 489
     - 491
     - 509
     - 600
     - 620
     - 898

(The full product streams 320 — each antenna exactly once.) The two columns
pull in opposite directions, and the measured optimum sits where they balance.
This also explains ``VectorDot``: it has the least arithmetic of all, but
streams :math:`Z` twice per baseline over 1 501 separate tiny GEMMs.

Isolating the pieces with a micro-benchmark at the ``max_blocks=4`` shapes
(complex64, :math:`K = 65\,536`, A2000) splits the cost almost exactly in half:

.. list-table::
   :header-rows: 1

   * - Operation
     - Time
   * - Full product, ``cherk`` (what ``MatMul`` does)
     - 40.7 ms
   * - 4 blocks: gather of :math:`Z` rows/columns only
     - 6.4 ms
   * - 4 blocks: ``cgemm3m`` calls only
     - 6.5 ms
   * - 4 blocks: gather + GEMM (what ``MatBlock`` does)
     - 12.5 ms

Two things follow. First, even with the gather removed entirely, the four GEMMs
would only be 6.3x faster than the full ``cherk``, not 37.8x — the arithmetic
saving is genuinely not collectible at these shapes. Second, **half of
``MatBlock``'s time is the gather**, not the matrix product: the blocks need
contiguous operands, so the relevant rows and columns of :math:`Z` are copied
before each GEMM. A future optimization could avoid much of this (for example
by having the :math:`Z` construction write antennas in block order, so the
larger blocks become slices rather than fancy-index copies).

A third, smaller effect: ``MatMul`` uses the Hermitian rank-k routine
``cherk``, which halves the work of a general GEMM, while the rectangular
blocks cannot and use ``cgemm3m``. On this card that alone is worth ~1.3x in
``MatMul``'s favour (see `GEMM strategy: hardware dependence`_), and it is
architecture-dependent, so the crossover will differ on other GPUs.

When it's worth it
------------------

- **Only for redundant arrays.** The speedup is bounded above by
  :math:`N_{\rm ant}^2 / N_{\rm antpairs}`, and in practice lands far below
  that bound. If you are simulating per-antenna unique beams, or otherwise
  want every pair, use the default ``MatMul``.
- **Benchmark ``max_blocks``; don't minimize area.** 3-4 blocks was best here;
  the FLOP-minimizing choice (12+) was 1.4x *worse* than the best. The
  ``matvis hera-profile --matprod-method MatBlock --max-blocks N`` sweep used
  for the table above takes a few minutes and is the reliable way to pick.
- **Never use ``VectorDot`` on a GPU for this.** It has the lowest FLOP count
  of any option and is 4.5x slower than doing nothing special at all.
- The gain applies to the matrix-product stage only, so the end-to-end benefit
  is capped by that stage's share of the run (see `Where the time goes`_) —
  here 81 per cent of GPU time before the change, 59 per cent after it.

Control: what happens without redundancy
----------------------------------------

Same array, same sky, but requesting *all* 102 400 antenna pairs (the
non-redundant case — e.g. every antenna having a unique beam). The best
available decomposition is then a single block spanning the whole array, and
the result is a straightforward loss:

.. list-table::
   :header-rows: 1

   * - Method
     - Matrix product
     - Wall / integration
   * - ``MatMul``
     - 38.9 ms
     - 0.305 s
   * - ``MatBlock``, ``max_blocks=4``
     - 57.4 ms
     - 0.417 s (**1.37x slower**)

The slowdown is exactly the two costs identified above with none of the
benefit: the gather of :math:`Z`, and ``cgemm3m`` in place of ``cherk``.

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
   * - `PR #153 <https://github.com/HERA-Team/matvis/pull/153>`_ (Sept 2026)
     - Added the block-decomposed matrix product
       (``matprod_method="CPUMatBlock"/"GPUMatBlock"``) plus
       :mod:`matvis.redundancy` helpers for building the decomposition.
       Opt-in; the default ``MatMul`` path is unchanged.
     - 2.3x steady-state wall time (3.4x on the matrix product itself) on a
       320-antenna redundant hex layout with 1 501 unique baselines, RTX
       A2000. **No benefit — a 1.4x slowdown — on non-redundant arrays**; see
       `Block-decomposed products on redundant arrays`_.
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
