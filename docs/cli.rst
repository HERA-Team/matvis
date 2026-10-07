===
CLI
===

``matvis`` installs a ``matvis`` command-line entry point with profiling and
benchmarking subcommands (see also the :doc:`Performance <performance>` page
and ``profiling/README.md`` for how to use these in practice).
The python code for this CLI command lives in ``src/matvis/cli.py`` and is delegated
through the ``main()`` function there.

.. click:: matvis.cli:main
   :prog: matvis
   :nested: full

Annotated example output
=========================

``matvis profile``/``hera-profile`` write a machine-readable
``summary-stats-*.json`` to ``--outdir``. Below is a real example, from the
production-slice config on a Tesla V100 (the ``stages`` block, and most of
``run_stats``/``event_timing_ms``'s per-stage entries, are omitted here for
brevity — see below for what they contain):

.. code-block:: json

    {
      "config": {
        "nants": 350, "nbeams": 350, "nsource": 1000000,
        "gpu": true, "precision": 1, "nchunks": 30
      },
      "total_time": 8.63,
      "derived": {
        "steady_wall_per_integration": 0.806,
        "gpu_time_per_integration": 0.799,
        "host_overhead_per_integration": 0.007
      },
      "run_stats": {
        "integration_times": [1.230, 0.816, 0.795, 0.806],
        "steady_time_per_integration": 0.806,
        "steady_gpu_time_per_integration": 0.799,
        "event_timing_ms": {
          "chunk_total": {"median": 26.62, "mean": 28.31, "std": 7.4, "n": 120}
        }
      }
    }

- **``derived``** — the three numbers to quote and compare (see the Rules of
  Thumb table on the :doc:`Performance <performance>` page):
  ``steady_wall_per_integration`` (median wall time per integration,
  excluding the first), ``gpu_time_per_integration`` (copied from
  ``run_stats.steady_gpu_time_per_integration`` — device time only), and
  ``host_overhead_per_integration`` (their difference).
- **``run_stats``** — the full detail behind ``derived``: every
  per-integration wall time (``integration_times``), and, with
  ``--gpu-event-timing``, per-stage CUDA-event median/mean/std/sample-count
  under ``event_timing_ms`` (one entry per stage: ``chunk_total``, ``beam``,
  ``tau``, ``z``, ``matprod``) plus ``steady_gpu_time_per_integration`` —
  each integration's actual chunk totals summed, then the median taken
  across integrations excluding the first (the same warmup-robust
  treatment as ``steady_time_per_integration``).
- **``stages``** — line-profiler timings of named code regions. Useful for
  the CPU backend; for the GPU backend the loop is asynchronous, so treat it
  as a rough indicator only (see the warning on the Performance page).
- **``run_stats_per_freq``** — a list with one entry per frequency channel,
  each in the same shape as ``run_stats``. ``simulate_vis`` calls the backend
  once per channel, so at ``--nfreq > 1`` the bare ``run_stats`` dict is only
  the *last* channel; this list is the whole run.

Multi-frequency and setup accounting
====================================

``simulate_vis`` runs one backend call per frequency channel, so everything a
backend call does at setup — planning the source chunking, building the
coordinate rotator, wrangling and uploading beams, allocating buffers — is
paid ``nfreq`` times. ``--nfreq > 1`` therefore reports an extra block:

.. code-block:: text

    Setup, per frequency
      Total setup per frequency:              2.516e-02 seconds
        frequency-independent (hoistable):    1.312e-02 seconds
        frequency-dependent:                  9.455e-03 seconds
      Projected saving from hoisting setup:   9.67% of this run

The split comes from ``run_stats.setup_breakdown``, a host-timed breakdown of
each named setup phase. Phases are classified in ``matvis.cli``:

- *frequency-independent*: ``validate``, ``chunk_planning``,
  ``coord_construct``, ``coord_setup``, ``z_setup``, ``matprod_setup``,
  ``vis_alloc``, and the ``BeamInterface``-wrangling half of beam construction
  (``beam_wrangle_freq_independent``).
- *frequency-dependent*: ``tau_setup`` (antenna positions are pre-scaled by
  :math:`2\pi\nu/c`), ``beam_setup`` (this channel's beam grid is uploaded to
  the device) and ``beam_wrangle_freq_dependent``
  (``UVBeam.interp`` onto this channel).

``projected_setup_hoist_saving`` is
:math:`(N_{\rm freq} - 1) \, S_{\rm indep} / T_{\rm total}` — the ceiling for
restructuring so that the frequency-independent setup happens once
(`issue #134 <https://github.com/HERA-Team/matvis/issues/134>`_). It says
nothing about the in-loop stages, which dominate once ``--ntimes`` is large.

The CPU backend additionally reports host-timed per-stage costs
(``derived.stage_seconds_per_integration``). Unlike the GPU backend's CUDA
events these are unambiguous, because the CPU path is synchronous;
``rotate`` and ``select_chunk`` are the frequency-independent stages.

Repeats
=======

``--repeat N`` runs the whole simulation ``N`` times and reports the median of
every derived quantity plus ``repeat_wall_spread_frac``,
``(max - min) / median`` of the per-repeat steady wall time. Any claimed
speed-up smaller than that spread is not measurable at that repeat count.

Coordinate-rotation accuracy
============================

``--update-bcrs-every`` sets, in seconds, how often the ERFA rotators
recompute the light-deflection and aberration corrections to the source
vectors. These are ~90% of the cost of a coordinate rotation. The default of
``0`` recomputes them at every integration, which is exact; ~180 keeps
differences below ~10 mas for far less work. It is only accepted by the ERFA
coordinate methods, and passing it to ``CoordinateRotationAstropy`` is an
error rather than a silent no-op.
