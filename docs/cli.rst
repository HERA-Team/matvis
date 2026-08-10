===
CLI
===

``matvis`` installs a ``matvis`` command-line entry point with profiling and
benchmarking subcommands (see also the :doc:`Performance <performance>` page
and ``profiling/README.md`` for how to use these in practice).

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
        "event_timing_ms": {
          "chunk_total": {"median": 26.62, "mean": 28.31, "std": 7.4, "n": 120}
        }
      }
    }

- **``derived``** — the three numbers to quote and compare (see the Rules of
  Thumb table on the :doc:`Performance <performance>` page):
  ``steady_wall_per_integration`` (median wall time per integration,
  excluding the first), ``gpu_time_per_integration`` (median per-chunk
  CUDA-event total × chunk count — device time only), and
  ``host_overhead_per_integration`` (their difference).
- **``run_stats``** — the full detail behind ``derived``: every
  per-integration wall time (``integration_times``), and, with
  ``--gpu-event-timing``, per-stage CUDA-event median/mean/std/sample-count
  under ``event_timing_ms`` (one entry per stage: ``chunk_total``, ``beam``,
  ``tau``, ``z``, ``matprod``).
- **``stages``** — line-profiler timings of named code regions. Useful for
  the CPU backend; for the GPU backend the loop is asynchronous, so treat it
  as a rough indicator only (see the warning on the Performance page).
