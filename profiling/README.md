# Benchmarking and profiling matvis

See the [Performance page](../docs/performance.rst) of the documentation for
scaling rules-of-thumb and measured numbers. The tools here reproduce those
measurements:

- **`run-canonical.sh [outdir] [dev|prodslice|both]`** — runs two canonical
  benchmark configurations through `matvis profile`: `dev` (64 antennas/beams,
  200k sources — small enough to iterate quickly) and `prodslice` (350
  antennas/beams, 1M sources — a 'production-scale' run).
  Both use the following settings: polarized, gridded beams, one beam per
  antenna, single precision. Writes human-readable summaries and
  machine-readable `summary-stats-*.json` files for before/after comparison.
  See `docs/cli.rst` for the full `matvis profile` parameter reference and an
  annotated example of the JSON output.

  The script runs an untimed warmup simulation first (`--no-warmup` to
  disable), so one-time costs — cupy kernel compilation, cuBLAS workspace
  allocation, ERFA/IERS cache loads — don't contaminate the timings.
  Per-integration wall times are recorded individually, and CUDA-event stage
  timings report medians and standard deviations as well as means. The
  numbers to quote are the JSON's `derived` block:

  - `steady_wall_per_integration` — median wall time per integration,
    excluding the first (which is skewed even after warmup: outliers still
    occur, e.g. a lazily-allocated buffer touched for the first time in a
    later integration).
  - `gpu_time_per_integration` — each integration's actual per-chunk
    CUDA-event totals summed, then the median taken across integrations
    (excluding the first), i.e. device compute and transfer time only, no
    host-side (CPU) dispatch. Comparable across machines that have the same
    GPU, since it excludes the host's contribution.
  - `host_overhead_per_integration` — `steady_wall_per_integration` minus
    `gpu_time_per_integration`: everything that isn't GPU compute
    (coordinate rotation, Python dispatch, horizon-cut bookkeeping). Varies
    with the machine's CPU, not the GPU.

  Don't use the line-profiler `stages` table for GPU work: the loop is
  asynchronous, so host-side timings mostly show where the host happens to
  block. For quieter numbers on shared nodes, consider locking GPU clocks
  (`nvidia-smi -lgc <clock>`) if you have permission.

- **`roofline.py`** — GPU benchmarks for the operations that dominate a
  matvis run: the bare cuBLAS Gram product at the specified simulation size,
  the beam interpolation, and the Z-matrix construction. These isolate the
  GPU-bound floor for each stage, independent of matvis's own orchestration
  code. If a real matvis run is much slower than these numbers for the same
  stage, the gap is in matvis's scheduling/dispatch around that stage, not
  in the underlying GPU operation itself.

- **`gemm_experiments.py`** — compares cuBLAS strategies for V = Z Z^H
  (cgemm vs cgemm3m vs cherk) at a given shape.

## Block-decomposed matrix product (`MatBlock`)

`matvis profile`/`hera-profile` take `--matprod-method MatBlock` together with
`--max-blocks N`. The decomposition is built for you with
`matvis.redundancy.find_dense_blocks` over whatever antenna pairs the run
requests, and the resulting block count, total sub-matrix area and setup time
are printed and recorded under a `blocks` key in the summary JSON. Runs at
different `--max-blocks` are written to separate files (the label gets an
`_mbN` suffix), so a sweep doesn't overwrite itself.

`hera-profile` is the one to use here, since it builds a redundant HERA-like
hex array and deduplicates the baselines — `MatBlock` does nothing for a
non-redundant set of pairs. It needs `21cmSense` for the antenna layout, which
isn't a declared dependency; run it as `uv run --with 21cmSense matvis
hera-profile ...`.

The sweep behind the Performance page's "Block-decomposed products on
redundant arrays" table:

```bash
for mb in 1 2 3 4 6 8 12; do
    uv run --with 21cmSense matvis hera-profile -a 11 -s 128 -b 1 -t 6 -f 1 \
        --nchunks 6 --gpu --interpolated-beam --single-precision \
        --gpu-event-timing --coord-method CoordinateRotationERFA \
        --matprod-method MatBlock --max-blocks $mb -o profiling/results
done
```

Compare against `--matprod-method MatMul` and `--matprod-method VectorDot` at
the same settings. The number to watch is `run_stats.event_timing_ms.matprod`
(the stage the decomposition actually changes) alongside the usual
`derived.steady_wall_per_integration`. Note that the best `--max-blocks` is
*not* the one with the smallest area; see the Performance page for why.

## nsys

Stages are annotated with NVTX ranges (`rotate`, `select_chunk`, `beam`,
`tau`, `z`, `matprod`, `sum_chunks`):

```bash
nsys profile -t cuda,nvtx -o profiling/results/myrun \
    uv run matvis profile -a 350 -b 350 -s 1000000 -t 2 --nchunks 30 \
    --gpu --interpolated-beam --single-precision \
    --coord-method CoordinateRotationERFA -o profiling/results
nsys stats --report nvtx_sum --report cuda_gpu_kern_sum profiling/results/myrun.nsys-rep
```

With the fully-asynchronous pipeline, host-side NVTX ranges mostly measure
*waiting*, not work: the host spends most of its time waiting during
`select_chunk`'s `cp.where` call, and that wait dominates the reported NVTX
range rather than the call's own cost. Use the CUDA kernel summary for true
device cost.

Outputs under `profiling/results/` are git-ignored.
