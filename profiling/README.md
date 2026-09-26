# Benchmarking and profiling matvis

See the [Performance page](../docs/performance.rst) of the documentation for
scaling rules-of-thumb and measured numbers. The tools here reproduce those
measurements:

- **`run-canonical.sh [outdir] [dev|prodslice|both] [spline-order]`** — runs two canonical
  benchmark configurations through `matvis profile`: `dev` (64 antennas/beams,
  200k sources — small enough to iterate quickly) and `prodslice` (350
  antennas/beams, 1M sources — a 'production-scale' run).
  Both use the following settings: polarized, gridded beams, one beam per
  antenna, single precision, and linear beam interpolation unless a third
  argument selects another spline order (3 = bicubic; see the docs Beam
  Interpolation page). Writes human-readable summaries and
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
    (excluding the first). This is the time spanned by the chunk pipeline
    *on the stream*, which is an upper bound on device compute: CUDA events
    bracket a region of the stream, so device idle *inside* a chunk (waiting
    for the host to enqueue work) is counted here too. Use `gpu_idle.py`
    below to separate the two.
  - `host_overhead_per_integration` — `steady_wall_per_integration` minus
    `gpu_time_per_integration`: host work that happens *outside* the chunk
    loop (coordinate rotation, chunk summing, Python dispatch). Host stalls
    *inside* the loop do not appear here — they are hidden in
    `gpu_time_per_integration`; `gpu_idle.py` is what surfaces those.
  - `sum_chunks_per_integration` — the once-per-integration visibility
    readout, timed after an explicit stream drain so it measures its own
    cost rather than the queued pipeline it would otherwise block on.

  Also check `nchunks_used`. `--nchunks` is only a *minimum*: if device
  memory is tight the run can use many more chunks, which changes the
  per-chunk problem size and makes stage timings incomparable between runs.
  The profiler warns when this happens.

  Don't use the line-profiler `stages` table for GPU work: the loop is
  asynchronous, so host-side timings mostly show where the host happens to
  block. `Sum Chunks` is the worst offender — it read ~73 ms per integration
  there against a true cost of 13.9 ms, the difference being time waiting on
  queued chunk work plus a first-integration outlier skewing a 4-sample
  mean. For quieter numbers on shared nodes, consider locking GPU clocks
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
*waiting*, not work, so a range's duration is not its cost. Use the CUDA
kernel summary for true device cost, and `gpu_idle.py` (below) to find out
where the device is idle.

- **`gpu_idle.py`** — the complement to the CUDA-event timings: it takes the
  union of all kernel and memcpy intervals in an nsys trace as the device's
  *busy* time, subtracts that from each integration's wall span, and
  attributes the remaining idle to the NVTX range the host was inside.

  ```bash
  profiling/gpu_idle.py --run -- -a 350 -b 350 -s 1000000 -t 3 --nchunks 30
  profiling/gpu_idle.py profiling/results/mytrace.nsys-rep   # existing trace
  ```

  Idle is the headroom for anything that only makes the *host* faster
  (removing a synchronization, fewer launches, deeper queueing); it does not
  shrink on a faster GPU, so it is a *larger* fraction of the run there. To
  predict a card `f` times faster on this workload, scale `busy` by `1/f` and
  leave `idle` alone. This is how a host-side change can be judged honestly
  on a modest development GPU.

  Caveat: once the host successfully runs ahead of the device, the NVTX range
  the host occupies during a gap is no longer the range that *caused* it. The
  per-range breakdown is diagnostic when the host is the critical path (which
  is exactly when it matters); the total is always meaningful.

Outputs under `profiling/results/` are git-ignored.
