# Benchmarking and profiling matvis

See the "Performance" page of the documentation for scaling rules-of-thumb and
measured numbers. The tools here reproduce those measurements:

- **`run-canonical.sh [outdir] [dev|prodslice|both]`** — runs the two canonical
  benchmark configurations through `matvis profile` (polarized, gridded beams,
  one beam per antenna, single precision). Writes human-readable summaries and
  machine-readable `summary-stats-*.json` files for before/after comparison.

  The harness is warmup- and noise-robust by default: an untimed miniature
  simulation runs first (`--no-warmup` to disable) so one-time costs (cupy
  kernel compilation, cuBLAS workspace allocation, ERFA/IERS caches) don't
  contaminate the timings, per-integration wall times are recorded
  individually, and CUDA-event stage timings report medians as well as means.
  The numbers to quote are the JSON's `derived` block:

  - `steady_wall_per_integration` — median wall time per integration,
    excluding the first (the throughput you'll actually get);
  - `gpu_time_per_integration` — median per-chunk CUDA-event total × chunk
    count (measures the card; transfers between machines with the same GPU);
  - `host_overhead_per_integration` — the difference (measures the host).

  Don't use the line-profiler `stages` table for GPU work: the loop is
  asynchronous, so host-side timings mostly show where the host happens to
  block. For quieter numbers on shared nodes, consider locking GPU clocks
  (`nvidia-smi -lgc <clock>`) if you have permission.

- **`roofline.py`** — "speed of light" micro-benchmarks: the bare cuBLAS Gram
  product at the exact Z-matrix shape, the beam interpolation, and the
  Z-construction, at any problem size. If a matvis stage is much slower than
  its roofline here, the orchestration is the problem, not the math.

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

Note that with the fully-asynchronous pipeline, host-side NVTX ranges show
where the host *waits* (currently the `cp.where` in `select_chunk` acts as the
back-pressure point); use the CUDA kernel summary for true device cost.

Outputs under `profiling/results/` are git-ignored.
