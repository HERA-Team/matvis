# Benchmarking and profiling matvis

See the "Performance" page of the documentation for scaling rules-of-thumb and
measured numbers. The tools here reproduce those measurements:

- **`run-canonical.sh [outdir] [dev|prodslice|both]`** — runs the two canonical
  benchmark configurations through `matvis profile` (polarized, gridded beams,
  one beam per antenna, single precision). Writes human-readable summaries and
  machine-readable `summary-stats-*.json` files (wall/setup/loop time,
  per-stage line-profiler and CUDA-event timings) for before/after comparison.
  Use `--gpu-event-timing` numbers (per-chunk CUDA events) rather than
  line-profiler attribution for GPU work: the loop is asynchronous, so
  host-side timings mostly show where the host happens to block.

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
