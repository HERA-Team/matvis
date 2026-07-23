"""Speed-of-light micro-benchmarks for the matvis GPU hot path.

Times the individual device operations that dominate a production-scale matvis
run, independent of the orchestration code:

1. The cuBLAS complex GEMM at the exact Z-matrix shape (the theoretical best
   for the ``Compute V`` stage — if the matprod stage is much slower than
   this, the wrapper is wasting time; if this itself is far below the GPU's
   peak, the GEMM call/shape needs work).
2. The beam interpolation at nbeam=nant (production has one beam per antenna).
3. The Z-matrix construction.

Run with e.g.::

    uv run python profiling/roofline.py --nant 350 --nsrc 50000

Sizes default to a production-slice chunk that fits a 4 GB GPU.
"""

import argparse
import itertools
import time

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage

from matvis.gpu._cublas import zdotz


def timeit(fn, n=10):
    """Average wall time of fn() over n runs after a warmup call."""
    fn()
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - t0) / n


def main():
    """Run the micro-benchmarks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--nant", type=int, default=350)
    ap.add_argument("--nsrc", type=int, default=50_000)
    ap.add_argument("--nbeam", type=int, default=0, help="default: nant")
    ap.add_argument("--nfeed", type=int, default=2)
    ap.add_argument("--nax", type=int, default=2)
    ap.add_argument("--naz", type=int, default=360)
    ap.add_argument("--nza", type=int, default=180)
    ap.add_argument("--double", action="store_true")
    args = ap.parse_args()

    nant, nsrc, nfeed, nax = args.nant, args.nsrc, args.nfeed, args.nax
    nbeam = args.nbeam or nant
    nza, naz = args.nza + 1, args.naz + 1
    rtype = np.float64 if args.double else np.float32
    ctype = np.complex128 if args.double else np.complex64

    rng = cp.random.default_rng(0)

    def crandom(shape):
        r = rng.standard_normal(shape, dtype=rtype)
        i = rng.standard_normal(shape, dtype=rtype)
        return (r + 1j * i).astype(ctype)

    dev = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {dev['name'].decode()}  dtype: {np.dtype(ctype).name}")
    print(f"nant={nant} nbeam={nbeam} nfeed={nfeed} nax={nax} nsrc={nsrc}")
    print()

    # --- 1. GEMM speed of light at the Z shape ------------------------------
    M = nant * nfeed
    K = nax * nsrc
    z = crandom((M, K))
    out = cp.empty((M, M), dtype=ctype, order="F")
    t = timeit(lambda: zdotz(z, out=out))
    flops = 8 * M * M * K
    print(
        f"zdotz cuBLAS gemm (M={M}, K={K}): {t * 1e3:8.2f} ms  {flops / t / 1e12:6.2f} TFLOPS"
    )

    # --- 2. Beam interpolation (current map_coordinates storm) --------------
    beam = crandom((nbeam, nax, nfeed, nza, naz))
    az = rng.random(nsrc, dtype=rtype) * rtype(2 * np.pi)
    za = rng.random(nsrc, dtype=rtype) * rtype(np.pi / 2)
    daz, dza = 2 * np.pi / (naz - 1), np.pi / 2 / (nza - 1)
    out_beam = cp.zeros((nbeam, nfeed, nax, nsrc), dtype=ctype)

    def storm():
        for bm in range(nbeam):
            coords = cp.asarray([za / dza, az / daz])
            for fd, ax in itertools.product(range(nfeed), range(nax)):
                ndimage.map_coordinates(
                    beam[bm, ax, fd], coords, order=1, output=out_beam[bm, fd, ax]
                )

    t = timeit(storm, n=3)
    print(f"beam interp map_coordinates x{nbeam * nfeed * nax}:  {t * 1e3:8.2f} ms")

    # --- 3. Z construction (current implementation) -------------------------
    exptau = crandom((nant, nsrc))
    sqrt_flux = rng.standard_normal(nsrc, dtype=rtype)
    zbuf = cp.zeros((nant, nfeed, nax, nsrc), dtype=ctype)
    beam_idx = np.arange(nant) % nbeam

    def getz():
        e = exptau * sqrt_flux
        for fd in range(nfeed):
            for ax in range(nax):
                zbuf[:, fd, ax, :] = e
        for ant, bmidx in enumerate(beam_idx):
            zbuf[ant] *= out_beam[bmidx]

    t = timeit(getz, n=3)
    print(f"getz (broadcast + per-antenna loop):    {t * 1e3:8.2f} ms")


if __name__ == "__main__":
    main()
