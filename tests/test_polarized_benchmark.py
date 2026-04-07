"""Runtime benchmarks comparing old (sqrt I) vs new (eigendecomp/sign-split) approaches.

Not run in CI — use: pytest tests/test_polarized_benchmark.py -v -s
"""

import pytest

import numpy as np
import time
from astropy import units as un
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pyuvdata.analytic_beam import GaussianBeam

from matvis import simulate_vis


def _make_bench_params(nsrc=1000, nant=50, precision=2):
    """Create simulation parameters for benchmarking."""
    rng = np.random.default_rng(42)

    ants = {i: rng.uniform(0, 200, 3) * np.array([1, 1, 0]) for i in range(nant)}
    telescope_loc = EarthLocation(
        lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0 * un.m
    )
    ra = rng.uniform(0, 2 * np.pi, nsrc)
    dec = rng.uniform(-0.6, -0.4, nsrc)
    freqs = np.array([100e6])
    times = Time([2459863.0], format="jd", scale="utc")
    beams = [GaussianBeam(diameter=14.0)]

    return {
        "ants": ants,
        "ra": ra,
        "dec": dec,
        "freqs": freqs,
        "times": times,
        "beams": beams,
        "telescope_loc": telescope_loc,
        "precision": precision,
    }


def _time_call(func, repeats=3, **kwargs):
    """Time a function call, return median time."""
    times_list = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(**kwargs)
        t1 = time.perf_counter()
        times_list.append(t1 - t0)
    return np.median(times_list), result


@pytest.mark.benchmark
class TestBenchmarks:
    """Runtime benchmarks for polarized sky."""

    def test_bench_unpolarized_old_vs_new(self):
        """Compare old sqrt(I) path vs eigendecomp with stokes=[I,0,0,0]."""
        params = _make_bench_params(nsrc=5000, nant=150)
        rng = np.random.default_rng(99)
        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = rng.uniform(0.5, 5.0, (nsrc, nfreq))

        # OLD PATH
        time_old, vis_old = _time_call(
            simulate_vis, fluxes=fluxes, polarized=True, **params
        )

        # NEW PATH with stokes=[I,0,0,0]
        stokes = np.zeros((4, nsrc, nfreq))
        stokes[0] = fluxes
        time_new, vis_new = _time_call(
            simulate_vis, fluxes=fluxes, polarized=True, stokes=stokes, **params
        )

        ratio = time_new / time_old
        print(f"\n  Old path:  {time_old:.4f}s")
        print(f"  New path:  {time_new:.4f}s")
        print(f"  Ratio:     {ratio:.2f}x")

        # Results should match
        np.testing.assert_allclose(vis_new, vis_old, atol=1e-10)

    def test_bench_polarized_eigendecomp(self):
        """Benchmark full polarized sky with eigendecomp."""
        params = _make_bench_params(nsrc=5000, nant=150)
        rng = np.random.default_rng(55)
        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = rng.uniform(1.0, 5.0, (nsrc, nfreq))

        stokes = np.zeros((4, nsrc, nfreq))
        stokes[0] = fluxes
        stokes[1] = 0.2 * fluxes
        stokes[2] = 0.1 * fluxes
        stokes[3] = 0.05 * fluxes

        time_pol, _ = _time_call(
            simulate_vis, fluxes=fluxes, polarized=True, stokes=stokes, **params
        )
        print(f"\n  Polarized eigendecomp: {time_pol:.4f}s")

    def test_bench_sign_split_overhead(self):
        """Measure sign-split overhead vs eigendecomp."""
        params = _make_bench_params(nsrc=5000, nant=150)
        rng = np.random.default_rng(33)
        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = rng.uniform(1.0, 5.0, (nsrc, nfreq))

        # All positive
        stokes_pos = np.zeros((4, nsrc, nfreq))
        stokes_pos[0] = fluxes

        time_eigen, _ = _time_call(
            simulate_vis, fluxes=fluxes, polarized=True, stokes=stokes_pos, **params
        )
        time_split_pos, _ = _time_call(
            simulate_vis,
            fluxes=fluxes,
            polarized=True,
            stokes=stokes_pos,
            negative_flux="split",
            **params,
        )

        # 50% negative
        stokes_neg = np.zeros((4, nsrc, nfreq))
        signs = np.ones(nsrc)
        signs[: nsrc // 2] = -1
        stokes_neg[0] = fluxes * signs[:, np.newaxis]
        time_split_neg, _ = _time_call(
            simulate_vis,
            fluxes=fluxes,
            polarized=True,
            stokes=stokes_neg,
            negative_flux="split",
            **params,
        )

        print(f"\n  Eigendecomp (all pos):       {time_eigen:.4f}s")
        print(
            f"  Sign-split (all pos):        {time_split_pos:.4f}s  ({time_split_pos / time_eigen:.2f}x)"
        )
        print(
            f"  Sign-split (50% neg):        {time_split_neg:.4f}s  ({time_split_neg / time_eigen:.2f}x)"
        )

    def test_bench_scaling_with_sources(self):
        """Measure scaling with number of sources."""
        print("\n  === Scaling with sources (150 antennas) ===")
        print("  Nsrc    | Old path  | Eigendecomp | Sign-split (50% neg)")
        print("  --------|-----------|-------------|---------------------")

        for nsrc in [100, 500, 1000, 5000, 10000]:
            params = _make_bench_params(nsrc=nsrc, nant=150)
            rng = np.random.default_rng(42)
            nfreq = len(params["freqs"])
            fluxes = rng.uniform(0.5, 5.0, (nsrc, nfreq))

            # Old path
            time_old, _ = _time_call(
                simulate_vis, fluxes=fluxes, polarized=True, repeats=2, **params
            )

            # Eigendecomp
            stokes = np.zeros((4, nsrc, nfreq))
            stokes[0] = fluxes
            time_eigen, _ = _time_call(
                simulate_vis,
                fluxes=fluxes,
                polarized=True,
                stokes=stokes,
                repeats=2,
                **params,
            )

            # Sign-split 50% neg
            signs = np.ones(nsrc)
            signs[: nsrc // 2] = -1
            stokes_neg = np.zeros((4, nsrc, nfreq))
            stokes_neg[0] = fluxes * signs[:, np.newaxis]
            time_split, _ = _time_call(
                simulate_vis,
                fluxes=fluxes,
                polarized=True,
                stokes=stokes_neg,
                negative_flux="split",
                repeats=2,
                **params,
            )

            print(
                f"  {nsrc:>7} | {time_old:.4f}s   | {time_eigen:.4f}s     | {time_split:.4f}s"
            )

    def test_bench_polarized_vs_unpolarized(self):
        """Compare unpolarized sky vs fully polarized sky (Q,U,V ≠ 0).

        Shows the cost of having a polarized sky model vs the old scalar-I approach.
        """
        params = _make_bench_params(nsrc=5000, nant=150)
        rng = np.random.default_rng(88)
        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = rng.uniform(1.0, 5.0, (nsrc, nfreq))

        # Unpolarized (old path)
        time_unpol, _ = _time_call(
            simulate_vis, fluxes=fluxes, polarized=True, **params
        )

        # Unpolarized via eigendecomp (stokes=[I,0,0,0])
        stokes_unpol = np.zeros((4, nsrc, nfreq))
        stokes_unpol[0] = fluxes
        time_eigen_unpol, _ = _time_call(
            simulate_vis, fluxes=fluxes, polarized=True, stokes=stokes_unpol, **params
        )

        # Weakly polarized (Q=0.1I, U=0.05I, V=0.02I)
        stokes_weak = np.zeros((4, nsrc, nfreq))
        stokes_weak[0] = fluxes
        stokes_weak[1] = 0.1 * fluxes
        stokes_weak[2] = 0.05 * fluxes
        stokes_weak[3] = 0.02 * fluxes
        time_weak_pol, _ = _time_call(
            simulate_vis, fluxes=fluxes, polarized=True, stokes=stokes_weak, **params
        )

        # Strongly polarized (Q=0.5I, U=0.4I, V=0.3I)
        stokes_strong = np.zeros((4, nsrc, nfreq))
        stokes_strong[0] = fluxes
        stokes_strong[1] = 0.5 * fluxes
        stokes_strong[2] = 0.4 * fluxes
        stokes_strong[3] = 0.3 * fluxes
        time_strong_pol, _ = _time_call(
            simulate_vis, fluxes=fluxes, polarized=True, stokes=stokes_strong, **params
        )

        print("\n  === Polarized vs Unpolarized (5000 src, 150 ant) ===")
        print(f"  Unpolarized (old sqrt I):     {time_unpol:.4f}s  (baseline)")
        print(
            f"  Unpolarized (eigendecomp):    {time_eigen_unpol:.4f}s  ({time_eigen_unpol / time_unpol:.2f}x)"
        )
        print(
            f"  Weakly polarized (Q,U,V≠0):  {time_weak_pol:.4f}s  ({time_weak_pol / time_unpol:.2f}x)"
        )
        print(
            f"  Strongly polarized:           {time_strong_pol:.4f}s  ({time_strong_pol / time_unpol:.2f}x)"
        )

    def test_bench_negative_flux_scenarios(self):
        """Compare runtime for different negative flux fractions.

        Shows sign-split overhead as a function of how many sources have negative I.
        """
        params = _make_bench_params(nsrc=5000, nant=150)
        rng = np.random.default_rng(66)
        nsrc = len(params["ra"])
        nfreq = len(params["freqs"])
        fluxes = rng.uniform(1.0, 5.0, (nsrc, nfreq))

        # Baseline: eigendecomp all positive
        stokes_pos = np.zeros((4, nsrc, nfreq))
        stokes_pos[0] = fluxes
        time_baseline, _ = _time_call(
            simulate_vis, fluxes=fluxes, polarized=True, stokes=stokes_pos, **params
        )

        print("\n  === Negative Flux Scenarios (5000 src, 150 ant) ===")
        print(f"  Eigendecomp (all positive):   {time_baseline:.4f}s  (baseline)")

        for neg_frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            n_neg = int(nsrc * neg_frac)
            signs = np.ones(nsrc)
            signs[:n_neg] = -1
            rng.shuffle(signs)

            stokes_mix = np.zeros((4, nsrc, nfreq))
            stokes_mix[0] = fluxes * signs[:, np.newaxis]

            time_split, _ = _time_call(
                simulate_vis,
                fluxes=fluxes,
                polarized=True,
                stokes=stokes_mix,
                negative_flux="split",
                **params,
            )
            print(
                f"  Sign-split ({neg_frac * 100:>5.1f}% neg):     {time_split:.4f}s  ({time_split / time_baseline:.2f}x)"
            )
