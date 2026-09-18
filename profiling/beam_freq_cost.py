#!/usr/bin/env python
"""What `matvis` pays to put a beam on the frequency it is simulating.

``simulate_vis`` calls the backend once per frequency channel, and every one of
those calls runs ``_wrangle_beams``, which does
``UVBeam.interp(freq_array=[freq], new_object=True)`` for each unique beam. So
the cost is ``nbeam x nfreq`` interpolations of a beam that may cover the whole
observing band.

Two things about that are worth measuring, and this script measures both.

``batching``
    One ``interp(freq_array=freqs)`` against ``nfreq`` separate calls. The cost
    of an interpolation is dominated by processing the *source* beam, not by
    how many output channels are asked for, so the batched call is close to
    flat in ``nfreq`` -- i.e. a backend that saw every channel at once could
    collapse this to a single call. See issue #134.

``redundant``
    What the same call costs when the beam is *already* on the requested
    channel, which is what HERA passes in production (beams are interpolated
    to the job's channel before ``matvis`` sees them). ``new_object=True``
    rebuilds the ``UVBeam`` regardless, so this is pure overhead, paid per
    channel per beam.

Usage::

    uv run python profiling/beam_freq_cost.py batching
    uv run python profiling/beam_freq_cost.py redundant
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
from pyuvdata.analytic_beam import GaussianBeam

warnings.filterwarnings("ignore")

FREQ = 150e6


def timeit(fn, n: int = 3) -> float:
    """Median of ``n`` runs, after one untimed warmup."""
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def make_beam(nfreq: int, naz: int, nza: int, fmin=100e6, fmax=200e6):
    """A gridded efield beam with ``nfreq`` channels."""
    return GaussianBeam(diameter=14.0).to_uvbeam(
        freq_array=np.linspace(fmin, fmax, nfreq) if nfreq > 1 else np.array([FREQ]),
        axis1_array=np.linspace(0, 2 * np.pi, naz + 1)[:-1],
        axis2_array=np.linspace(0, np.pi, nza),
    )


def cmd_batching(a):
    """One interp for many channels, versus one interp per channel."""
    beam = make_beam(a.beam_nfreq, a.naz, a.nza)
    print(
        f"source beam: {beam.data_array.shape} {beam.data_array.dtype} "
        f"({beam.data_array.nbytes / 1e6:.0f} MB over {a.beam_nfreq} channels)\n"
    )
    print(
        f"{'nfreq':>6} {'one-at-a-time s':>16} {'batched s':>11} {'speedup':>8} "
        f"{'per channel ms':>15}"
    )
    for nfreq in a.nfreqs:
        # Offset off the beam's own nodes so the interpolation is real work.
        want = np.linspace(105e6, 195e6, nfreq) + 1e3

        t_sep = timeit(
            lambda w=want: [
                beam.interp(freq_array=np.array([f]), new_object=True, run_check=False)
                for f in w
            ]
        )
        t_bat = timeit(
            lambda w=want: beam.interp(freq_array=w, new_object=True, run_check=False)
        )
        print(
            f"{nfreq:>6} {t_sep:16.4f} {t_bat:11.4f} {t_sep / t_bat:7.2f}x "
            f"{t_sep / nfreq * 1e3:15.2f}"
        )

    print(
        "\nEach row is ONE beam; multiply the left column by the number of unique\n"
        "beams for a run's per-channel cost. The batched column is roughly flat,\n"
        "so the saving grows with nfreq."
    )


def cmd_redundant(a):
    """The production case: the beam is already on the requested channel."""
    from matvis.core.beams import _wrangle_beams

    beam = make_beam(1, a.naz, a.nza)
    print(
        f"single-channel beam at {FREQ / 1e6:.0f} MHz: {beam.data_array.shape} "
        f"({beam.data_array.nbytes / 1e6:.1f} MB)\n"
    )
    print(
        f"{'nbeams':>7} {'_wrangle_beams s':>17} {'per beam ms':>12} "
        f"{'of which interp s':>18} {'cost to detect s':>17}"
    )
    for nb in a.nbeams:
        beams = [beam] * nb
        t_wrangle = timeit(lambda b=beams, n=nb: _wrangle_beams(None, b, True, n, FREQ))
        t_interp = timeit(
            lambda b=beams: [
                x.interp(freq_array=np.array([FREQ]), new_object=True, run_check=False)
                for x in b
            ]
        )
        t_check = timeit(lambda b=beams: [np.allclose(x.freq_array, [FREQ]) for x in b])
        print(
            f"{nb:>7} {t_wrangle:17.4f} {t_wrangle / nb * 1e3:12.2f} "
            f"{t_interp:18.4f} {t_check:17.6f}"
        )

    print(
        "\nThe interpolation has nothing to do -- the beam is already on that\n"
        "channel -- but new_object=True rebuilds the UVBeam anyway. The last\n"
        "column is what it costs to notice."
    )


def main():
    """Parse arguments and dispatch to the chosen sub-command."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--naz", type=int, default=360)
    common.add_argument("--nza", type=int, default=181)

    b = sub.add_parser("batching", parents=[common], help=cmd_batching.__doc__)
    b.add_argument("--beam-nfreq", type=int, default=32)
    b.add_argument("--nfreqs", type=int, nargs="+", default=[1, 2, 4, 8])
    b.set_defaults(func=cmd_batching)

    r = sub.add_parser("redundant", parents=[common], help=cmd_redundant.__doc__)
    r.add_argument("--nbeams", type=int, nargs="+", default=[1, 50, 350])
    r.set_defaults(func=cmd_redundant)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
