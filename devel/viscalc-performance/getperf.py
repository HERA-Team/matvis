"""Script to get the performance metric of a particular solver."""
from __future__ import annotations

import click
import importlib
import numpy as np
import time
import yaml
from dataclasses import asdict, dataclass
from methods._lib import RedundantSolver, Solver
from pathlib import Path

@dataclass
class TimeResult:
    times: list[float]
    n: int

    @property
    def best(self):
        return np.min(self.times)

    @property
    def mean(self):
        return np.mean(self.times)

    @property
    def repeats(self):
        return len(self.times)

    @property
    def std(self):
        return np.std(self.times)


def get_timing(sln, repeats: int = 3) -> float:
    sln.setup()

    t0 = time.time()
    sln.compute()
    t1 = time.time() - t0

    times = []

    if t1 > 2:  # assume overhead from compilation etc. is negligible compared to 5sec
        times.append(t1)
        n = 1

    if repeats > len(times):
        # Need to do it at least twice to check if the time comes down.

        if t1 < 2:
            t0 = time.time()
            sln.compute()
            t2 = time.time() - t0

            n = int(2 / t2) + 1

            if t2 > 2:
                times.append(t2)

        if n == 1:
            for _ in range(max(0, repeats - len(times))):
                t0 = time.time()
                sln.compute()
                times.append(time.time() - t0)

        else:
            for _ in range(repeats):
                t0 = time.time()
                for _ in range(n):
                    sln.compute()
                times.append((time.time() - t0) / n)

    return TimeResult(times, n)


def test_solver(solver, nants, nsrcs, ctype=complex):
    z0 = getz((nants, nsrcs), ctype)
    solver.test(z0, np.dot(z0, z0.T.conj()))


def get_timings(
    solver,
    nants,
    nsides,
    nsrcs,
    repeats: int = 3,
    rerun: bool = False,
    cache=Path("."),
    ctype=complex,
    nants_redundant: int | None = None,
    pairs: dict | None = None,
    transpose: bool = False,
) -> dict[tuple[int, int], float]:
    out = {}
    cache = Path(cache)

    # First, test the solver.
    test_solver(solver, nants[0], nsrcs[0], ctype)

    # Get the outer iterator.
    outer = pairs.items() if solver.is_redundant else nants

    for outer_thing in outer:
        if solver.is_redundant:
            label_outer = outer_thing[0]  # pairfrac
            use_nants = nants_redundant
        else:
            label_outer = outer_thing  # nant
            use_nants = outer_thing

        for nside, nsrc in zip(nsides, nsrcs):
            size = (use_nants, nsrc)
            z = getz(size, ctype, transpose=transpose)

            print((label_outer, nside), end=": ")
            prec = "double" if ctype is complex else "single"
            trns = "col" if transpose else "row"
            pth = cache / f"{solver.__name__}_{label_outer}x{nside}_{prec}_{trns}.yaml"
            if not rerun and pth.exists():
                with open(pth) as fl:
                    o = out[(label_outer, nside)] = TimeResult(**yaml.safe_load(fl))
            else:
                sln = solver(z, outer_thing[1]) if solver.is_redundant else solver(z)

                o = out[(label_outer, nside)] = get_timing(sln, repeats=repeats)
                del sln  # Ensure memory is freed.

            print(f"{o.mean:1.3e}s ± {o.std:1.3e}s [{o.repeats} loops of {o.n}]")

            # Cache it
            with open(pth, "w") as fl:
                yaml.dump(asdict(o), fl)

    return out


def getz(shape, ctype, transpose=False):
    if transpose:
        return (
            np.random.random(shape[::-1]) + np.random.random(shape[::-1]) * 1j
        ).astype(ctype)
    else:
        return (np.random.random(shape) + np.random.random(shape) * 1j).astype(ctype)


def get_sizes(
    max_nants: int,
    max_nside: int,
    n_nants: int,
    n_nsides: int,
):
    # Note that "nants" here represents Nants * Nfeed, which is why we
    # go to double the number of ants that HERA has.
    nants = sorted([max_nants * 2 // 2**i for i in range(n_nants)])
    nsides = sorted(max_nside // 2**i for i in range(n_nsides))
    nsrcs = [2 * 12 * nside**2 for nside in nsides]

    return nants, nsides, nsrcs


def get_solver(solver):
    mdl = importlib.import_module(f"methods.{solver}")

    for k, v in mdl.__dict__.items():
        if (
            np.issubclass_(v, (Solver, RedundantSolver))
            and v is not Solver
            and v is not RedundantSolver
            and not k.startswith("_")
        ):
            solver = v
            break
    else:
        raise ValueError(f"Cannot find a solver in '{solver}'")

    return solver


cli = click.Group()

@cli.command()
@click.argument("solver", type=str, required=True)
@click.option("--max-nants", type=int, default=350)
@click.option("--n-nants", type=int, default=4)
@click.option("--max-nside", type=int, default=256)
@click.option("--n-nsides", type=int, default=4)
@click.option("--double/--single", default=True)
@click.option("--repeats", type=int, default=3)
@click.option("--rerun/--use-cache", default=False)
@click.option("--transpose/--no-transpose", default=False)
@click.option(
    "--cache", type=click.Path(exists=True, file_okay=False), default=Path(".")
)
def main(
    solver,
    max_nants: int,
    n_nants: int,
    max_nside: int,
    n_nsides: int,
    double: bool,
    repeats: int,
    rerun: bool,
    cache,
    transpose: bool,
):
    """Get the performance metric of a particular solver."""

    nants, nsides, nsrcs = get_sizes(
        max_nants=max_nants, max_nside=max_nside, n_nants=n_nants, n_nsides=n_nsides
    )

    # Note that "nants" here represents Nants * Nfeed, which is why we
    # go to double the number of ants that HERA has.
    nants = sorted([max_nants * 2 // 2**i for i in range(n_nants)])
    nsides = sorted(max_nside // 2**i for i in range(n_nsides))
    nsrcs = [2 * 12 * nside**2 for nside in nsides]

    redundant_nants = nants[-1] // 2

    solver = get_solver(solver)

    if solver.is_redundant:
        allpairs = np.array(
            [(0, 0)]
            + [
                (a, b)
                for a in range(redundant_nants * 2)
                for b in range(a + 1, redundant_nants * 2)
            ]
        )

        pairs = {}
        pairfracs = [3, 10, 25, 50, 100]

        for pc in pairfracs:
            pairs[pc] = allpairs[
                np.sort(
                    np.random.choice(
                        np.arange(len(allpairs)),
                        size=int(len(allpairs) * pc / 100),
                        replace=False,
                    )
                )
            ]
    else:
        pairs = None

    get_timings(
        solver,
        nants=nants,
        nsides=nsides,
        nsrcs=nsrcs,
        repeats=repeats,
        rerun=rerun,
        cache=cache,
        ctype=complex if double else np.complex64,
        nants_redundant=redundant_nants * 2,
        pairs=pairs,
        transpose=transpose,
    )

def get_redundancies(bls, ndecimals: int=2):
    uvbins = set()
    pairs = []

    # Everything here is in wavelengths
    bls = np.round(bls, decimals=ndecimals)
    nant = bls.shape[0]

    # group redundant baselines
    for i in range(nant):
        for j in range(i + 1, nant):
            u, v = bls[i, j]
            if (u, v) not in uvbins and (-u, -v) not in uvbins:
                uvbins.add((u, v))
                pairs.append([i,j])

    return pairs

@cli.command()
@click.argument("solver", type=str, required=True)
@click.option("--double/--single", default=True)
@click.option("--transpose/--no-transpose", default=False)
@click.option("--outriggers/--core", default=False)
@click.option("--nside", type=int, default=256)
@click.option("--ax-moves-first/--ant-moves-first", default=True)
def hera_profile(solver, double, transpose, outriggers, nside, ax_moves_first):
    from py21cmsense.antpos import hera

    antpos = hera(hex_num=11, split_core=True, outriggers=2 if outriggers else 0)
    bls = antpos[np.newaxis, :, :2] - antpos[:, np.newaxis, :2]
    pairs = np.array(get_redundancies(bls.value))

    # We also need the pairs for the pol axis
    if ax_moves_first:
        pairs *= 2
        pairs = np.array([[p, p + 1] for p in pairs]).reshape((2*len(pairs), 2))
    else:
        pairs = np.concatenate((pairs, pairs + len(pairs)), axis=0)

    solver = get_solver(solver)

    ctype = complex if double else np.complex64
    test_solver(solver, 50, 1000, ctype)

    nant = len(antpos)
    nsrc = 12*nside**2

    # now run the actual computation
    z = getz((2*nant, 2*nsrc), transpose=transpose, ctype=ctype)

    if solver.is_redundant:
        sln = solver(z, pairs=pairs)
    else:
        sln = solver(z)

    sln()

if __name__ == "__main__":
    cli()
