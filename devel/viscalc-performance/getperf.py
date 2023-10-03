"""Script to get the performance metric of a particular solver."""
from __future__ import annotations

import click
from dataclasses import dataclass, asdict
import numpy as np
import time
from pathlib import Path
import yaml
import importlib
from methods._lib import Solver, RedundantSolver

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
    
    if t1 > 2: # assume overhead from compilation etc. is negligible compared to 5sec
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
                times.append((time.time() - t0)/n)

    return TimeResult(times, n)

def test_solver(solver, nants, nsrcs, ctype=complex):
    z0 = getz((nants, nsrcs), ctype)
    solver.test(z0, np.dot(z0, z0.T.conj()))

def get_timings(
    solver, nants, nsides, nsrcs, 
    repeats: int=3, rerun: bool = False, cache=Path('.'),
    ctype=complex,
    nants_redundant: int | None = None,
    pairs: dict | None = None,
) -> dict[tuple[int, int], float]:
    out = {}
    
    # First, test the solver.
    test_solver(solver, nants[0], nsrcs[0], ctype)
    
    # Get the outer iterator. 
    if nants_redundant:
        outer = pairs
    else:
        outer = nants

    for outer_thing in outer:
        if nants_redundant:
            label_outer = outer_thing[0]  # pairfrac
            use_nants = nants_redundant
        else:
            label_outer = outer_thing     # nant
            use_nants = outer_thing

        for nside, nsrc in zip(nsides, nsrcs):
            size = (use_nants, nsrc)
            z = getz(size, ctype)

            print(size, end=': ')
            prec = 'double' if ctype is complex else 'single'
            pth = cache / f"{solver.__name__}_{label_outer}x{nside}_{prec}.yaml"
            if not rerun and pth.exists():
                with open(pth, 'r') as fl:
                    o = out[(label_outer, nside)] = TimeResult(**yaml.safe_load(fl))
            else:
                if issubclass(solver, RedundantSolver):
                    sln = solver(z, outer_thing[1])
                else:
                    sln = solver(z)
                
                o = out[(label_outer, nside)] = get_timing(sln, repeats=repeats)
                del sln  # Ensure memory is freed.
                
            print(f"{o.mean:1.3e}s ± {o.std:1.3e}s [{o.repeats} loops of {o.n}]")
            
            # Cache it
            with open(pth, 'w') as fl:
                yaml.dump(asdict(o), fl)
            
    return out

def getz(shape, ctype):
    return (
        np.random.random(shape) + 
        np.random.random(shape)*1j
    ).astype(ctype)

def get_sizes(max_nants: int, max_nside: int, n_nants: int, n_nsides: int,):
    # Note that "nants" here represents Nants * Nfeed, which is why we 
    # go to double the number of ants that HERA has.
    nants = sorted([max_nants*2 // 2**i for i in range(n_nants)])
    nsides = sorted(max_nside // 2**i for i in range(n_nsides))
    nsrcs = [2 * 12 * nside**2 for nside in nsides]

    return nants, nsides, nsrcs

@click.command
@click.argument('solver', type=str, required=True)
@click.option('--max-nants', type=int, default=350)
@click.option('--n-nants', type=int, default=4)
@click.option('--max-nside', type=int, default=256)
@click.option('--n-nsides', type=int, default=4)
@click.option("--redundant-nants", type=int, default=None)
@click.option("--double/--single", default=True)
@click.option("--repeats", type=int, default=3)
@click.option("--rerun/--use-cache", default=False)
@click.option("--cache", type=click.Path(exists=True, file_okay=False), default=Path('.'))
def main(solver, max_nants: int, n_nants: int, max_nside: int, n_nsides: int, redundant_nants: int | None, double: bool, repeats: int, rerun: bool, cache):
    """Get the performance metric of a particular solver."""

    nants, nsides, nsrcs = get_sizes(max_nants=max_nants, max_nside=max_nside, n_nants=n_nants, n_nsides=n_nsides)

    # Note that "nants" here represents Nants * Nfeed, which is why we 
    # go to double the number of ants that HERA has.
    nants = sorted([max_nants*2 // 2**i for i in range(n_nants)])
    nsides = sorted(max_nside // 2**i for i in range(n_nsides))
    nsrcs = [2 * 12 * nside**2 for nside in nsides]

    if redundant_nants is None:
        redundant_nants = nants[-1] // 2

    mdl = importlib.import_module(solver, package='methods')
    
    for k, v in mdl.__dict__:
        if issubclass(v, (Solver, RedundantSolver)) and v is not Solver and v is not RedundantSolver and not k.startswith("_"):
            solver = v
            break
    else:
        raise ValueError(f"Cannot find a solver in '{solver}'")
    
    if issubclass(solver, RedundantSolver):
        allpairs = np.array([(0,0)] + [(a, b) for a in range(redundant_nants*2) for b in range(a+1, redundant_nants*2)])

        pairs = {}
        pairfracs = [3, 10, 25, 50, 100]

        for pc in pairfracs:
            pairs[pc] = allpairs[np.sort(np.random.choice(np.arange(len(allpairs)), size=int(len(allpairs)*pc/100), replace=False))]
    else:
        pairs = None

    get_timings(
        solver, nants=nants, nsides=nsides, nsrcs=nsrcs, 
        repeats=repeats, rerun=rerun, cache=cache, ctype=complex if double else np.complex64,
        nants_redundant=redundant_nants*2 if issubclass(v, RedundantSolver) else None,
        pairs=pairs
    )
    

if __name__ == '__main__':
    main()