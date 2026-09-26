"""Tests for the profiling stats the backends expose.

These exist because the harness reads them, and because the per-frequency
accounting they provide is only correct if every backend call appends exactly
one entry (see issue #134: ``LAST_RUN_STATS`` alone silently reported only the
last frequency of a multi-frequency run).
"""

import numpy as np
import pytest

from matvis import HAVE_GPU, simulate_vis
from matvis._test_utils import get_standard_sim_params
from matvis.cli import SETUP_FREQ_DEPENDENT, SETUP_FREQ_INDEPENDENT, classify_setup
from matvis.cpu import cpu as cpu_module

BACKENDS = [(False, cpu_module)]
if HAVE_GPU:
    from matvis.gpu import gpu as gpu_module

    BACKENDS.append(pytest.param((True, gpu_module), marks=pytest.mark.gpu))


@pytest.fixture
def backends(request):
    """Parameterized (use_gpu, module) pair."""
    return request.param


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_stats_entry_per_frequency(backend):
    """ALL_RUN_STATS must have one entry per channel, in frequency order."""
    use_gpu, module = backend
    nfreq = 3
    kw, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=True, nfreq=nfreq, ntime=2, nsource=20
    )

    module.reset_run_stats()
    assert module.ALL_RUN_STATS == []

    simulate_vis(**kw, use_gpu=use_gpu)

    assert len(module.ALL_RUN_STATS) == nfreq
    assert [st["freq"] for st in module.ALL_RUN_STATS] == list(kw["freqs"])
    # LAST_RUN_STATS is the final entry, not a separate accounting.
    assert module.LAST_RUN_STATS["freq"] == kw["freqs"][-1]

    # A second run must not accumulate on top of the first.
    module.reset_run_stats()
    simulate_vis(**kw, use_gpu=use_gpu)
    assert len(module.ALL_RUN_STATS) == nfreq


@pytest.mark.parametrize("backend", BACKENDS)
def test_setup_breakdown_accounts_for_setup(backend):
    """The named setup phases must sum to roughly the reported setup time."""
    use_gpu, module = backend
    kw, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=True, nfreq=1, ntime=2, nsource=20
    )

    module.reset_run_stats()
    simulate_vis(**kw, use_gpu=use_gpu)
    stats = module.LAST_RUN_STATS
    breakdown = stats["setup_breakdown"]

    # beam_wrangle_* are sub-phases of beam_construct, so they are excluded
    # from the sum to avoid double-counting.
    phases = {k: v for k, v in breakdown.items() if not k.startswith("beam_wrangle_")}
    assert sum(phases.values()) == pytest.approx(stats["setup_time"], rel=0.25)

    # The two wrangle halves together are all of beam_construct.
    assert (
        breakdown["beam_wrangle_freq_independent"]
        + breakdown["beam_wrangle_freq_dependent"]
        <= breakdown["beam_construct"] + 1e-6
    )

    split = classify_setup(breakdown)
    assert split["freq_independent"] > 0
    assert split["freq_dependent"] >= 0
    # Every phase must be classified one way or the other, or accounted for as
    # the beam_construct remainder -- otherwise the harness silently drops it.
    known = set(SETUP_FREQ_INDEPENDENT) | set(SETUP_FREQ_DEPENDENT)
    assert set(breakdown) - known == {"beam_construct"}


def test_cpu_stage_totals_cover_the_loop():
    """The CPU host-timed stages must account for most of the loop time."""
    kw, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=True, nfreq=1, ntime=3, nsource=50
    )
    cpu_module.reset_run_stats()
    simulate_vis(**kw, use_gpu=False)
    stats = cpu_module.LAST_RUN_STATS

    total = sum(stats["stage_totals"].values())
    assert 0 < total <= stats["loop_time"]
    # The stages are the loop; anything left over is logging and bookkeeping.
    assert total > 0.5 * stats["loop_time"]

    assert set(stats["stage_totals"]) == {
        "rotate",
        "select_chunk",
        "beam",
        "tau",
        "z",
        "matprod",
        "sum_chunks",
    }
    # rotate happens once per integration; the rest once per chunk per
    # integration.
    assert len(stats["integration_times"]) == 3
    assert np.all(np.array(stats["integration_times"]) > 0)
