"""Tests for putting beams on the simulated frequency.

`_wrangle_beams` runs once per frequency channel, and re-interpolating a beam
that is already on the requested channel is pure overhead -- the HERA
production case, where beams reach matvis pre-interpolated. These tests pin the
shortcuts that avoid it, and that the results are unchanged by them.
"""

import numpy as np
import pytest
from pyuvdata.analytic_beam import GaussianBeam
from pyuvdata.beam_interface import BeamInterface

from matvis.core.beams import (
    _already_at_freq,
    _interp_beams_to_freq,
    _wrangle_beams,
)

FREQ = 150e6


@pytest.fixture(scope="module")
def multifreq_beam():
    """A gridded beam covering a band, as read from a beam file."""
    return GaussianBeam(diameter=14.0).to_uvbeam(
        freq_array=np.linspace(100e6, 200e6, 5),
        axis1_array=np.linspace(0, 2 * np.pi, 37)[:-1],
        axis2_array=np.linspace(0, np.pi, 19),
    )


@pytest.fixture(scope="module")
def singlefreq_beam():
    """A beam already interpolated onto the channel being simulated."""
    return GaussianBeam(diameter=14.0).to_uvbeam(
        freq_array=np.array([FREQ]),
        axis1_array=np.linspace(0, 2 * np.pi, 37)[:-1],
        axis2_array=np.linspace(0, np.pi, 19),
    )


def test_already_at_freq(singlefreq_beam, multifreq_beam):
    """Only a single channel on exactly the requested frequency counts."""
    assert _already_at_freq(BeamInterface(singlefreq_beam), FREQ)
    # A band-covering beam has work to do even if FREQ is one of its channels.
    assert not _already_at_freq(BeamInterface(multifreq_beam), FREQ)
    # A single channel somewhere else is not a match.
    assert not _already_at_freq(BeamInterface(singlefreq_beam), 151e6)


def test_beam_already_at_freq_is_passed_through(singlefreq_beam):
    """No interpolation, and no new object, when there is nothing to do."""
    bi = BeamInterface(singlefreq_beam)
    (out,) = _interp_beams_to_freq([bi], FREQ)
    assert out is bi


def test_repeated_beam_objects_interpolate_once(multifreq_beam):
    """`[beam] * n` costs one interpolation, not n."""
    calls = 0
    original = type(multifreq_beam).interp

    def counting_interp(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    bl = [BeamInterface(multifreq_beam)] * 4
    try:
        type(multifreq_beam).interp = counting_interp
        out = _interp_beams_to_freq(bl, FREQ)
    finally:
        type(multifreq_beam).interp = original

    assert calls == 1
    assert len(out) == 4
    # All four entries share the single interpolated beam.
    assert all(o is out[0] for o in out)


def test_distinct_beams_each_interpolate(multifreq_beam):
    """Separate objects are separate work, even if their data matches."""
    bl = [BeamInterface(multifreq_beam.copy()) for _ in range(3)]
    out = _interp_beams_to_freq(bl, FREQ)
    assert len({id(o) for o in out}) == 3
    for o in out:
        assert o.beam.Nfreqs == 1
        assert np.isclose(np.atleast_1d(o.beam.freq_array)[0], FREQ)


def test_interpolated_result_matches_the_unshortcut_path(multifreq_beam):
    """The shortcuts must not change the beam data that comes out."""
    bi = BeamInterface(multifreq_beam)
    (viashortcut,) = _interp_beams_to_freq([bi], FREQ)
    direct = bi.clone(
        beam=bi.beam.interp(
            freq_array=np.array([FREQ]), new_object=True, run_check=False
        )
    )
    np.testing.assert_allclose(viashortcut.beam.data_array, direct.beam.data_array)


def test_wrangle_beams_leaves_a_ready_beam_alone(singlefreq_beam):
    """End to end: a pre-interpolated beam survives _wrangle_beams untouched."""
    bl = [singlefreq_beam] * 3
    out, nbeam, beam_idx = _wrangle_beams(None, bl, True, 3, FREQ)
    assert nbeam == 3
    assert beam_idx is None
    for o in out:
        assert o.beam.Nfreqs == 1
        assert np.isclose(np.atleast_1d(o.beam.freq_array)[0], FREQ)


def test_analytic_beams_are_untouched():
    """Analytic beams have no frequency axis to interpolate."""
    beams = [BeamInterface(GaussianBeam(diameter=14.0)) for _ in range(2)]
    out = _interp_beams_to_freq(beams, FREQ)
    assert out == beams
