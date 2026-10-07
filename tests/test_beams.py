"""Test that pixel and analytic beams are properly aligned."""

import copy

import numpy as np
import pytest
from pytest_lazy_fixtures import lf
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AnalyticBeam, GaussianBeam
from pyuvdata.beam_interface import BeamInterface
from pyuvdata.utils.pol import polnum2str

from matvis import HAVE_GPU
from matvis.core.beams import (
    DEFAULT_SPLINE_OPTS,
    _wrangle_beams,
    prepare_beam_unpolarized,
)
from matvis.cpu.beams import UVBeamInterpolator


@pytest.fixture(scope="module")
def efield_beam(uvbeam):
    """An e-field uvbeam."""
    return BeamInterface(uvbeam)


@pytest.fixture(scope="module")
def efield_beam_single_feed(efield_beam):
    """An e-field uvbeam."""
    return efield_beam.with_feeds(["x"])


@pytest.fixture(scope="function")
def efield_single_freq(uvbeam):
    """Single frequency beam."""
    return BeamInterface(uvbeam.select(freq_chans=[0], inplace=False))


@pytest.fixture(scope="module")
def power_beam(uvbeam):
    """A power-beam."""
    beam = uvbeam.copy()
    return BeamInterface(beam, beam_type="power")


@pytest.fixture(scope="function")
def efield_analytic_beam() -> AnalyticBeam:
    """An efield analytic beam."""
    return BeamInterface(GaussianBeam(diameter=14.0), beam_type="efield")


@pytest.fixture(scope="function")
def power_analytic_beam() -> AnalyticBeam:
    """A power analytic beam."""
    return BeamInterface(GaussianBeam(diameter=14.0), beam_type="efield")


class TestWrangleBeams:
    """Test the _wrangle_beams function."""

    def test_exceptions(self, efield_analytic_beam, power_beam):
        """Test that errors are raised for incorrect arguments."""
        default_kw = {
            "beam_idx": None,
            "beam_list": [efield_analytic_beam],
            "polarized": True,
            "nant": 3,
            "freq": 100e6,
        }

        with pytest.raises(ValueError, match="beam_idx must be provided"):
            _wrangle_beams(
                **(
                    default_kw
                    | {"beam_list": [efield_analytic_beam, efield_analytic_beam]}
                )
            )

        with pytest.raises(ValueError, match="beam_idx must be length nant"):
            _wrangle_beams(**(default_kw | {"beam_idx": np.zeros(2, dtype=int)}))

        with pytest.raises(
            ValueError,
            match="beam_idx contains indices greater than the number of beams",
        ):
            _wrangle_beams(**(default_kw | {"beam_idx": np.array([0, 1, 1])}))

        with pytest.raises(
            ValueError, match="beam type must be efield if using polarized=True"
        ):
            _wrangle_beams(**(default_kw | {"beam_list": [power_beam]}))


class TestPrepareBeamUnpolarized:
    """Test the prepare_beam_unpolarized function."""

    @pytest.mark.parametrize(
        "beam",
        [
            lf("efield_beam"),
            lf("power_beam"),
            lf("efield_analytic_beam"),
            lf("power_analytic_beam"),
            lf("efield_beam_single_feed"),
        ],
    )
    def test_different_input_beams(self, beam):
        """Test that prepare_beam correctly handles different beam inputs."""
        new_beam = prepare_beam_unpolarized(beam)

        assert new_beam.beam_type == "power"

        assert len(new_beam.beam.polarization_array) == 1
        assert polnum2str(new_beam.beam.polarization_array[0]).lower() == "xx"

    def test_noop(self, power_beam):
        """Test that passing in a power beam with a single pol is a no-op."""
        new_beam = power_beam.with_feeds(["x"])

        same_beam = prepare_beam_unpolarized(new_beam)
        assert same_beam is new_beam


class TestUVBeamInterpolator:
    """Test the UVBeamInterpolator."""

    def test_nan_in_cpu_beam(self, efield_beam):
        """Test nan in cpu beam."""
        beam = copy.deepcopy(efield_beam)
        beam.beam.data_array[1] = np.nan

        tx = np.linspace(-1, 1, 100)
        ty = tx

        freq = beam.beam.freq_array[0]

        bmfunc = UVBeamInterpolator(
            beam_list=[beam],
            beam_idx=np.zeros(1, dtype=int),
            polarized=True,
            nant=1,
            freq=freq,
            nsrc=len(tx),
        )
        bmfunc.setup()
        with pytest.raises(
            ValueError, match="Beam interpolation resulted in an invalid value"
        ):
            bmfunc(tx, ty, check=True)


@pytest.mark.gpu
@pytest.mark.skipif(not HAVE_GPU, reason="GPU is not available")
class TestGPUBeamInterpolator:
    """Test the GPUBeamInterpolator."""

    def setup_class(self):
        """Import the GPUBeamInterpolator."""
        from matvis.gpu.beams import GPUBeamInterpolator

        self.gpuinterp = GPUBeamInterpolator

    @pytest.mark.skipif(not HAVE_GPU, reason="GPU is not available")
    def test_exceptions(self, efield_single_freq: BeamInterface):
        """Test that proper exceptions are raised when bad params are passed."""
        beam = efield_single_freq
        beam = beam.clone(beam=beam.beam.to_healpix(inplace=False))
        bm = self.gpuinterp(
            beam_list=[beam],
            beam_idx=np.zeros(1, dtype=int),
            polarized=True,
            nant=1,
            freq=100e6,
            nsrc=100,
            spline_opts={"order": 1},
            precision=2,
        )

        with pytest.raises(ValueError, match="pixel coordinate system must be"):
            bm.setup()

    def test_analytic_beam(self, efield_analytic_beam):
        """Test that using analytic beams still works."""
        import cupy as cp

        tx = np.linspace(-0.7, 0.7, 100)
        ty = np.linspace(-0.7, 0.7, 100)

        bm = self.gpuinterp(
            beam_list=[efield_analytic_beam],
            beam_idx=np.zeros(1, dtype=int),
            polarized=True,
            nant=1,
            freq=100e6,
            nsrc=len(tx),
            spline_opts={"order": 1},
            precision=2,
        )

        bm.setup()
        interp_beam = bm(tx, ty)
        assert interp_beam.shape == (1, 2, 2, len(tx))
        assert isinstance(interp_beam, cp.ndarray)


@pytest.mark.gpu
@pytest.mark.parametrize("order", [1, 3])
def test_gpu_beam_interp_against_cpu(efield_single_freq, order):
    """Test that GPU beam interpolation matches the CPU interpolation.

    ``mode`` is deliberately left unset so that both backends take it from
    :data:`DEFAULT_SPLINE_OPTS`. At ``order=3`` that matters: the mode selects
    the B-spline prefilter, so a backend inheriting scipy's ``"constant"``
    default would disagree with the fused kernel near the edges of the grid,
    not merely outside it.
    """
    if not HAVE_GPU:
        pytest.skip("GPU is not available")

    from matvis.gpu.beams import GPUBeamInterpolator

    rt = np.linspace(0, 1, 100)
    tht = np.linspace(0, 2 * np.pi, 100)

    tx = rt * np.cos(tht)
    ty = rt * np.sin(tht)

    cpu_bmfunc = UVBeamInterpolator(
        beam_list=[efield_single_freq],
        beam_idx=np.zeros(1, dtype=int),
        polarized=True,
        nant=1,
        freq=100e6,
        nsrc=len(tx),
        spline_opts={"order": order},
        precision=2,
    )

    gpu_bmfunc = GPUBeamInterpolator(
        beam_list=[efield_single_freq],
        beam_idx=np.zeros(1, dtype=int),
        polarized=True,
        nant=1,
        freq=100e6,
        nsrc=len(tx),
        spline_opts={"order": order},
        precision=2,
    )

    np.testing.assert_allclose(
        cpu_bmfunc.beam_list[0].beam.data_array,
        gpu_bmfunc.beam_list[0].beam.data_array,
        atol=1e-8,
    )
    cpu_bmfunc.setup()
    gpu_bmfunc.setup()

    cpu_bmfunc(tx, ty)
    gpu_bmfunc(tx, ty)

    np.testing.assert_allclose(
        cpu_bmfunc.interpolated_beam, gpu_bmfunc.interpolated_beam.get(), atol=1e-6
    )


class TestDefaultSplineOpts:
    """The interpolation defaults both backends share."""

    def _interp(self, beam, **kw):
        return UVBeamInterpolator(
            beam_list=[beam],
            beam_idx=np.zeros(1, dtype=int),
            polarized=True,
            nant=1,
            freq=100e6,
            nsrc=10,
            precision=2,
            **kw,
        )

    def test_unset_opts_take_the_defaults(self, efield_single_freq):
        """An unset spline_opts must not leave order/mode to the backend."""
        assert self._interp(efield_single_freq).spline_opts == DEFAULT_SPLINE_OPTS
        assert self._interp(efield_single_freq, spline_opts=None).spline_opts == (
            DEFAULT_SPLINE_OPTS
        )

    def test_partial_opts_merge_over_the_defaults(self, efield_single_freq):
        """Keys the caller gives win; keys they omit keep their default."""
        opts = self._interp(efield_single_freq, spline_opts={"order": 1}).spline_opts
        assert opts["order"] == 1
        assert opts["mode"] == DEFAULT_SPLINE_OPTS["mode"]

    def test_caller_opts_are_not_mutated(self, efield_single_freq):
        """Merging must not write the defaults back into the caller's dict."""
        given = {"order": 1}
        self._interp(efield_single_freq, spline_opts=given)
        assert given == {"order": 1}

    def test_cpu_backend_passes_them_to_pyuvdata(self, efield_single_freq, monkeypatch):
        """Order and mode must reach map_coordinates explicitly, not via scipy's defaults.

        scipy's own default mode is "constant". That happens to share the
        "mirror" prefilter, so it agrees with the GPU kernels inside the grid
        today -- but only by coincidence, and it differs outside. Pinning the
        mode makes the agreement something the tests hold us to.
        """
        seen = {}
        original = type(efield_single_freq).compute_response

        def spy(self, **kwargs):
            seen.update(kwargs.get("spline_opts") or {})
            return original(self, **kwargs)

        monkeypatch.setattr(type(efield_single_freq), "compute_response", spy)

        bmfunc = self._interp(efield_single_freq)
        bmfunc.setup()
        bmfunc(np.linspace(-0.5, 0.5, 10), np.zeros(10))

        assert seen["order"] == DEFAULT_SPLINE_OPTS["order"] == 3
        assert seen["mode"] == DEFAULT_SPLINE_OPTS["mode"] == "mirror"
