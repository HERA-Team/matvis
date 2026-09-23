"""Test the utils module."""

import logging
import time

import psutil
import pytest

from matvis import _utils

# Default parameters that fit in a single chunk with generous memory
_BASE_KWARGS = {
    "freemem": 10 * 1024**3,  # 10 GB
    "nax": 2,
    "nfeed": 2,
    "nant": 4,
    "nsrc": 100,
    "nbeam": 2,
    "nbeampix_tot": 0,
    "precision": 1,
}


def test_human_readable_size():
    """Test the human_readable_size function."""
    assert _utils.human_readable_size(0) == "0.00 B"
    assert _utils.human_readable_size(1) == "1.00 B"
    assert _utils.human_readable_size(1023) == "1023.00 B"
    assert _utils.human_readable_size(1024) == "1.00 KiB"
    assert _utils.human_readable_size(1024**2) == "1.00 MiB"
    assert _utils.human_readable_size(1024**3) == "1.00 GiB"
    assert _utils.human_readable_size(1024**4) == "1.00 TiB"
    assert _utils.human_readable_size(1024**5) == "1.00 PiB"
    assert _utils.human_readable_size(1024**6) == "1024.00 PiB"
    assert _utils.human_readable_size(1024**6, decimal_places=3) == "1024.000 PiB"
    assert (
        _utils.human_readable_size(1024**6, decimal_places=3, indicate_sign=True)
        == "+1024.000 PiB"
    )


class TestVisBuffers:
    """Tests for the vis_buffers argument of get_required_chunks."""

    def test_defaults_to_one_per_chunk(self):
        """Omitting vis_buffers reproduces the historical behaviour."""
        assert _utils.get_required_chunks(**_BASE_KWARGS) == _utils.get_required_chunks(
            vis_buffers=None, **_BASE_KWARGS
        )

    def test_fewer_buffers_never_needs_more_chunks(self):
        """Holding a couple of vis buffers instead of one per chunk can only help.

        The GPU backend accumulates chunks into a single buffer, so its
        visibility memory no longer grows with the chunk count.
        """
        # A config where the visibility buffers are a large share of the
        # budget: many antennas, relatively few sources.
        kwargs = dict(_BASE_KWARGS)
        kwargs.update(nant=512, nsrc=20000, freemem=2 * 1024**3)
        with_accum = _utils.get_required_chunks(vis_buffers=2, **kwargs)
        per_chunk = _utils.get_required_chunks(**kwargs)
        assert with_accum <= per_chunk


class TestGetRequiredChunks:
    """Tests for get_required_chunks."""

    def test_returns_at_least_one(self):
        """Result is always >= 1."""
        result = _utils.get_required_chunks(**_BASE_KWARGS)
        assert result >= 1

    def test_lower_memory_buffer_increases_chunks(self):
        """Lower memory_buffer means less available memory, so more chunks needed."""
        chunks_high = _utils.get_required_chunks(
            **{**_BASE_KWARGS, "memory_buffer": 0.9}
        )
        chunks_low = _utils.get_required_chunks(
            **{**_BASE_KWARGS, "memory_buffer": 0.1}
        )
        assert chunks_low >= chunks_high

    def test_clamped_to_one_when_loop_never_runs(self):
        """When all data fits in memory at ch=0, result is clamped to 1 (not 0)."""
        # Very large freemem means the while-loop condition is False immediately.
        result = _utils.get_required_chunks(
            **{**_BASE_KWARGS, "freemem": 10**18, "memory_buffer": 1.0}
        )
        assert result == 1

    def test_invalid_memory_buffer_above_one(self):
        """memory_buffer > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="memory_buffer"):
            _utils.get_required_chunks(**{**_BASE_KWARGS, "memory_buffer": 1.1})

    def test_invalid_memory_buffer_zero(self):
        """memory_buffer = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="memory_buffer"):
            _utils.get_required_chunks(**{**_BASE_KWARGS, "memory_buffer": 0.0})

    def test_invalid_memory_buffer_negative(self):
        """memory_buffer < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="memory_buffer"):
            _utils.get_required_chunks(**{**_BASE_KWARGS, "memory_buffer": -0.5})

    def test_memory_buffer_exactly_one(self):
        """memory_buffer = 1.0 is valid (use all free memory)."""
        result = _utils.get_required_chunks(**{**_BASE_KWARGS, "memory_buffer": 1.0})
        assert result >= 1


class TestLogProgress:
    """Tests for log_progress."""

    def test_noop_when_info_disabled(self, caplog):
        """When INFO is not enabled, inputs are returned unchanged and nothing is logged."""
        pr = psutil.Process()
        prev_time = time.time()
        last_mem = 12345

        with caplog.at_level(logging.WARNING, logger="matvis._utils"):
            t, mem = _utils.log_progress(prev_time - 5, prev_time, 1, 10, pr, last_mem)

        assert t == prev_time
        assert mem == last_mem
        assert caplog.text == ""

    def test_logs_progress_when_info_enabled(self, caplog):
        """When INFO is enabled, progress is logged and updated time/memory returned."""
        pr = psutil.Process()
        start_time = time.time() - 10
        prev_time = time.time() - 1
        last_mem = 0

        with caplog.at_level(logging.INFO, logger="matvis._utils"):
            t, mem = _utils.log_progress(start_time, prev_time, 5, 10, pr, last_mem)

        assert t > prev_time
        assert mem > 0
        assert "Progress Info" in caplog.text
