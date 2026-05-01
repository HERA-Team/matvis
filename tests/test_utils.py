"""Test the utils module."""

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
    "nbeampix": 0,
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
