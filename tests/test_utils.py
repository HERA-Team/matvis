"""Test the utils module."""

from matvis import _utils


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
