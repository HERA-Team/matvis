"""Test functions in the _utils module."""
import pytest

import logging

from vis_cpu._utils import addLoggingLevel


def test_add_logging_level():
    """Test that adding a logging level works as expected."""
    addLoggingLevel("TRACE", 5)

    assert hasattr(logging, "trace")  # method
    assert logging.TRACE == 5

    logger = logging.getLogger("test")
    start_level = logger.getEffectiveLevel()

    logger.trace("test")
    logger.setLevel(logging.DEBUG)
    logger.trace("test2")
    logging.trace("test")

    addLoggingLevel("TRACETHIS", 6, "log_a_trace")
    assert hasattr(logging, "log_a_trace")  # method

    with pytest.raises(AttributeError, match="TRACE already defined in logging module"):
        addLoggingLevel("TRACE", 10)

    with pytest.raises(
        AttributeError, match="log_a_trace already defined in logging module"
    ):
        addLoggingLevel("NEWLEVEL", 11, "log_a_trace")

    logging.getLoggerClass().a_stupid_name = lambda x: x

    with pytest.raises(
        AttributeError, match="a_stupid_name already defined in logger class"
    ):
        addLoggingLevel("STUPID", 12, "a_stupid_name")

    logger.setLevel(start_level)
