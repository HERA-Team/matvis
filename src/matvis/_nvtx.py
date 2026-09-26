"""NVTX range annotation, usable from any module and without cupy installed.

The GPU loop is annotated with NVTX ranges so that an ``nsys`` trace can
attribute device time -- and device *idle* time -- to the algorithmic stage the
host was inside. Ranges are a no-op when cupy is not available, so annotated
code stays importable on a CPU-only install.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

try:
    from cupy.cuda import nvtx as _nvtx

    @contextmanager
    def nvtx_range(name: str):
        """Annotate a block as an NVTX range (visible in nsys timelines)."""
        _nvtx.RangePush(name)
        try:
            yield
        finally:
            _nvtx.RangePop()

except ImportError:

    def nvtx_range(name: str):  # noqa: D103
        return nullcontext()
