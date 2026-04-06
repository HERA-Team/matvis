"""A fast visibility simulator based on per-antenna calculations."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

try:
    import cupy

    HAVE_GPU = True
except ImportError:
    HAVE_GPU = False


from . import cpu
from .wrapper import simulate_vis

if HAVE_GPU:
    from . import gpu

DATA_PATH = Path(__file__).parent / "data"

del Path
