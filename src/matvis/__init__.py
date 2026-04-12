"""A fast visibility simulator based on per-antenna calculations."""

from importlib.metadata import DistributionNotFound, version
from pathlib import Path

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, DistributionNotFound

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
