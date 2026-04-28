"""A fast visibility simulator based on per-antenna calculations."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

try:
    import cupy

    HAVE_GPU = True
except ImportError:
    HAVE_GPU = False


from . import cpu, gpu
from .wrapper import simulate_vis

DATA_PATH = Path(__file__).parent / "data"

del Path
