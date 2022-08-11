"""A fast visibility simulator based on per-antenna calculations."""
from pkg_resources import DistributionNotFound, get_distribution

from pathlib import Path

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

try:
    import pycuda

    HAVE_GPU = True
except ImportError:
    HAVE_GPU = False


from . import plot
from .cpu import vis_cpu
from .gpu import vis_gpu
from .wrapper import simulate_vis

DATA_PATH = Path(__file__).parent / "data"

del Path
