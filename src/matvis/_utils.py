import datetime
import itertools
import logging
import numpy as np
import psutil
import time
import tracemalloc as tm
from typing import Union

try:
    import cupy as cp

    ArrayType = Union[np.ndarray, cp.ndarray]
    HAVE_CUDA = True
except ImportError:
    ArrayType = np.ndarray
    HAVE_CUDA = False

logger = logging.getLogger(__name__)


def no_op(fnc):
    """No-op function."""
    return fnc


def ceildiv(a: int, b: int) -> int:
    """Ceiling division for integers.

    From https://stackoverflow.com/a/17511341/1467820
    """
    return -(a // -b)


def human_readable_size(size, decimal_places=2, indicate_sign=False):
    """Get a human-readable data size.

    From: https://stackoverflow.com/a/43690506/1467820
    """
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if abs(size) < 1024.0:
            break
        if unit != "PiB":
            size /= 1024.0

    if indicate_sign:
        return f"{size:+.{decimal_places}f} {unit}"
    else:
        return f"{size:.{decimal_places}f} {unit}"


def memtrace(highest_peak: int) -> int:
    """
    Log the memory usage since last call.

    Parameters
    ----------
    highest_peak : int
        The highest peak memory usage in bytes.

    Returns
    -------
    int
        The new highest peak memory usage.
    """
    if logger.isEnabledFor(logging.INFO):
        cm, pm = tm.get_traced_memory()
        logger.info(f"Starting Memory usage  : {cm / 1024**3:.3f} GB")
        logger.info(f"Starting Peak Mem usage: {pm / 1024**3:.3f} GB")
        logger.info(f"Traemalloc Peak Memory (tot)(GB): {highest_peak / 1024**3:.2f}")
        tm.reset_peak()
        return max(pm, highest_peak)


def logdebug(name: str, x: ArrayType):
    """Debug logging of the value of an array."""
    if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
        loc = "GPU" if HAVE_CUDA and isinstance(x, cp.ndarray) else "CPU"

        cornerstr = "".join(
            f"\t{idx}: {x[idx]}\n"
            for idxc in itertools.combinations_with_replacement([0, -1], x.ndim)
            for idx in set(itertools.permutations(idxc))
        )
        logger.debug(f"{loc}: {name} <{x.shape}> [{x.dtype}]:\n{cornerstr}")


def log_progress(
    start_time: float,
    prev_time: float,
    iters: int,
    niters: int,
    pr: psutil.Process,
    last_mem: float,
) -> tuple[float, int]:
    """Logging of progress."""
    if not logger.isEnabledFor(logging.INFO):
        return prev_time, last_mem

    t = time.time()
    lapsed = datetime.timedelta(seconds=(t - prev_time))
    total = datetime.timedelta(seconds=(t - start_time))
    per_iter = total / iters
    expected = per_iter * niters

    rss = pr.memory_info().rss
    mem = human_readable_size(rss)
    memdiff = human_readable_size(rss - last_mem, indicate_sign=True)

    logger.info(
        f"""
        Progress Info   [{iters}/{niters} times ({100 * iters / niters:.1f}%)]
            -> Update Time:   {lapsed}
            -> Total Time:    {total} [{per_iter} per integration]
            -> Expected Time: {expected} [{expected - total} remaining]
            -> Memory Usage:  {mem}  [{memdiff}]
        """
    )

    return t, rss


def get_required_chunks(
    freemem: int,
    nax: int,
    nfeed: int,
    nant: int,
    nsrc: int,
    nbeam: int,
    nbeampix: int,
    precision: int,
    source_buffer: float = 0.55,
) -> int:
    """
    Compute number of chunks (over sources) required to fit data into available memory.

    Parameters
    ----------
    freemem : int
        The amount of free memory in bytes.
    nax : int
        The number of axes.
    nfeed : int
        The number of feeds.
    nant : int
        The number of antennas.
    nsrc : int
        The number of sources.
    nbeam : int
        The number of beams.
    nbeampix : int
        The number of beam pixels.
    precision : int
        The precision of the data.

    Returns
    -------
    int
        The number of chunks required.

    Examples
    --------
    >>> get_required_chunks(1024, 2, 4, 8, 16, 32, 64, 32)
    1
    """
    rsize = 4 * precision
    csize = 2 * rsize

    gpusize = {"a": freemem}
    ch = 0
    while sum(gpusize.values()) >= freemem and ch < 100:
        ch += 1
        nchunk = int(nsrc // ch * source_buffer)

        gpusize = {
            "antpos": nant * 3 * rsize,
            "flux": nsrc * rsize,
            "beam": nbeampix * nfeed * nax * csize,
            "crd_eq": 3 * nsrc * rsize,
            "eq2top": 3 * 3 * rsize,
            "crd_top": 3 * nsrc * rsize,
            "crd_chunk": 3 * nchunk * rsize,
            "flux_chunk": nchunk * rsize,
            "exptau": nant * nchunk * csize,
            "beam_interp": nbeam * nfeed * nax * nchunk * csize,
            "zmat": nchunk * nfeed * nant * nax * csize,
            "vis": ch * nfeed * nant * nfeed * nant * csize,
        }
        logger.debug(
            f"nchunks={ch}. Array Sizes (bytes)={gpusize}. Total={sum(gpusize.values())}"
        )

    logger.info(
        f"Total free mem: {freemem / (1024**3):.2f} GB. Requires {ch} chunks "
        f"(estimate {sum(gpusize.values()) / 1024**3:.2f} GB)"
    )
    return ch


def get_desired_chunks(
    freemem: int,
    min_chunks: int,
    beam_list: int,
    nax: int,
    nfeed: int,
    nant: int,
    nsrc: int,
    precision: int,
    source_buffer: float = 0.55,
) -> tuple[int, int]:
    """Get the desired number of chunks.

    Parameters
    ----------
    freemem : int
        The amount of free memory in bytes.
    min_chunks : int
        The minimum number of chunks desired.
    beam_list : list
        A list of beams.
    nax : int
        The number of axes.
    nfeed : int
        The number of feeds.
    nant : int
        The number of antennas.
    nsrc : int
        The number of sources.
    precision : int
        The precision of the data.

    Returns
    -------
    nchunk
        Number of chunks
    nsrcs_per_chunk
        Number of sources per chunk

    Examples
    --------
    >>> get_desired_chunks(1024, 2, [beam1, beam2], 3, 4, 8, 16, 32)
    (2, 8)
    """
    nbeampix = sum(
        beam.data_array.shape[-2] * beam.data_array.shape[-1]
        for beam in beam_list
        if hasattr(beam, "data_array")
    )

    nchunks = min(
        max(
            min_chunks,
            get_required_chunks(
                freemem,
                nax,
                nfeed,
                nant,
                nsrc,
                len(beam_list),
                nbeampix,
                precision,
                source_buffer,
            ),
        ),
        nsrc,
    )

    return nchunks, int(np.ceil(nsrc / nchunks))


def get_dtypes(precision: int):
    """
    Get the data types for the given precision.

    Parameters
    ----------
    precision : int
        The precision. 1 for single, 2 for double.

    Returns
    -------
    real_dtype
        The real data type.
    complex_dtype
        The complex data type.

    Examples
    --------
    >>> get_dtypes(1)
    (numpy.float32, numpy.complex64)
    """
    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    return real_dtype, complex_dtype
