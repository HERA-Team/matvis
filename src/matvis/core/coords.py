"""Core abstract class for coordinate rotation."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from .._utils import get_dtypes
from ..coordinates import calc_coherency_rotation, enu_to_az_za

try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False

# Grid geometry of the horizon-compaction kernels; must match the #defines in
# kernels/horizon_compact.cu.
_HC_BLOCK = 256
_HC_NBLOCKS = 256

_HC_MODULE = None


def _hc_module():
    """Lazily compile the horizon-compaction CUDA module."""
    global _HC_MODULE
    if _HC_MODULE is None:
        path = Path(__file__).parent.parent / "gpu" / "kernels" / "horizon_compact.cu"
        _HC_MODULE = cp.RawModule(code=path.read_text())
    return _HC_MODULE


class CoordinateRotation(ABC):
    """Abstract class for converting Equatorial (RA/DEC) coordinates to observed.

    Subclasses must, at the very least, implement the ``rotate(t)`` method, which takes
    an integer ``t``, which indexes into the ``times`` array, and sets the
    ``all_coords_topo`` attribute, which are unit-vector topocentric coordinates in the
    horizontal ENU frame of the telescope.

    All defined subclasses of this class can be found in the
    ``CoordinateRotation._methods`` dictionary.
    """

    _methods = {}
    requires_gpu: bool = False

    def __init_subclass__(cls) -> None:
        """Register the subclass."""
        CoordinateRotation._methods[cls.__name__] = cls
        return super().__init_subclass__()

    def __init__(
        self,
        flux: np.ndarray,
        times: Time,
        telescope_loc: EarthLocation,
        skycoords: SkyCoord,
        chunk_size: int | None = None,
        source_buffer: float = 0.55,
        precision: int = 1,
        gpu: bool = False,
    ):
        self.gpu = gpu
        if self.gpu and not HAVE_CUDA:
            raise ValueError("GPU requested but cupy not installed.")

        self.xp = cp if self.gpu else np

        self.precision = precision
        self.rtype, self.ctype = get_dtypes(precision)

        # Check if the flux is complex and set the dtype accordingly.
        if self.xp.iscomplexobj(flux):
            self.sky_model_dtype = self.ctype
        else:
            self.sky_model_dtype = self.rtype

        self.flux = self.xp.asarray(flux.astype(self.sky_model_dtype))
        self._polarized = flux.ndim == 4
        self.nsrc = len(flux)
        self.times = times
        self.telescope_loc = telescope_loc
        self.skycoords = skycoords

        assert times.ndim == 1
        assert len(skycoords) == self.nsrc
        assert len(flux) == self.nsrc

        self.chunk_size = chunk_size or self.nsrc
        self.nchunks = -(-self.nsrc // self.chunk_size)
        self.source_buffer = source_buffer
        if self.chunk_size > 1000:
            self.nsrc_alloc = int(self.chunk_size * self.source_buffer)
        else:
            self.nsrc_alloc = self.chunk_size

        # A device-side horizon cut is only possible for the simple case of a
        # real, one-value-per-source flux: the 4D-flux (polarized-sky) path
        # needs host-side astropy indexing for the coherency rotation anyway.
        self._use_gpu_compaction = (
            self.gpu
            and not self._polarized
            and self.flux.ndim == 1
            and self.sky_model_dtype == self.rtype
        )

    def setup(self):
        """Allocate memory for the rotation."""
        # Initialize arrays that all subclasses must use.
        self.all_coords_topo = self.xp.full(
            (3, self.nsrc), self.rtype(0.0), dtype=self.rtype
        )
        self.coords_above_horizon = self.xp.full(
            (3, self.nsrc_alloc), self.rtype(0.0), dtype=self.rtype
        )
        self.flux_above_horizon = self.xp.full(
            (self.nsrc_alloc,) + self.flux.shape[1:],
            self.sky_model_dtype(0.0),
            dtype=self.sky_model_dtype,
        )

        if self._use_gpu_compaction:
            self._setup_compaction()

    def _setup_compaction(self):
        """Allocate the scratch buffers used by the device-side horizon cut."""
        suffix = "f32" if self.rtype == np.float32 else "f64"
        mod = _hc_module()
        self._hc_count_kernel = mod.get_function(f"hc_count_{suffix}")
        self._hc_scan_kernel = mod.get_function("hc_scan")
        self._hc_compact_kernel = mod.get_function(f"hc_compact_{suffix}")

        # Per-block hit counts, overwritten in place with their exclusive scan.
        self._hc_block_counts = cp.zeros((self.nchunks, _HC_NBLOCKS), dtype=np.int32)
        self._hc_counts = cp.zeros(self.nchunks, dtype=np.int32)
        self._hc_seg = -(-max(self.chunk_size, 1) // _HC_NBLOCKS)
        # Host mirror of _hc_counts, and the time index it was computed for.
        self._hc_counts_host = None
        self._hc_counts_time = None

    def _count_above_horizon(self, t: int) -> np.ndarray:
        """Count the sources above the horizon in every chunk, in one pass.

        This is the only host synchronization left in the horizon cut, and it
        happens once per integration rather than once per chunk. The counts are
        needed on the host so that chunks with nothing above the horizon can be
        skipped entirely, and so that an over-full chunk raises where the caller
        can act on it.
        """
        if self._hc_counts_time == t:
            return self._hc_counts_host

        self._hc_count_kernel(
            (_HC_NBLOCKS, self.nchunks),
            (_HC_BLOCK,),
            (
                self.all_coords_topo[2],
                np.int64(self.nsrc),
                np.int64(self.chunk_size),
                np.int64(self._hc_seg),
                self._hc_block_counts,
            ),
        )
        self._hc_scan_kernel(
            (self.nchunks,), (_HC_BLOCK,), (self._hc_block_counts, self._hc_counts)
        )
        counts = self._hc_counts.get()

        biggest = int(counts.max()) if len(counts) else 0
        if biggest > self.nsrc_alloc:
            raise ValueError(
                f"nsrc_alloc ({self.nsrc_alloc}) is too small for the number of "
                f"sources above horizon ({biggest}). Try increasing source_buffer."
            )

        self._hc_counts_host = counts
        self._hc_counts_time = t
        return counts

    def select_chunk(self, chunk: int, t: int):
        """
        Set the chunk of coordinates to rotate.

        This function sets the chunk of coordinates to rotate. It is used following the
        `rotate` method and returns a chunk of sources coordinates and fluxes for the sources
        above the horizon. If the sky model is polarized, it also rotates the frame of the
        coherency matrix to the alt/az frame. The chunk size is determined by the `chunk_size`
        parameter.

        Returns
        -------
        coords_above_horizon
            Topocentric coordinates of the surviving sources, shape
            ``(3, nsrc_alloc)``, padded at the tail.
        flux_above_horizon
            Square-root fluxes of the surviving sources, zero-padded at the tail.
        nsrcs_up
            The number of sources above the horizon.
        """
        if self._use_gpu_compaction:
            return self._select_chunk_compacted(chunk, t)

        # The last index can be larger than the actual size of the array without error.
        slc = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)

        topo = self.all_coords_topo[:, slc]
        flux = self.flux[slc]

        above_horizon = self.xp.where(topo[2] > 0)[0]
        n = len(above_horizon)
        if n > self.nsrc_alloc:
            raise ValueError(
                f"nsrc_alloc ({self.nsrc_alloc}) is too small for the number of "
                f"sources above horizon ({n}). Try increasing source_buffer."
            )

        if self._polarized:
            # Compute the alt/az coordinates for the sources above the horizon.
            az, za = enu_to_az_za(
                enu_e=topo[0, above_horizon],
                enu_n=topo[1, above_horizon],
                orientation="astropy",
            )

            # For polarized flux, rotate the frame coherency
            self.flux_above_horizon[:n] = self._rotate_frame_coherency(
                coherency_matrix=flux[above_horizon],
                ra=self.skycoords.ra.rad[slc][above_horizon],
                dec=self.skycoords.dec.rad[slc][above_horizon],
                alt=np.pi / 2 - za,
                az=az,
                time=self.times[t],
            )
        else:
            # For unpolarized flux, just copy the flux.
            self.flux_above_horizon[:n] = flux[above_horizon]

        self.coords_above_horizon[:, :n] = topo[:, above_horizon]
        self.flux_above_horizon[n:] = 0

        return self.coords_above_horizon, self.flux_above_horizon, n

    def _select_chunk_compacted(self, chunk: int, t: int):
        """Perform the horizon cut for one chunk with a single device kernel.

        Produces exactly the same buffers as the ``cp.where`` branch of
        :meth:`select_chunk` -- same sources, same order -- but without
        synchronizing the stream: the per-chunk counts were already gathered in
        one batch by :meth:`_count_above_horizon`. The tail of the buffers is
        padded with zero flux at the zenith.
        """
        counts = self._count_above_horizon(t)
        offset = chunk * self.chunk_size
        if chunk >= self.nchunks:
            # get_desired_chunks can hand the caller more chunks than there are
            # sources for; those are empty, and the cp.where path zeroes the
            # whole flux buffer for them.
            self.flux_above_horizon[:] = 0
            return self.coords_above_horizon, self.flux_above_horizon, 0

        n = min(self.chunk_size, self.nsrc - offset)
        total = int(counts[chunk])
        # Launched even when nothing is above the horizon: the same kernel pads
        # the tail, and leaving the previous chunk's values in the buffers would
        # break the "flux beyond nsrcs_up is zero" invariant.
        self._hc_compact_kernel(
            (_HC_NBLOCKS,),
            (_HC_BLOCK,),
            (
                self.all_coords_topo,
                np.int64(self.nsrc),
                np.int64(offset),
                self.flux,
                np.int64(n),
                np.int64(self._hc_seg),
                self._hc_block_counts[chunk],
                self.coords_above_horizon,
                self.flux_above_horizon,
                np.int64(self.nsrc_alloc),
                np.int64(total),
            ),
        )
        return self.coords_above_horizon, self.flux_above_horizon, total

    def _rotate_frame_coherency(self, coherency_matrix, ra, dec, alt, az, time) -> None:
        """
        Rotate the frame of the coherency matrix.

        This function rotates the coherency matrix from the equatorial frame to the
        alt/az frame. It is used in the `rotate` method of subclasses.
        """
        # Calculate the rotation matrix for the current time.
        coherency_rotator = calc_coherency_rotation(
            ra=ra,
            dec=dec,
            alt=alt,
            az=az,
            time=time,
            location=self.telescope_loc,
        )

        # Rotate the coherency matrix. Note that here the coherency rotator is a matrix of
        # size (2, 2, nsources), and coherency matrix is a matrix of size
        # (nsources, nfreq, 2, 2).
        coherency_matrix = self.xp.einsum(
            "ban,nfbc,cdn->nfad", coherency_rotator, coherency_matrix, coherency_rotator
        )
        return coherency_matrix

    @abstractmethod
    def rotate(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Perform the rotation for a single time."""
