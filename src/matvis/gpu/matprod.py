"""GPU-accelerated source-summing operation."""

import cupy as cp
import numpy as np

from ..core.matprod import MatProd
from ._cublas import complex_matmul, finalize_zdotz, zdotz


def _pinned_empty(shape, dtype):
    """Allocate a page-locked (pinned) host array.

    Device-to-host copies out of pinned memory run on the DMA engine rather
    than being staged through a driver bounce buffer, which is roughly 3x
    faster for the few-MB visibility buffer.
    """
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(mem, dtype, int(np.prod(shape))).reshape(shape)


class _AccumulatingMatProd(MatProd):
    """Base for GPU matprods that accumulate chunks in a single device buffer.

    Rather than giving each source chunk its own visibility buffer and summing
    them at the end of the integration, every chunk adds into one buffer via
    the ``beta=1`` argument of the underlying cuBLAS call. That removes both
    the summation pass and ``nchunks - 1`` buffers' worth of device memory.

    The buffer is zeroed by :meth:`sum_chunks`, so each integration starts from
    zero. As well as being cheaper, this is what makes chunks that are entirely
    below the horizon (and so never passed to ``compute``) contribute nothing:
    with per-chunk buffers they would contribute the *previous* integration's
    result.
    """

    def __call__(self, z: cp.ndarray, chunk: int) -> cp.ndarray:
        """Accumulate the source-sum for a single chunk into the visibilities.

        ``chunk`` is accepted for interface compatibility but unused: all
        chunks accumulate into the same buffer.
        """
        self.compute(z, out=self.vis)
        return self.vis


class GPUMatMul(_AccumulatingMatProd):
    """Use cupy.gemm to perform the source-summing operation."""

    def allocate_vis(self):
        """Allocate memory for the visibilities.

        The shape here is (nant, nfeed, nant, nfeed), which is backwards
        from what you'd expect (nfeed,nant, nfeed, nant). This is because the
        fortran ordering is used in CUBLAS, which is the same as the transpose of the
        expected shape.

        A single buffer is used for all chunks (see :class:`_AccumulatingMatProd`),
        along with a device buffer holding the result in *output* ordering and a
        pinned host buffer to receive it.
        """
        # The shape is required to be like this to use the fortran ordering
        self.vis = cp.zeros(
            (self.nfeed, self.nant, self.nfeed, self.nant),
            dtype=self.ctype,
            order="F",
        )

        # Transposing on the device and downloading a contiguous buffer is far
        # cheaper than downloading the Fortran-ordered matrix and transposing
        # on the host (issue #132).
        self._dev_out = cp.empty(
            (self.npairs, self.nfeed, self.nfeed), dtype=self.ctype
        )
        self._host_out = _pinned_empty(self._dev_out.shape, self.ctype)
        if self.all_pairs:
            self._dev_out_4d = self._dev_out.reshape(
                (self.nant, self.nant, self.nfeed, self.nfeed)
            )
        else:
            self._ant1_idx = cp.asarray(self.ant1_idx)
            self._ant2_idx = cp.asarray(self.ant2_idx)

    def compute(self, z: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
        """Accumulate the source-summing operation for a single time and chunk."""
        # mirror=False: only the lower triangle is accumulated; the upper half
        # is filled in once per integration by sum_chunks.
        zdotz(z, out=out, beta=1.0, mirror=False)
        return out

    def sum_chunks(self, out: np.ndarray):
        """Write the accumulated visibilities into the output array.

        The chunks have already been summed on the device, so this completes
        the Hermitian matrix, reorders it into the output layout on the device,
        copies it down, and resets the accumulator for the next integration.

        Parameters
        ----------
        out
            The output visibilities, with shape (Npairs, Nfeed, Nfeed).
        """
        finalize_zdotz(self.vis)

        # (nfeed, nant, nfeed, nant) -> (nant, nant, nfeed, nfeed)
        transposed = self.vis.transpose((1, 3, 2, 0))
        if self.all_pairs:
            self._dev_out_4d[:] = transposed
        else:
            self._dev_out[:] = transposed[self._ant1_idx, self._ant2_idx]

        self._dev_out.get(out=self._host_out)
        out[:] = self._host_out

        self.vis.fill(0)


class GPUVectorDot(_AccumulatingMatProd):
    """Use a loop over specific pairs, performing a vdot over the source axis."""

    def allocate_vis(self):
        """Allocate memory for the visibilities."""
        self.vis = cp.zeros(
            (self.nfeed, self.nfeed, self.npairs), dtype=self.ctype, order="F"
        )

    def compute(self, z: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
        """Accumulate the source-summing operation for a single time and chunk."""
        z = z.reshape((self.nant, self.nfeed, -1))

        for i, (ai, aj) in enumerate(self.antpairs):
            complex_matmul(z[ai], z[aj], out=out[:, :, i], beta=1.0)
        return out

    def sum_chunks(self, out: np.ndarray):
        """Write the accumulated visibilities into the output array."""
        out[:] = self.vis.transpose((2, 1, 0)).get()
        self.vis.fill(0)


class GPUMatBlock(_AccumulatingMatProd):
    """Compute a set of rectangular antenna-block sub-matrix products on the GPU.

    GPU counterpart of :class:`~matvis.cpu.matprod.CPUMatBlock`; see its
    docstring for the rationale. Uses :func:`~matvis.gpu._cublas.complex_matmul`
    (which supports rectangular, non-square blocks) per block, and never calls
    any host/device synchronization inside :meth:`compute` -- all gather/scatter
    is done with device-side indexing, keeping the per-chunk loop async like the
    rest of this repo's GPU pipeline (see ``gpu.py``'s single-stream design).
    """

    supports_blocks = True

    def setup(self):
        """Set up memory and upload the block-dispatch plan to the device once."""
        super().setup()
        # Slice selectors stay host-side slices (they index z without a copy);
        # everything else is an index array that must live on the device.
        self._gpu_block_plan = [
            entry._replace(
                **{
                    f: cp.asarray(v)
                    for f, v in zip(entry._fields, entry, strict=True)
                    if isinstance(v, np.ndarray)
                }
            )
            for entry in self._block_plan
        ]

    def allocate_vis(self):
        """Allocate one (npairs, nfeed, nfeed) accumulator for all chunks."""
        self.vis = cp.zeros((self.npairs, self.nfeed, self.nfeed), dtype=self.ctype)

    def compute(self, z: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
        """Accumulate the source-summing operation for a single time and chunk."""
        z = z.reshape((self.nant, self.nfeed, -1))

        for blk in self._gpu_block_plan:
            # Only genuinely scattered antenna sets pay for a staging copy;
            # consecutive ones are read straight out of z. That staging used to
            # be about half of this class's runtime (issue #161), which is why
            # the drivers order z's antenna axis to avoid it.
            zr = self._operand(z, blk.rows, blk.nrow)
            zc = self._operand(z, blk.cols, blk.ncol)

            # complex_matmul's output is F-contiguous, but its *values* at
            # flat index (local_row*nfeed + feed) match the row-major
            # convention zr/zc were built with -- so reshape with the default
            # (C) order semantics, same as the CPU path, not the array's own
            # (F) memory layout. cupy (like numpy) copies as needed here.
            block = complex_matmul(zr, zc)
            block = block.reshape((blk.nrow, self.nfeed, blk.ncol, self.nfeed))
            block = block.transpose((0, 2, 3, 1))  # -> (rows, cols, nfeed_j, nfeed_i)

            # `+=`, not `=`: every chunk adds into the one accumulator (see
            # _AccumulatingMatProd). Each slot is claimed by exactly one block
            # and antpairs are deduplicated, so the scattered indices are
            # unique and this needs no atomics.
            if blk.slots.size:
                out[blk.slots] += block[blk.lr, blk.lc]
            if blk.rslots.size:
                # This block holds the reversed pair; V_ij is the Hermitian
                # conjugate of V_ji over the two feed axes.
                out[blk.rslots] += block[blk.rlr, blk.rlc].conj().transpose((0, 2, 1))

        return out

    def _operand(self, z: cp.ndarray, sel, nant: int) -> cp.ndarray:
        """Stage one side of a block product as a C-contiguous cuBLAS operand.

        ``sel`` is a ``slice`` when the block's antennas are consecutive, and
        then ``z[sel]`` is already a contiguous view that cuBLAS can read in
        place; ``ascontiguousarray`` passes it through untouched. Otherwise the
        fancy index gathers the rows into a staging copy.
        """
        return cp.ascontiguousarray(z[sel]).reshape(nant * self.nfeed, -1)

    def sum_chunks(self, out: np.ndarray):
        """Write the accumulated visibilities into the output array."""
        out[:] = self.vis.get()
        self.vis.fill(0)
