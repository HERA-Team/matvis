"""Base class for performing the source-summing operation."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np

from .._utils import get_dtypes


class MatProd(ABC):
    """
    Abstract base class for performing the source-summing operation.

    Parameters
    ----------
    nchunks
        Number of chunks to split the sources into.
    nfeed
        Number of feeds.
    nant
        Number of antennas.
    antpairs
        The antenna pairs to sum over. If None, all pairs are used.
    precision
        The precision of the data (1 or 2).
    antenna_blocks
        Advanced/optional. A list of ``(row_antenna_idx, col_antenna_idx)`` tuples
        of integer antenna-index arrays, one per rectangular sub-matrix block to
        compute, instead of the full ``Nant x Nant`` product. Only understood by
        matprod classes that declare ``supports_blocks = True`` (currently
        :class:`~matvis.cpu.matprod.CPUMatBlock` and
        :class:`~matvis.gpu.matprod.GPUMatBlock`); passing it to any other class
        is a ``ValueError`` rather than a silent no-op. See
        :mod:`matvis.redundancy` for helpers that build sensible block lists
        (e.g. from caller-supplied antenna groupings, or a generic memory-bounded
        tiling of the full antenna set). Every entry of ``antpairs`` must be
        covered by at least one block; this is validated in :meth:`setup`.
    """

    #: Whether this class understands ``antenna_blocks``. Subclasses that do
    #: must set this to True; it exists so that passing ``antenna_blocks`` to a
    #: class that would otherwise silently ignore it is a hard error instead of
    #: a confusing footgun.
    supports_blocks: bool = False

    def __init__(
        self,
        nchunks: int,
        nfeed: int,
        nant: int,
        antpairs: np.ndarray | None,
        precision=1,
        *,
        antenna_blocks: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        if antpairs is None:
            self.all_pairs = True
            self.antpairs = np.array([(i, j) for i in range(nant) for j in range(nant)])
        else:
            self.all_pairs = False
            self.antpairs = antpairs
            seen = set()
            for i, j in self.antpairs:
                pair = (int(i), int(j))
                if pair in seen:
                    raise ValueError(
                        f"antpairs contains duplicate pair {pair}; matvis does not "
                        "support requesting the same antenna pair more than once."
                    )
                seen.add(pair)

        if antenna_blocks is not None and not self.supports_blocks:
            raise ValueError(
                f"{type(self).__name__} does not support antenna_blocks "
                "(it always computes the full Nant x Nant product); use "
                "CPUMatBlock/GPUMatBlock if you need block-decomposed matprod."
            )
        if antenna_blocks is None and self.supports_blocks:
            raise ValueError(
                f"{type(self).__name__} requires antenna_blocks to be provided "
                "(see matvis.redundancy for helpers to build one)."
            )
        self.antenna_blocks = antenna_blocks

        self.nchunks = nchunks
        self.nfeed = nfeed
        self.nant = nant

        self.npairs = len(self.antpairs)
        self.ctype = get_dtypes(precision)[1]

        self.ant1_idx = self.antpairs[:, 0]
        self.ant2_idx = self.antpairs[:, 1]

    def allocate_vis(self):
        """Allocate memory for the visibilities.

        The shape of the visibilities must have a first axis of length nchunks,
        but then can be arbitrary shaped after that, so long as it is consistently
        used throughout the class.
        """
        self.vis = np.full(
            (self.nchunks, self.npairs, self.nfeed, self.nfeed), 0.0, dtype=self.ctype
        )

    def _prepare_antenna_blocks(self):
        """Validate ``antenna_blocks`` and precompute the block dispatch plan.

        Builds ``self._block_plan``: one entry per block that contributes at
        least one requested pair, holding the block's row/column antenna indices
        plus two sets of gather indices -- one for pairs the block holds in the
        requested orientation, and one for pairs it holds *reversed*. A reversed
        pair is still exact, since ``V_ij`` is the Hermitian conjugate (in feed
        space) of ``V_ji``; supporting it is what lets a decomposition permute
        and flip the antenna axes freely when hunting for dense sub-matrices.

        Deliberately avoids ever materializing an ``(Nant, Nant)``-shaped
        structure: coverage is tracked with a dict keyed by ``(i, j)``, sized by
        the number of *requested* pairs, not by ``Nant**2``.

        Raises
        ------
        ValueError
            If any block's index arrays are malformed, or if any requested
            antpair is not covered by any block in either orientation.
        """
        normalized = []
        for b, (rows, cols) in enumerate(self.antenna_blocks):
            rows = np.asarray(rows)
            cols = np.asarray(cols)
            if rows.ndim != 1 or cols.ndim != 1:
                raise ValueError(
                    f"antenna_blocks[{b}]: row/col index arrays must be 1-D, "
                    f"got shapes {rows.shape} and {cols.shape}"
                )
            if rows.size == 0 or cols.size == 0:
                raise ValueError(
                    f"antenna_blocks[{b}]: row/col arrays must be non-empty"
                )
            if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(
                cols.dtype, np.integer
            ):
                raise ValueError(
                    f"antenna_blocks[{b}]: row/col index arrays must have integer dtype"
                )
            if (
                rows.min() < 0
                or rows.max() >= self.nant
                or cols.min() < 0
                or cols.max() >= self.nant
            ):
                raise ValueError(
                    f"antenna_blocks[{b}]: indices must be in [0, {self.nant}), "
                    f"got rows in [{rows.min()}, {rows.max()}], "
                    f"cols in [{cols.min()}, {cols.max()}]"
                )
            normalized.append((rows, cols))

        # Map each requested (i, j) pair to the list of output slots that want
        # it (supports duplicate entries in antpairs), sized by npairs, not by
        # nant**2.
        pair_to_slots = defaultdict(list)
        for slot, (i, j) in enumerate(self.antpairs):
            pair_to_slots[(int(i), int(j))].append(slot)

        block_plan = []
        for rows, cols in normalized:
            row_pos = {int(a): lr for lr, a in enumerate(rows)}
            col_pos = {int(a): lc for lc, a in enumerate(cols)}

            direct: tuple[list, list, list] = ([], [], [])
            reversed_: tuple[list, list, list] = ([], [], [])

            # Two passes over the *remaining* pairs (not over the block's full
            # row x col grid, which would scale with the block area): claim
            # everything the block holds directly first, so the cheaper path is
            # always preferred, then mop up whatever it holds reversed.
            for target, (get_row, get_col) in (
                (direct, (lambda i, j: i, lambda i, j: j)),
                (reversed_, (lambda i, j: j, lambda i, j: i)),
            ):
                for (i, j), slots in list(pair_to_slots.items()):
                    ra, ca = get_row(i, j), get_col(i, j)
                    if ra in row_pos and ca in col_pos:
                        lr, lc = row_pos[ra], col_pos[ca]
                        target[0].extend([lr] * len(slots))
                        target[1].extend([lc] * len(slots))
                        target[2].extend(slots)
                        del pair_to_slots[(i, j)]

            if direct[2] or reversed_[2]:
                block_plan.append(
                    (
                        rows,
                        cols,
                        *(np.array(a, dtype=np.intp) for a in direct),
                        *(np.array(a, dtype=np.intp) for a in reversed_),
                    )
                )

        if pair_to_slots:
            missing = list(pair_to_slots.keys())
            raise ValueError(
                f"{len(missing)} requested antpairs are not covered by any "
                f"antenna_blocks entry (in either orientation), "
                f"e.g. {missing[:10]}"
            )

        self._block_plan = block_plan

    def setup(self):
        """Setup the memory for the object."""
        self.allocate_vis()
        if self.antenna_blocks is not None:
            self._prepare_antenna_blocks()

    @abstractmethod
    def compute(self, z: np.ndarray, out: np.ndarray):
        """
        Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nant, Nfeed, Nax, Nsrc).
        out
            Output array, shaped like the visibilities set in `allocate_vis`, but
            without the first chunk axis.
        """

    def __call__(self, z: np.ndarray, chunk: int) -> np.ndarray:
        """
        Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nant, Nfeed, Nax, Nsrc).
        chunk
            The chunk index.

        Returns
        -------
        out
            The output array, shaped like the visibilities set in `allocate_vis`, but
            without the first chunk axis.
        """
        self.compute(z, out=self.vis[chunk])
        return self.vis[chunk]

    def sum_chunks(self, out: np.ndarray):
        """
        Sum the chunks into the output array.

        Parameters
        ----------
        out
            The output visibilities, with shape (Nfeed, Nfeed, Npairs).
        """
        if self.nchunks == 1:
            out[:] = self.vis[0]
        else:
            self.vis.sum(axis=0, out=out)
