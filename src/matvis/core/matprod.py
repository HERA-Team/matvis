"""Base class for performing the source-summing operation."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, NamedTuple

import numpy as np

from .._utils import get_dtypes


class BlockPlan(NamedTuple):
    """Everything :meth:`MatProd.compute` needs to evaluate one antenna block.

    ``rows``/``cols`` are *selectors* into the antenna axis of ``z``, not
    necessarily index arrays: when a block's antenna set happens to be an
    ascending run of consecutive antennas, the selector is a :class:`slice`, so
    that ``z[rows]`` is a zero-copy view instead of a fancy-index copy. See
    :func:`as_contiguous_slice`.
    """

    #: Selector for the block's row antennas (``slice`` or integer index array).
    rows: Any
    #: Selector for the block's column antennas.
    cols: Any
    #: Number of row antennas (``rows`` may be a slice, so this can't be ``len``).
    nrow: int
    #: Number of column antennas.
    ncol: int
    #: Row/column/output-slot indices for pairs this block holds as requested.
    lr: np.ndarray
    lc: np.ndarray
    slots: np.ndarray
    #: Row/column/output-slot indices for pairs this block holds *reversed*.
    rlr: np.ndarray
    rlc: np.ndarray
    rslots: np.ndarray


def as_contiguous_slice(idx: np.ndarray) -> slice | None:
    """Return a ``slice`` equivalent to ``idx``, or None if there isn't one.

    Only an ascending run of *consecutive* integers qualifies, since that is
    exactly the case in which ``z[idx]`` can be replaced by the zero-copy view
    ``z[slice]`` while keeping both the selected rows and their order identical.
    """
    if idx.size == 0:
        return None
    if idx.size > 1 and not np.all(np.diff(idx) == 1):
        return None
    return slice(int(idx[0]), int(idx[-1]) + 1)


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
    antenna_order
        Advanced/optional. The antenna order the rows of ``z`` will arrive in:
        ``antenna_order[p]`` is the index (in ``antpairs``/``antenna_blocks``
        numbering) of the antenna occupying row ``p`` of ``z``. Defaults to the
        natural order. Only meaningful alongside ``antenna_blocks``, and only
        useful if the ``z`` you pass to :meth:`compute` really is built that way
        -- pass the *same* array to
        :class:`~matvis.core.getz.ZMatrixCalc`, which is what the ``matvis``
        drivers do. Choosing it with
        :func:`~matvis.redundancy.contiguity_order` lets most blocks be handed
        to BLAS as views of ``z`` rather than gathered copies; see that
        function for why this matters.
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
        antenna_order: np.ndarray | None = None,
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

        if antenna_order is not None:
            antenna_order = np.asarray(antenna_order)
            if antenna_blocks is None:
                raise ValueError(
                    "antenna_order is only meaningful together with "
                    "antenna_blocks; the other matprod classes always read z in "
                    "the natural antenna order."
                )
            if antenna_order.shape != (nant,) or sorted(antenna_order.tolist()) != list(
                range(nant)
            ):
                raise ValueError(
                    f"antenna_order must be a permutation of range({nant}), got "
                    f"an array of shape {antenna_order.shape}"
                )
        self.antenna_order = antenna_order

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

        # Everything below works in *row-of-z* space rather than antenna-index
        # space. They only differ when the caller has told us z is built in a
        # non-natural antenna order (see the antenna_order docstring); pushing
        # the relabelling in here keeps it out of the per-chunk compute path.
        if self.antenna_order is None:
            row_of = np.arange(self.nant)
        else:
            row_of = np.empty(self.nant, dtype=np.intp)
            row_of[self.antenna_order] = np.arange(self.nant)
        # Sorted, so a block whose rows land on consecutive z rows is detected
        # as such below regardless of the order the caller listed them in.
        normalized = [(np.sort(row_of[r]), np.sort(row_of[c])) for r, c in normalized]

        # Map each requested (i, j) pair to the list of output slots that want
        # it (supports duplicate entries in antpairs), sized by npairs, not by
        # nant**2.
        pair_to_slots = defaultdict(list)
        for slot, (i, j) in enumerate(self.antpairs):
            pair_to_slots[(int(row_of[i]), int(row_of[j]))].append(slot)

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
                # A block whose antennas happen to be a consecutive run needs no
                # gather at all: `z[slice]` is a view, and the per-block gather
                # is about half of MatBlock's runtime (see issue #161). This is
                # common in practice -- e.g. a block whose column set is "every
                # antenna but one" is a run whenever the odd one out sits at
                # either end of the antenna axis.
                block_plan.append(
                    BlockPlan(
                        as_contiguous_slice(rows) or rows,
                        as_contiguous_slice(cols) or cols,
                        len(rows),
                        len(cols),
                        *(np.array(a, dtype=np.intp) for a in direct),
                        *(np.array(a, dtype=np.intp) for a in reversed_),
                    )
                )

        if pair_to_slots:
            # Report antenna indices, not the z rows they were mapped onto.
            ant_of = (
                np.arange(self.nant)
                if self.antenna_order is None
                else self.antenna_order
            )
            missing = [(int(ant_of[i]), int(ant_of[j])) for i, j in pair_to_slots]
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
