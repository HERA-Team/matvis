"""Helpers for building ``antenna_blocks`` lists for the block-decomposed matprod classes.

``matvis`` itself stays agnostic about *which* antenna groupings are good for a given
array -- that is a domain/array-geometry decision for the caller to make, not something
this library tries to derive automatically. This module only provides small, generic,
independently-testable building blocks for constructing the ``(row_antenna_idx,
col_antenna_idx)`` block lists that the block-decomposed matprod classes consume:

- :func:`find_dense_blocks` is the one to reach for on a redundant array, and the
  reason the block mechanism exists: given the unique-baseline pairs, it permutes
  the antenna axes to concentrate them and cuts the result into a few dense
  sub-matrices, minimizing the total area (and hence FLOPs) of the product. On a
  320-antenna hex layout, four blocks cut the product to ~1/38th of the full
  ``Nant**2`` area while still issuing only four GEMMs, which measures as a 2.3x
  end-to-end speedup (the FLOP saving is not fully collectible -- see the docs
  Performance page).
- :func:`blocks_from_groups` turns caller-supplied antenna group labels (e.g. "these
  350 antennas are the compact core, these 8 are outriggers") into the full grid of
  group-by-group blocks, for when you already know a grouping you want to impose
  rather than having one found for you.
- :func:`tile_antennas` is a generic, redundancy-agnostic full-array tiling, useful
  when a caller wants every pair computed but with bounded per-block memory.
- :func:`antpairs_to_blocks` is the trivial 1-antenna x 1-antenna block per requested
  pair. It is functionally equivalent to ``CPUVectorDot``/``GPUVectorDot``'s existing
  per-pair loop (same FLOP count, same per-pair BLAS-call granularity), so it exists
  for API completeness and as a reference/fixture, not as a performance win in its
  own right.
- :func:`find_redundant_antpairs` finds one representative antenna pair per unique
  (rounded) baseline vector.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def antpairs_to_blocks(
    antpairs: np.ndarray | Sequence[tuple[int, int]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build one singleton (1-antenna x 1-antenna) block per requested pair.

    Parameters
    ----------
    antpairs
        Array or sequence of ``(ant1, ant2)`` integer pairs.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        One ``(array([ant1]), array([ant2]))`` tuple per input pair, suitable as the
        ``antenna_blocks`` argument to :class:`~matvis.cpu.matprod.CPUMatBlock` /
        :class:`~matvis.gpu.matprod.GPUMatBlock`.
    """
    return [(np.array([i]), np.array([j])) for i, j in antpairs]


def blocks_from_groups(
    labels: np.ndarray | Sequence[int],
    col_labels: np.ndarray | Sequence[int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the full grid of group-by-group blocks from antenna group labels.

    Parameters
    ----------
    labels
        Length-``Nant`` array of group labels (any hashable value, typically small
        integers); antenna ``i`` belongs to group ``labels[i]``.
    col_labels
        Same convention as ``labels``, used for the "column" side of each block.
        Defaults to ``labels`` (i.e. produces every unique-label x unique-label
        combination over the same grouping).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        One block per ``(row group, column group)`` combination -- i.e.
        ``len(unique(labels)) * len(unique(col_labels))`` blocks in total. Their
        combined coverage is every ``(i, j)`` pair for ``i`` in ``labels``' index
        range and ``j`` in ``col_labels``' index range.
    """
    labels = np.asarray(labels)
    col_labels = labels if col_labels is None else np.asarray(col_labels)
    row_ids = np.arange(labels.size)
    col_ids = np.arange(col_labels.size)

    blocks = []
    for lbl in np.unique(labels):
        rows = row_ids[labels == lbl]
        for clbl in np.unique(col_labels):
            cols = col_ids[col_labels == clbl]
            blocks.append((rows, cols))
    return blocks


def _partition_rows(
    antpairs: np.ndarray, max_blocks: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition the row antennas into runs, minimizing total sub-matrix area.

    Rows are first sorted by degree (how many requested pairs each row antenna
    appears in), which tends to put rows with similar, large partner sets next to
    each other so that a contiguous run of them has a small column union. The
    partition of that ordering into at most ``max_blocks`` contiguous runs is then
    chosen *optimally* by dynamic programming over the exact cost
    ``sum(len(rows) * len(cols))``, rather than by a density-threshold heuristic.
    """
    partners: dict[int, set[int]] = defaultdict(set)
    for i, j in antpairs:
        partners[int(i)].add(int(j))

    rows_sorted = sorted(partners, key=lambda a: (-len(partners[a]), a))
    n = len(rows_sorted)

    # Column sets as integer bitmasks, so unions are single integer ORs and
    # sizes are popcounts -- this keeps the O(n^2) union scan cheap.
    all_cols = sorted({c for s in partners.values() for c in s})
    col_bit = {c: k for k, c in enumerate(all_cols)}
    masks = [sum(1 << col_bit[c] for c in partners[a]) for a in rows_sorted]

    inf = float("inf")
    nblocks = min(max_blocks, n)
    # dp[j][i]: least area covering the first i rows with at most j blocks.
    dp = [[inf] * (n + 1) for _ in range(nblocks + 1)]
    choice = [[0] * (n + 1) for _ in range(nblocks + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        # areas[m] = cost of a single block spanning rows_sorted[m:i]
        areas = [0] * i
        union = 0
        for m in range(i - 1, -1, -1):
            union |= masks[m]
            areas[m] = (i - m) * union.bit_count()

        for j in range(1, nblocks + 1):
            best, arg = inf, 0
            for m in range(i):
                prev = dp[j - 1][m]
                if prev == inf:
                    continue
                cand = prev + areas[m]
                if cand < best:
                    best, arg = cand, m
            dp[j][i] = best
            choice[j][i] = arg

    # Fewest blocks that achieve the best area (extra blocks cost extra GEMM
    # calls, so don't take one that doesn't pay for itself).
    best_j = min(range(1, nblocks + 1), key=lambda j: (dp[j][n], j))

    blocks = []
    i, j = n, best_j
    while i > 0:
        m = choice[j][i]
        rows = rows_sorted[m:i]
        cols = set().union(*(partners[a] for a in rows))
        blocks.append(
            (np.array(sorted(rows), dtype=int), np.array(sorted(cols), dtype=int))
        )
        i, j = m, j - 1
    return blocks[::-1]


def find_dense_blocks(
    antpairs: np.ndarray | Sequence[tuple[int, int]],
    max_blocks: int = 4,
    allow_conjugates: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decompose a set of antenna pairs into a few dense rectangular blocks.

    This is the helper that makes the block-decomposed matprod pay off for a
    redundant array. Given the *unique-baseline* antenna pairs of such an array
    (see :func:`find_redundant_antpairs`), the wanted pairs occupy a small,
    scattered fraction of the full ``Nant x Nant`` grid. Permuting the antenna
    axes concentrates them, and this function then cuts the permuted grid into at
    most ``max_blocks`` rectangular sub-matrices chosen to minimize the total
    sub-matrix area ``sum(len(rows) * len(cols))`` -- the quantity the matrix
    product's FLOP count is proportional to.

    The result sits between the two existing extremes: far fewer FLOPs than the
    full ``Nant x Nant`` product (``CPUMatMul``/``GPUMatMul``), but a handful of
    large GEMMs rather than one tiny GEMM per baseline
    (``CPUVectorDot``/``GPUVectorDot``).

    Two heuristics are used, matching the approach described in Appendix A of the
    ``matvis`` paper: rows are ordered by how many pairs they appear in, and cuts
    are only made along the row axis. Given that ordering, the partition itself is
    optimal (dynamic programming over the exact area). Finding the globally best
    antenna permutation is a much harder combinatorial problem and is not
    attempted; nor is choosing a different representative pair out of each
    redundant group, which would give further freedom.

    Parameters
    ----------
    antpairs
        The antenna pairs to cover, shape ``(Npairs, 2)``. For a redundant array
        this should be the deduplicated set (one pair per redundant group).
    max_blocks
        Maximum number of sub-matrices. More blocks means fewer wasted FLOPs but
        more (smaller) GEMM calls, so the best value is hardware- and
        array-dependent; 3-4 is a reasonable starting point.
    allow_conjugates
        If True (default), also consider orientations in which some or all pairs
        are represented reversed, which often packs the pairs more tightly.
        ``V_ij`` is the Hermitian conjugate of ``V_ji``, so the block-decomposed
        matprod classes compute a reversed pair exactly; set this to False only
        if you need every block to hold pairs in the exact orientation requested.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Blocks suitable as the ``antenna_blocks`` argument to
        :class:`~matvis.cpu.matprod.CPUMatBlock` /
        :class:`~matvis.gpu.matprod.GPUMatBlock`.

    Notes
    -----
    This only pays off when ``antpairs`` is much smaller than ``Nant**2``, i.e. on
    a genuinely redundant array. It cannot create redundancy that isn't there: if
    every antenna pair is wanted (for example because every antenna has its own
    beam), the best decomposition is the full product itself, and using it is
    measurably slower than ``CPUMatMul``/``GPUMatMul``. The realized speedup is
    also well below the area ratio, and the best ``max_blocks`` is not the one
    that minimizes the area -- see the docs Performance page for measurements.
    """
    antpairs = np.asarray(antpairs)
    if max_blocks < 1:
        raise ValueError(f"max_blocks must be at least 1, got {max_blocks}")
    if antpairs.size == 0:
        return []

    orientations = [antpairs]
    if allow_conjugates:
        lo = np.minimum(antpairs[:, 0], antpairs[:, 1])
        hi = np.maximum(antpairs[:, 0], antpairs[:, 1])
        orientations += [
            antpairs[:, ::-1],
            np.stack([lo, hi], axis=1),
            np.stack([hi, lo], axis=1),
        ]

    best, best_area = None, float("inf")
    for pairs in orientations:
        blocks = _partition_rows(pairs, max_blocks)
        area = sum(len(rows) * len(cols) for rows, cols in blocks)
        if area < best_area:
            best, best_area = blocks, area
    return best


def tile_antennas(nant: int, chunk_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tile the full ``Nant x Nant`` grid into ``chunk_size``-ish square blocks.

    Generic and redundancy-agnostic: covers every possible antenna pair, just with
    bounded per-block memory, unlike a single block spanning all antennas.

    Parameters
    ----------
    nant
        Number of antennas.
    chunk_size
        Maximum number of antennas per block edge.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        ``ceil(nant / chunk_size) ** 2`` blocks whose combined coverage is every
        ``(i, j)`` pair for ``i, j`` in ``range(nant)``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    edges = list(range(0, nant, chunk_size)) + [nant]
    tiles = [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    return [(rows, cols) for rows in tiles for cols in tiles]


def find_redundant_antpairs(
    bl_vectors: np.ndarray,
    antpairs: Sequence[tuple[int, int]] | None = None,
    ndecimals: int = 2,
) -> list[tuple[int, int]]:
    """Find one representative antenna pair per unique (rounded) baseline vector.

    Parameters
    ----------
    bl_vectors
        Array of shape ``(Nant, Nant, 2)`` giving the projected ``(u, v)`` baseline
        vector from antenna ``i`` to antenna ``j`` at ``bl_vectors[i, j]``, in
        whatever units the desired redundancy tolerance is meaningful in (e.g.
        wavelengths). A three-component ``(u, v, w)`` array is not supported --
        project to 2D first.
    antpairs
        If given, restrict the search to just these ``(i, j)`` pairs (with ``i``
        used to index the first axis of ``bl_vectors`` and ``j`` the second),
        instead of scanning every ``i < j`` combination. Pairs with ``i == j``
        (autocorrelations) are ignored.
    ndecimals
        Number of decimal places to round baseline vectors to before grouping;
        larger values require closer agreement to be considered redundant.

    Returns
    -------
    list[tuple[int, int]]
        One representative ``(i, j)`` pair per unique baseline-vector bin (a
        vector and its negation are treated as the same bin).
    """
    if bl_vectors.shape[-1] != 2:
        raise ValueError(
            "bl_vectors' last dimension must be 2 (a projected (u, v) vector); "
            f"got shape {bl_vectors.shape}. Project a 3-component (u, v, w) "
            "baseline down to 2D first."
        )

    bl_vectors = np.round(bl_vectors, decimals=ndecimals)
    nant = bl_vectors.shape[0]

    if antpairs is None:
        antpairs = [(i, j) for i in range(nant) for j in range(i + 1, nant)]

    uvbins = set()
    pairs = []
    for i, j in antpairs:
        if i == j:
            continue
        u, v = bl_vectors[i, j]
        if (u, v) not in uvbins and (-u, -v) not in uvbins:
            uvbins.add((u, v))
            pairs.append((int(i), int(j)))

    return pairs
