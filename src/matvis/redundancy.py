"""Helpers for building ``antenna_blocks`` lists for the block-decomposed matprod classes.

``matvis`` itself stays agnostic about *which* antenna groupings are good for a given
array -- that is a domain/array-geometry decision for the caller to make, not something
this library tries to derive automatically. This module only provides small, generic,
independently-testable building blocks for constructing the ``(row_antenna_idx,
col_antenna_idx)`` block lists that the block-decomposed matprod classes consume:

- :func:`blocks_from_groups` turns caller-supplied antenna group labels (e.g. "these
  350 antennas are the compact core, these 8 are outriggers") into the full grid of
  group-by-group blocks. This is the actual performance-relevant path: grouping
  antennas by known array structure into a handful of labels turns most of the
  ``Nant**2`` product into a few large, dense sub-blocks.
- :func:`radial_groups` is a ready-to-use way to get those labels without any
  array-specific knowledge: it bins antennas into concentric shells by distance
  from the array center, which works reasonably well for arrays with a compact
  core plus a small number of more remote antennas (e.g. HERA's own
  core-plus-outriggers layout).
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


def radial_groups(
    antpos: np.ndarray, n_groups: int, center: np.ndarray | None = None
) -> np.ndarray:
    """Group antennas into concentric shells by distance from a center point.

    A simple, geometry-aware default for arrays with a compact core plus a small
    number of more remote antennas (e.g. HERA's own core-plus-outriggers layout):
    antennas at similar distance from the array center land in the same group, so
    feeding the result to :func:`blocks_from_groups` makes most of the short,
    within-core baselines share one (or a few) group-diagonal blocks, while
    antennas that are genuinely far from the core end up in smaller, more
    numerous groups of their own.

    This is a reasonable default, not a guarantee of optimality for any
    particular array -- a caller with more specific domain knowledge (e.g. an
    exact core/outrigger split from their array-layout generator) should prefer
    that instead.

    Parameters
    ----------
    antpos
        Antenna positions, shape ``(Nant, Ndim)``. Only relative distances
        matter, so any consistent units/dimensionality work.
    n_groups
        Number of radial groups to form. Antennas are split into equal-population
        (not equal-width) bins by distance, so each group contains roughly
        ``Nant / n_groups`` antennas.
    center
        Reference point to measure distance from, shape ``(Ndim,)``. Defaults to
        the antennas' centroid.

    Returns
    -------
    np.ndarray
        Length-``Nant`` integer array of group labels in ``range(n_groups)``,
        ordered from closest to the center (label 0) to farthest
        (label ``n_groups - 1``), suitable as the ``labels`` argument to
        :func:`blocks_from_groups`.
    """
    antpos = np.asarray(antpos)
    nant = antpos.shape[0]
    if not 0 < n_groups <= nant:
        raise ValueError(
            f"n_groups must satisfy 0 < n_groups <= nant ({nant}), got {n_groups}"
        )

    center = antpos.mean(axis=0) if center is None else np.asarray(center)
    radius = np.linalg.norm(antpos - center, axis=-1)

    order = np.argsort(radius)
    labels = np.empty(nant, dtype=int)
    for label, idx in enumerate(np.array_split(order, n_groups)):
        labels[idx] = label
    return labels


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
