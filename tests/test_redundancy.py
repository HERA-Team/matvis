"""Tests for ``matvis.redundancy``: helpers for building antenna-block lists.

matvis itself stays agnostic about *which* antenna groupings are good (that's a
domain/array-geometry decision -- see the accuracy/feasibility discussion around
``matvis#135``). This module only provides small, generic, independently-testable
building blocks for constructing the ``antenna_blocks`` argument to
``CPUMatBlock``/``GPUMatBlock``:

- ``antpairs_to_blocks``: the trivial 1-antenna x 1-antenna block per pair, a direct
  (bugfixed, renamed) replacement for PR#79's ``get_matrix_sets``. Functionally
  equivalent to ``CPUVectorDot``/``GPUVectorDot``'s existing per-pair loop -- included
  for API completeness, not as a performance win.
- ``blocks_from_groups``: turns caller-supplied antenna group labels (domain
  knowledge matvis doesn't try to derive itself) into the full grid of
  group-by-group blocks. This is the actual payoff path: grouping antennas by
  known array structure (e.g. a compact core vs outriggers) into a handful of
  labels turns most of the ``Nant^2`` product into a few large, dense sub-blocks.
- ``tile_antennas``: a generic, redundancy-agnostic full-array tiling, useful when a
  caller wants every pair but with bounded per-block memory.
- ``find_redundant_antpairs``: refactored out of ``cli.py::get_redundancies`` so it's
  independently testable and reusable outside the CLI.
"""

from __future__ import annotations

import numpy as np
import pytest

from matvis._utils import get_dtypes
from matvis.redundancy import (
    antpairs_to_blocks,
    blocks_from_groups,
    find_redundant_antpairs,
    tile_antennas,
)


def _assert_valid_blocks(blocks, nant):
    for rows, cols in blocks:
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        assert rows.ndim == 1 and cols.ndim == 1
        assert np.issubdtype(rows.dtype, np.integer)
        assert np.issubdtype(cols.dtype, np.integer)
        if rows.size:
            assert rows.min() >= 0 and rows.max() < nant
        if cols.size:
            assert cols.min() >= 0 and cols.max() < nant


def _covered_pairs(blocks):
    covered = set()
    for rows, cols in blocks:
        for i in rows:
            for j in cols:
                covered.add((int(i), int(j)))
    return covered


def _make_z(nant, nfeed, nsrc, precision, seed=0):
    ctype = get_dtypes(precision)[1]
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((nfeed * nant, nsrc))
        + 1j * rng.standard_normal((nfeed * nant, nsrc))
    ).astype(ctype)


def _reference_vis(z, nant, nfeed, antpairs):
    v = z.conj().dot(z.T)
    v = v.reshape((nant, nfeed, nant, nfeed)).transpose((0, 2, 3, 1))
    ant1, ant2 = antpairs[:, 0], antpairs[:, 1]
    return v[ant1, ant2]


class TestAntpairsToBlocks:
    """Tests for ``antpairs_to_blocks``."""

    def test_produces_one_singleton_block_per_pair(self):
        """Each input pair becomes its own 1x1 (row, col) block, in order."""
        antpairs = np.array([(0, 1), (2, 3), (1, 3)])
        blocks = antpairs_to_blocks(antpairs)
        assert len(blocks) == len(antpairs)
        for (rows, cols), (i, j) in zip(blocks, antpairs, strict=True):
            assert list(rows) == [i]
            assert list(cols) == [j]

    def test_covers_every_requested_pair(self):
        """The produced blocks' combined coverage includes every requested pair."""
        antpairs = np.array([(i, j) for i in range(5) for j in range(5) if i != j])
        blocks = antpairs_to_blocks(antpairs)
        covered = _covered_pairs(blocks)
        for i, j in antpairs:
            assert (i, j) in covered

    def test_empty_antpairs_gives_empty_blocks(self):
        """An empty antpairs array produces an empty block list, not an error."""
        assert antpairs_to_blocks(np.zeros((0, 2), dtype=int)) == []


class TestBlocksFromGroups:
    """Tests for ``blocks_from_groups``."""

    def test_single_label_produces_one_block_covering_everything(self):
        """All antennas sharing one label collapse to a single full block."""
        nant = 5
        labels = np.zeros(nant, dtype=int)
        blocks = blocks_from_groups(labels)
        assert len(blocks) == 1
        rows, cols = blocks[0]
        assert sorted(rows) == list(range(nant))
        assert sorted(cols) == list(range(nant))

    def test_distinct_labels_produce_full_grid_covering_all_pairs(self):
        """Two groups produce a 2x2 grid of blocks covering every possible pair."""
        # 5 antennas, 2 groups of size 2 and 3 -> a 2x2 grid of blocks whose
        # combined coverage is every one of the 25 possible pairs.
        labels = np.array([0, 0, 1, 1, 1])
        blocks = blocks_from_groups(labels)
        assert len(blocks) == 4
        _assert_valid_blocks(blocks, nant=5)
        covered = _covered_pairs(blocks)
        assert covered == {(i, j) for i in range(5) for j in range(5)}

    def test_col_labels_defaults_to_row_labels(self):
        """Omitting col_labels is identical to passing the same labels explicitly."""
        labels = np.array([0, 0, 1, 1, 1])
        a = blocks_from_groups(labels)
        b = blocks_from_groups(labels, col_labels=labels)
        assert len(a) == len(b)
        for (r1, c1), (r2, c2) in zip(a, b, strict=True):
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(c1, c2)

    def test_different_row_and_col_labels(self):
        """Distinct row/col labelings produce the cross-product grid of blocks."""
        row_labels = np.array([0, 0, 1])
        col_labels = np.array([0, 1, 1])
        blocks = blocks_from_groups(row_labels, col_labels=col_labels)
        assert len(blocks) == 4  # 2 unique row labels x 2 unique col labels
        covered = _covered_pairs(blocks)
        assert covered == {(i, j) for i in range(3) for j in range(3)}

    def test_composes_with_matblock_and_matches_full_matmul(self):
        """blocks_from_groups + CPUMatBlock reproduces CPUMatMul for all pairs."""
        from matvis.cpu.matprod import CPUMatBlock

        nant, nfeed, nsrc, precision = 8, 2, 12, 1
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        blocks = blocks_from_groups(labels)
        antpairs = np.array([(i, j) for i in range(nant) for j in range(nant)])
        z = _make_z(nant, nfeed, nsrc, precision, seed=1)

        obj = CPUMatBlock(
            nchunks=1,
            nfeed=nfeed,
            nant=nant,
            antpairs=antpairs,
            precision=precision,
            antenna_blocks=blocks,
        )
        obj.setup()
        out = np.zeros((obj.npairs, nfeed, nfeed), dtype=z.dtype)
        obj(z, chunk=0)
        obj.sum_chunks(out)

        expected = _reference_vis(z, nant, nfeed, antpairs)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)


class TestTileAntennas:
    """Tests for ``tile_antennas``."""

    @pytest.mark.parametrize("nant,chunk_size", [(10, 3), (9, 3), (5, 100), (1, 1)])
    def test_covers_all_pairs_exactly_via_full_tiling(self, nant, chunk_size):
        """Tiling the full Nant x Nant grid must cover every possible pair."""
        blocks = tile_antennas(nant, chunk_size)
        _assert_valid_blocks(blocks, nant)
        covered = _covered_pairs(blocks)
        expected = {(i, j) for i in range(nant) for j in range(nant)}
        assert covered == expected

    def test_blocks_are_bounded_by_chunk_size(self):
        """No block's row or column count exceeds the requested chunk_size."""
        nant, chunk_size = 10, 4
        blocks = tile_antennas(nant, chunk_size)
        for rows, cols in blocks:
            assert len(rows) <= chunk_size
            assert len(cols) <= chunk_size

    def test_number_of_blocks_scales_as_expected(self):
        """Roughly (nant/chunk_size)^2 blocks, not O(nant^2) individual pairs."""
        nant, chunk_size = 12, 4
        blocks = tile_antennas(nant, chunk_size)
        n_tiles_per_side = -(-nant // chunk_size)  # ceil division
        assert len(blocks) == n_tiles_per_side**2

    def test_invalid_chunk_size_raises(self):
        """A non-positive chunk_size is rejected rather than producing garbage."""
        with pytest.raises(ValueError):
            tile_antennas(10, 0)
        with pytest.raises(ValueError):
            tile_antennas(10, -1)

    def test_tiling_composes_with_matblock_and_bounds_memory(self):
        """tile_antennas + CPUMatBlock reproduces CPUMatMul for the full pair set.

        This is the concrete "bounded peak memory for a full-pair product" use case
        this helper exists for.
        """
        from matvis.cpu.matprod import CPUMatBlock

        nant, nfeed, nsrc, precision = 11, 1, 9, 2
        blocks = tile_antennas(nant, chunk_size=4)
        antpairs = np.array([(i, j) for i in range(nant) for j in range(nant)])
        z = _make_z(nant, nfeed, nsrc, precision, seed=2)

        obj = CPUMatBlock(
            nchunks=1,
            nfeed=nfeed,
            nant=nant,
            antpairs=antpairs,
            precision=precision,
            antenna_blocks=blocks,
        )
        obj.setup()
        out = np.zeros((obj.npairs, nfeed, nfeed), dtype=z.dtype)
        obj(z, chunk=0)
        obj.sum_chunks(out)

        expected = _reference_vis(z, nant, nfeed, antpairs)
        np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


class TestFindRedundantAntpairs:
    """Tests for ``find_redundant_antpairs``."""

    def test_finds_exact_duplicate_baselines(self):
        """Two pairs sharing a baseline vector collapse to one representative."""
        # A 1D linear array with two pairs sharing the same separation vector:
        # (0,1) and (1,2) both have baseline vector (1, 0). (0,2) has (2, 0), unique.
        bl = np.zeros((3, 3, 2))
        pos = np.array([0.0, 1.0, 2.0])
        for i in range(3):
            for j in range(3):
                bl[i, j] = (pos[j] - pos[i], 0.0)

        pairs = find_redundant_antpairs(bl)
        pair_set = {tuple(p) for p in pairs}
        assert len(pair_set) == 2
        # Exactly one of (0,1)/(1,2) should be kept as the representative, plus (0,2).
        assert len(pair_set & {(0, 1), (1, 2)}) == 1
        assert (0, 2) in pair_set

    def test_conjugate_baselines_are_not_double_counted(self):
        """A vector and its negation are treated as the same redundant bin."""
        bl = np.zeros((2, 2, 2))
        bl[0, 1] = (1.0, 0.0)
        bl[1, 0] = (-1.0, 0.0)
        pairs = find_redundant_antpairs(bl, antpairs=[(0, 1), (1, 0)])
        assert len(pairs) == 1

    def test_ndecimals_controls_grouping_tolerance(self):
        """A coarser ndecimals groups nearly-identical baselines together."""
        bl = np.zeros((3, 3, 2))
        bl[0, 1] = (1.000, 0.0)
        bl[0, 2] = (1.004, 0.0)  # differs at the 3rd decimal
        antpairs = [(0, 1), (0, 2)]

        coarse = find_redundant_antpairs(bl, antpairs=antpairs, ndecimals=2)
        fine = find_redundant_antpairs(bl, antpairs=antpairs, ndecimals=3)

        assert len(coarse) == 1  # grouped together at 2 decimals
        assert len(fine) == 2  # kept separate at 3 decimals

    def test_restricting_to_antpairs_only_considers_those(self):
        """Restricting the candidate pool changes which pairs survive as reps."""
        bl = np.zeros((4, 4, 2))
        for i in range(4):
            for j in range(4):
                bl[i, j] = (float(j - i), 0.0)

        # Unrestricted i<j scan: baseline-vector magnitudes 1 ((0,1),(1,2),(2,3)),
        # 2 ((0,2),(1,3)), and 3 ((0,3)) -> three unique bins -> three
        # representative pairs.
        all_pairs = find_redundant_antpairs(bl)
        assert len(all_pairs) == 3

        # Restricting the candidate pool to (0,1) and (1,2) -- both magnitude 1
        # -- means only one of them is kept, even though the unrestricted scan
        # would have kept a representative for that bin regardless.
        restricted = find_redundant_antpairs(bl, antpairs=[(0, 1), (1, 2)])
        assert len(restricted) == 1

        # Restricting to two genuinely different vectors keeps both.
        restricted2 = find_redundant_antpairs(bl, antpairs=[(0, 1), (0, 2)])
        assert len(restricted2) == 2

    def test_autocorrelations_are_excluded(self):
        """A requested (i, i) "pair" is silently ignored, not treated as a baseline."""
        bl = np.zeros((3, 3, 2))
        pairs = find_redundant_antpairs(bl, antpairs=[(0, 0), (0, 1)])
        assert (0, 0) not in pairs

    def test_rejects_baseline_vectors_with_wrong_last_dimension(self):
        """A 3-component (u, v, w) input is rejected with a clear error."""
        bl = np.zeros((3, 3, 3))  # (u, v, w) -- w not supported, must be explicit
        with pytest.raises(ValueError, match="last dimension"):
            find_redundant_antpairs(bl)

    def test_return_type_is_list_of_int_tuples(self):
        """The output contract (list of int 2-tuples) is pinned.

        It feeds straight into ``simulate(antpairs=...)``.
        """
        bl = np.zeros((3, 3, 2))
        bl[0, 1] = (1.0, 0.0)
        bl[0, 2] = (2.0, 0.0)
        bl[1, 2] = (1.0, 0.0)
        pairs = find_redundant_antpairs(bl)
        assert isinstance(pairs, list)
        for p in pairs:
            assert isinstance(p, tuple) and len(p) == 2
            assert all(isinstance(x, (int, np.integer)) for x in p)

    def test_matches_cli_get_redundancies(self):
        """The already-shipped CLI redundancy finder keeps identical output.

        It's unrelated to #79 itself, but must keep producing identical output once
        refactored to delegate to this function.
        """
        from matvis.cli import get_redundancies

        rng = np.random.default_rng(3)
        nant = 6
        pos = rng.standard_normal(nant)
        bl = pos[np.newaxis, :] - pos[:, np.newaxis]
        bl2 = np.stack([bl, np.zeros_like(bl)], axis=-1)

        old = [tuple(p) for p in get_redundancies(bl2)]
        new = [tuple(p) for p in find_redundant_antpairs(bl2)]
        assert old == new

    def test_output_is_usable_as_antpairs_for_blocks(self):
        """The representative-pair output feeds directly into antpairs_to_blocks."""
        bl = np.zeros((5, 5, 2))
        rng = np.random.default_rng(0)
        pos = rng.standard_normal(5)
        for i in range(5):
            for j in range(5):
                bl[i, j] = (pos[j] - pos[i], 0.0)

        pairs = find_redundant_antpairs(bl)
        blocks = antpairs_to_blocks(np.array(pairs))
        _assert_valid_blocks(blocks, nant=5)
