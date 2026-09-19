"""Tests for ``matvis.redundancy``: helpers for building antenna-block lists."""

import numpy as np
import pytest

from matvis._utils import get_dtypes
from matvis.redundancy import (
    antpairs_to_blocks,
    blocks_from_groups,
    contiguity_order,
    find_dense_blocks,
    find_redundant_antpairs,
    tile_antennas,
)


def _block_area(blocks):
    """Total sub-matrix area, the FLOP proxy a decomposition tries to minimize."""
    return sum(len(rows) * len(cols) for rows, cols in blocks)


def _covers(blocks, antpairs, allow_conjugates=True):
    """Whether every requested pair is inside some block (optionally reversed)."""
    covered = _covered_pairs(blocks)
    for i, j in antpairs:
        if (int(i), int(j)) in covered:
            continue
        if allow_conjugates and (int(j), int(i)) in covered:
            continue
        return False
    return True


def _hex_antpos(hex_num=4):
    """A HERA-like hex-packed layout (the canonical highly-redundant array)."""
    pos = []
    for row in range(-hex_num + 1, hex_num):
        n_in_row = 2 * hex_num - 1 - abs(row)
        for col in range(n_in_row):
            x = ((-n_in_row + 1) / 2 + col) * 14.6
            y = row * 14.6 * np.sqrt(3) / 2
            pos.append((x, y))
    return np.array(pos)


def _unique_baseline_antpairs(antpos):
    """The deduplicated (one per redundant group) antpairs for a layout."""
    bl = antpos[np.newaxis, :, :2] - antpos[:, np.newaxis, :2]
    return np.array(find_redundant_antpairs(bl))


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


class TestFindDenseBlocks:
    """Tests for ``find_dense_blocks``, the redundancy-aware decomposition.

    This is the helper that makes the block mechanism actually pay off: given the
    (deduplicated) unique-baseline antpairs of a redundant array, it permutes the
    antenna axes and partitions them into a handful of sub-matrices chosen to be
    as *dense* in wanted pairs as possible, so the total sub-matrix area (the FLOP
    proxy) is far below Nant^2 while still being a few big GEMMs rather than one
    tiny GEMM per pair.
    """

    def test_covers_every_requested_pair(self):
        """The decomposition must cover every requested pair (possibly reversed)."""
        antpairs = _unique_baseline_antpairs(_hex_antpos(3))
        blocks = find_dense_blocks(antpairs, max_blocks=4)
        _assert_valid_blocks(blocks, nant=len(_hex_antpos(3)))
        assert _covers(blocks, antpairs)

    def test_beats_the_full_matrix_on_a_redundant_array(self):
        """On a hex array the decomposition must be far cheaper than the full product.

        This is the whole point of the feature: a highly redundant array has far
        fewer unique baselines than antenna pairs, and those unique pairs can be
        packed into a few dense sub-matrices.
        """
        antpos = _hex_antpos(4)
        nant = len(antpos)
        antpairs = _unique_baseline_antpairs(antpos)

        blocks = find_dense_blocks(antpairs, max_blocks=4)
        assert _covers(blocks, antpairs)

        full_area = nant * nant
        assert _block_area(blocks) < 0.5 * full_area, (
            f"decomposition area {_block_area(blocks)} is not much better than the "
            f"full product ({full_area}) for a {nant}-antenna hex array"
        )

    def test_more_blocks_never_increases_area(self):
        """Allowing more sub-matrices can only reduce (or tie) the total area."""
        antpairs = _unique_baseline_antpairs(_hex_antpos(3))
        areas = [
            _block_area(find_dense_blocks(antpairs, max_blocks=k)) for k in range(1, 5)
        ]
        assert areas == sorted(areas, reverse=True) or all(
            b <= a for a, b in zip(areas, areas[1:], strict=True)
        )

    def test_never_returns_more_blocks_than_allowed(self):
        """max_blocks is a hard cap on the number of sub-matrices."""
        antpairs = _unique_baseline_antpairs(_hex_antpos(3))
        for k in range(1, 6):
            assert len(find_dense_blocks(antpairs, max_blocks=k)) <= k

    def test_single_block_is_the_bounding_box(self):
        """With max_blocks=1 the answer is just the bounding box of the pairs."""
        antpairs = np.array([(0, 3), (1, 4), (2, 5)])
        blocks = find_dense_blocks(antpairs, max_blocks=1, allow_conjugates=False)
        assert len(blocks) == 1
        rows, cols = blocks[0]
        assert sorted(rows) == [0, 1, 2]
        assert sorted(cols) == [3, 4, 5]

    def test_finds_the_exact_block_diagonal_structure(self):
        """Two disjoint antenna clusters must come out as two separate blocks.

        A decomposition that missed this would return one big block covering both
        clusters plus all the (never-requested) cross-cluster pairs.
        """
        antpairs = np.array(
            [(i, j) for i in (0, 1, 2) for j in (0, 1, 2)]
            + [(i, j) for i in (3, 4, 5) for j in (3, 4, 5)]
        )
        blocks = find_dense_blocks(antpairs, max_blocks=2, allow_conjugates=False)
        assert len(blocks) == 2
        assert _block_area(blocks) == 18  # 3x3 + 3x3, i.e. zero waste
        assert _covers(blocks, antpairs, allow_conjugates=False)

    def test_allow_conjugates_can_only_help(self):
        """Permitting reversed pairs must never produce a worse decomposition."""
        antpairs = _unique_baseline_antpairs(_hex_antpos(3))
        with_conj = _block_area(find_dense_blocks(antpairs, max_blocks=3))
        without = _block_area(
            find_dense_blocks(antpairs, max_blocks=3, allow_conjugates=False)
        )
        assert with_conj <= without

    def test_empty_antpairs_gives_no_blocks(self):
        """No requested pairs means nothing to compute."""
        assert find_dense_blocks(np.zeros((0, 2), dtype=int)) == []

    def test_invalid_max_blocks_raises(self):
        """max_blocks must be a positive number of sub-matrices."""
        antpairs = np.array([(0, 1)])
        with pytest.raises(ValueError):
            find_dense_blocks(antpairs, max_blocks=0)

    @pytest.mark.parametrize("allow_conjugates", [True, False])
    @pytest.mark.parametrize("max_blocks", [1, 2, 3])
    def test_composes_with_matblock_on_a_redundant_array(
        self, allow_conjugates, max_blocks
    ):
        """End-to-end: the decomposition must give exactly the right visibilities.

        Covers the conjugate path too -- with ``allow_conjugates=True`` some pairs
        may only be present in their reversed orientation, so this also checks
        ``CPUMatBlock``'s Hermitian handling against a brute-force reference.
        """
        from matvis.cpu.matprod import CPUMatBlock

        antpos = _hex_antpos(3)
        nant, nfeed, nsrc, precision = len(antpos), 2, 14, 2
        antpairs = _unique_baseline_antpairs(antpos)
        blocks = find_dense_blocks(
            antpairs, max_blocks=max_blocks, allow_conjugates=allow_conjugates
        )
        z = _make_z(nant, nfeed, nsrc, precision, seed=8)

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


# ---------------------------------------------------------------------------
# contiguity_order
# ---------------------------------------------------------------------------


def _n_runs(idx):
    """Number of maximal consecutive runs in a set of antenna indices."""
    idx = np.sort(np.asarray(idx))
    return 1 + int(np.sum(np.diff(idx) != 1))


def _rows_needing_a_gather(blocks, order, nant):
    """Antenna-rows that would still have to be staged into a copy under `order`."""
    inv = np.empty(nant, dtype=int)
    inv[order] = np.arange(nant)
    return sum(
        len(side)
        for blk in blocks
        for side in blk
        if _n_runs(inv[np.asarray(side)]) > 1
    )


def test_contiguity_order_is_a_permutation():
    """Whatever it decides, the result must be a valid relabelling of the antennas."""
    nant = 12
    blocks = [
        (np.arange(0, nant, 3), np.arange(1, nant, 2)),
        (np.array([7]), np.arange(4)),
    ]
    order = contiguity_order(blocks, nant)
    assert order.shape == (nant,)
    np.testing.assert_array_equal(np.sort(order), np.arange(nant))


def test_contiguity_order_makes_interleaved_blocks_contiguous():
    """Evens x odds is the worst case for slicing, and is fully fixable."""
    nant = 10
    ev, od = np.arange(0, nant, 2), np.arange(1, nant, 2)
    blocks = [(ev, ev), (ev, od), (od, ev), (od, od)]

    assert _rows_needing_a_gather(blocks, np.arange(nant), nant) == 4 * nant
    assert _rows_needing_a_gather(blocks, contiguity_order(blocks, nant), nant) == 0


def test_contiguity_order_never_makes_things_worse():
    """It must not break sets that were already contiguous in the natural order."""
    nant = 16
    blocks = tile_antennas(nant, chunk_size=4)
    order = contiguity_order(blocks, nant)
    assert _rows_needing_a_gather(blocks, order, nant) == 0


def test_contiguity_order_on_a_partially_satisfiable_set():
    """Overlapping blocks can't all be runs at once; the big ones must win.

    ``{0..5}`` and ``{4..9}`` overlap, so a third set interleaved with both can't
    be made contiguous as well. The greedy takes sets largest-first, so it is the
    small one that is left to be gathered.
    """
    nant = 10
    big_a, big_b = np.arange(6), np.arange(4, 10)
    small = np.array([0, 9])
    blocks = [(big_a, big_a), (big_b, big_b), (small, small)]

    order = contiguity_order(blocks, nant)
    inv = np.empty(nant, dtype=int)
    inv[order] = np.arange(nant)

    assert _n_runs(inv[big_a]) == 1
    assert _n_runs(inv[big_b]) == 1


def test_contiguity_order_on_a_real_decomposition():
    """On a redundant array it should remove most of the gather, not a little of it.

    The antenna *labels* are shuffled relative to the geometry, as they generally
    are in a real array, so the blocks ``find_dense_blocks`` picks are scattered
    over the antenna axis and slicing them needs a relabelling.
    """
    rng = np.random.default_rng(0)
    side = 6
    pos = np.array([(x, y) for x in range(side) for y in range(side)], dtype=float)
    nant = len(pos)
    pos = pos[rng.permutation(nant)]  # geometry-independent antenna numbering

    bls = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    antpairs = np.array(find_redundant_antpairs(bls))
    blocks = find_dense_blocks(antpairs, max_blocks=3)

    before = _rows_needing_a_gather(blocks, np.arange(nant), nant)
    after = _rows_needing_a_gather(blocks, contiguity_order(blocks, nant), nant)
    assert before > 0, "test array isn't scattered enough to be interesting"
    assert after < before / 2, f"gather only fell from {before} to {after} rows"


def test_contiguity_order_validates_inputs():
    """Out-of-range block indices are a hard error, not a mangled permutation."""
    with pytest.raises(ValueError, match=r"must be in \[0, 4\)"):
        contiguity_order([(np.array([0, 4]), np.array([1]))], 4)
    with pytest.raises(ValueError, match="nant must be non-negative"):
        contiguity_order([], -1)


def test_contiguity_order_with_no_blocks():
    """No blocks means nothing to satisfy; the natural order is a fine answer."""
    np.testing.assert_array_equal(contiguity_order([], 5), np.arange(5))
