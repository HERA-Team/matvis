"""Tests for the block-decomposition (``antenna_blocks``) matrix-product mechanism.

A caller can supply an explicit list of rectangular antenna-index blocks
(``antenna_blocks``); matvis then computes only those blocks (as sub-matrix
products) instead of the full ``Nant x Nant`` Gram matrix, gathering just the
requested ``antpairs`` out of them. It's a mechanism only -- matvis does not decide
*which* blocks are good; the caller (e.g. a redundancy-aware helper in
``matvis.redundancy``) does.

Named ``MatBlock`` (not ``MatChunk``): "chunk" already means the source-axis memory
chunking (``nchunks``, ``get_desired_chunks``, ``sum_chunks``, ``select_chunk``)
throughout this codebase, and reusing it for a completely different antenna-axis
concept would be an avoidable footgun.

Design properties exercised below:

- No dense ``(Nant, Nant, Nfeed, Nfeed)`` intermediate, in ``compute()`` *or* in the
  one-time ``setup()`` coverage validation: each block's wanted entries are gathered
  directly into the final ``(Npairs, Nfeed, Nfeed)`` output, and coverage is checked
  without ever materializing an ``(Nant, Nant)``-shaped structure.
- No per-call ``meshgrid``/Python bookkeeping: block coverage is resolved once in
  ``setup()``, not on every ``compute()`` call.
- Aware of ``nchunks`` (memory-based source-axis chunking) and of being called once per
  chunk *per time sample*, i.e. the same block-index precomputation must be reused
  correctly across repeated ``compute()`` calls on the same chunk index.
- ``antenna_blocks`` is keyword-only on ``MatProd.__init__``, so inserting it can't
  silently reinterpret an existing positional call site.
"""

import numpy as np
import pytest

from matvis._utils import get_dtypes

ALL_METHODS = ["CPUMatBlock", pytest.param("GPUMatBlock", marks=pytest.mark.gpu)]


def _get_cls(method):
    if method.startswith("GPU"):
        pytest.importorskip("cupy")
        from matvis.gpu import matprod as module
    else:
        from matvis.cpu import matprod as module
    return getattr(module, method)


def _to_backend(z, method):
    if method.startswith("GPU"):
        import cupy as cp

        return cp.asarray(z)
    return z


def _make_z(nant, nfeed, nsrc, precision, seed=0):
    ctype = get_dtypes(precision)[1]
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((nfeed * nant, nsrc))
        + 1j * rng.standard_normal((nfeed * nant, nsrc))
    ).astype(ctype)


def _reference_vis(z, nant, nfeed, antpairs):
    """Brute-force full Gram product, sliced to the requested antpairs.

    Mirrors the ``simple_matprod`` reference already used in ``test_matprod.py``.
    """
    v = z.conj().dot(z.T)
    v = v.reshape((nant, nfeed, nant, nfeed)).transpose((0, 2, 3, 1))
    ant1, ant2 = antpairs[:, 0], antpairs[:, 1]
    return v[ant1, ant2]


def _construct(cls, nant, nfeed, antpairs, blocks, precision, nchunks=1):
    return cls(
        nchunks=nchunks,
        nfeed=nfeed,
        nant=nant,
        antpairs=antpairs,
        precision=precision,
        antenna_blocks=blocks,
    )


def _run(cls, z, nant, nfeed, antpairs, blocks, precision, method, nchunks=1):
    obj = _construct(cls, nant, nfeed, antpairs, blocks, precision, nchunks)
    obj.setup()

    ctype = get_dtypes(precision)[1]
    out = np.zeros((obj.npairs, nfeed, nfeed), dtype=ctype)

    zb = _to_backend(z, method)
    if nchunks == 1:
        obj(zb, chunk=0)
    else:
        splits = np.array_split(np.arange(z.shape[1]), nchunks)
        for c, idx in enumerate(splits):
            obj(_to_backend(z[:, idx], method), chunk=c)
    obj.sum_chunks(out)
    return obj, out


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("nfeed", [1, 2])
@pytest.mark.parametrize("precision", [1, 2])
def test_single_block_covering_everything_matches_full_matmul(method, nfeed, precision):
    """One block spanning all antennas must reproduce the exact full-GEMM result."""
    nant, nsrc = 6, 25
    antpairs = np.array([(i, j) for i in range(nant) for j in range(nant)])
    z = _make_z(nant, nfeed, nsrc, precision)

    all_idx = np.arange(nant)
    blocks = [(all_idx, all_idx)]

    _, out = _run(_get_cls(method), z, nant, nfeed, antpairs, blocks, precision, method)
    expected = _reference_vis(z, nant, nfeed, antpairs)

    rtol = 1e-12 if precision == 2 else 1e-4
    np.testing.assert_allclose(out, expected, rtol=rtol)


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("nfeed", [1, 2])
def test_1x1_blocks_match_vector_dot(method, nfeed):
    """Degenerate 1x1 blocks (one per pair) reproduce the existing VectorDot path.

    It should be a drop-in equivalent to ``VectorDot``, not just "close" to a
    hand-rolled reference.
    """
    precision = 1
    nant, nsrc = 5, 15
    antpairs = np.array([(i, j) for i in range(nant) for j in range(nant) if i != j])
    z = _make_z(nant, nfeed, nsrc, precision)

    blocks = [(np.array([i]), np.array([j])) for i, j in antpairs]

    _, out = _run(_get_cls(method), z, nant, nfeed, antpairs, blocks, precision, method)

    vd_cls_name = "GPUVectorDot" if method.startswith("GPU") else "CPUVectorDot"
    vd_cls = _get_cls(vd_cls_name)
    vd = vd_cls(
        nchunks=1, nfeed=nfeed, nant=nant, antpairs=antpairs, precision=precision
    )
    vd.setup()
    vd_out = np.zeros((vd.npairs, nfeed, nfeed), dtype=out.dtype)
    vd(_to_backend(z, method), chunk=0)
    vd.sum_chunks(vd_out)

    np.testing.assert_allclose(out, vd_out, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_multiple_nonoverlapping_rectangular_blocks(method):
    """A partition into several differently-sized rectangular blocks is correct.

    Uses antpairs in a shuffled (non-block-concatenation) order -- an implementation
    that just concatenates per-block results in block order, without actually
    gathering each requested pair to its own output slot, would fail this.
    """
    precision = 2
    nfeed = 2
    nant, nsrc = 9, 30
    z = _make_z(nant, nfeed, nsrc, precision)

    groups = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6, 7, 8])]
    blocks = [(g, g) for g in groups]
    within_group_pairs = np.array([(i, j) for g in groups for i in g for j in g])
    antpairs = np.random.default_rng(42).permutation(within_group_pairs)

    _, out = _run(_get_cls(method), z, nant, nfeed, antpairs, blocks, precision, method)
    expected = _reference_vis(z, nant, nfeed, antpairs)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("nfeed", [1, 2])
def test_asymmetric_rectangular_block_between_two_groups(method, nfeed):
    """A genuinely rectangular (unequal row/col count) cross-group block works.

    Includes nfeed=2 -- at nfeed=1 a row/col-feed transposition bug in the reshape
    would be invisible.
    """
    precision = 1
    nant, nsrc = 7, 12
    z = _make_z(nant, nfeed, nsrc, precision)

    rows = np.array([0, 1, 2])
    cols = np.array([3, 4, 5, 6])
    blocks = [(rows, cols)]
    antpairs = np.array([(i, j) for i in rows for j in cols])

    _, out = _run(_get_cls(method), z, nant, nfeed, antpairs, blocks, precision, method)
    expected = _reference_vis(z, nant, nfeed, antpairs)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_unsorted_and_overlapping_block_indices(method):
    """Row/col index arrays need not be sorted, contiguous, or non-overlapping.

    Rows/cols may partially overlap (including producing "autocorrelation-like"
    (i, i) entries inside an off-diagonal block). A tempting-but-wrong optimization
    -- taking a contiguous ``z[rows.min():rows.max()+1]`` slice instead of a real
    fancy-index gather -- would silently corrupt this.
    """
    precision = 1
    nfeed = 1
    nant, nsrc = 8, 10
    z = _make_z(nant, nfeed, nsrc, precision, seed=7)

    rows = np.array([5, 0, 3])
    cols = np.array([3, 6, 0])  # overlaps rows at antennas 0 and 3
    blocks = [(rows, cols)]
    antpairs = np.array([(i, j) for i in rows for j in cols])

    _, out = _run(_get_cls(method), z, nant, nfeed, antpairs, blocks, precision, method)
    expected = _reference_vis(z, nant, nfeed, antpairs)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_overlapping_blocks_do_not_double_count(method):
    """A pair covered by two different blocks is claimed once, not summed twice."""
    precision = 1
    nfeed = 1
    nant, nsrc = 4, 10
    z = _make_z(nant, nfeed, nsrc, precision)

    all_idx = np.arange(nant)
    blocks = [(all_idx, all_idx), (all_idx, all_idx)]  # fully overlapping
    antpairs = np.array([(i, j) for i in range(nant) for j in range(nant)])

    _, out = _run(_get_cls(method), z, nant, nfeed, antpairs, blocks, precision, method)
    expected = _reference_vis(z, nant, nfeed, antpairs)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("method", ["CPUMatMul", "CPUVectorDot", "CPUMatBlock"])
def test_duplicate_antpairs_raise(method):
    """Requesting the same pair twice in ``antpairs`` is never what the user wants."""
    kwargs = {}
    if method == "CPUMatBlock":
        all_idx = np.arange(4)
        kwargs["antenna_blocks"] = [(all_idx, all_idx)]

    with pytest.raises(ValueError, match="duplicate"):
        _get_cls(method)(
            nchunks=1,
            nfeed=1,
            nant=4,
            antpairs=np.array([(0, 1), (0, 1), (2, 3)]),
            precision=1,
            **kwargs,
        )


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("nchunks", [1, 2, 3])
def test_nchunks_accumulate_correctly(method, nchunks):
    """Splitting the source axis into chunks must agree with one chunk, shuffled.

    Uses shuffled antpairs so the gather-index mechanism, not just concatenation
    order, is exercised across chunk accumulation too.
    """
    precision = 1
    nfeed = 1
    nant, nsrc = 6, 31  # deliberately not evenly divisible by nchunks
    z = _make_z(nant, nfeed, nsrc, precision, seed=3)

    groups = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    blocks = [(g, g) for g in groups]
    within_group_pairs = np.array([(i, j) for g in groups for i in g for j in g])
    antpairs = np.random.default_rng(11).permutation(within_group_pairs)

    _, out = _run(
        _get_cls(method),
        z,
        nant,
        nfeed,
        antpairs,
        blocks,
        precision,
        method,
        nchunks=nchunks,
    )
    expected = _reference_vis(z, nant, nfeed, antpairs)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_repeated_calls_on_same_chunk_overwrite_not_accumulate(method):
    """Calling compute() again on the same chunk index must overwrite, not accumulate.

    This is the real usage pattern: matvis's per-time loop calls the matprod object
    once per chunk *per time sample*, reusing chunk indices across times.
    """
    precision = 1
    nfeed = 1
    nant, nsrc = 4, 8
    all_idx = np.arange(nant)
    blocks = [(all_idx, all_idx)]
    antpairs = np.array([(i, j) for i in range(nant) for j in range(nant)])

    cls = _get_cls(method)
    obj = _construct(cls, nant, nfeed, antpairs, blocks, precision)
    obj.setup()

    z1 = _make_z(nant, nfeed, nsrc, precision, seed=21)
    z2 = _make_z(nant, nfeed, nsrc, precision, seed=22)

    obj(_to_backend(z1, method), chunk=0)  # "time 0"
    obj(_to_backend(z2, method), chunk=0)  # "time 1" reusing the same chunk index

    out = np.zeros((obj.npairs, nfeed, nfeed), dtype=get_dtypes(precision)[1])
    obj.sum_chunks(out)

    expected = _reference_vis(z2, nant, nfeed, antpairs)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("method", ["CPUMatBlock"])
def test_setup_raises_on_missing_coverage(method):
    """A requested antpair not covered by any block must raise at setup().

    It must not silently produce garbage.
    """
    nant, nfeed, precision = 5, 1, 1
    antpairs = np.array([(0, 1), (2, 3)])
    blocks = [(np.array([0]), np.array([1]))]  # only covers (0, 1)

    obj = _construct(_get_cls(method), nant, nfeed, antpairs, blocks, precision)
    with pytest.raises(ValueError, match="not covered"):
        obj.setup()


@pytest.mark.parametrize("method", ["CPUMatBlock"])
def test_empty_blocks_raises_on_nonempty_antpairs(method):
    """An empty antenna_blocks list can't cover any real antpairs request."""
    obj = _construct(
        _get_cls(method),
        nant=3,
        nfeed=1,
        antpairs=np.array([(0, 1)]),
        blocks=[],
        precision=1,
    )
    with pytest.raises(ValueError, match="not covered"):
        obj.setup()


@pytest.mark.parametrize("method", ["CPUMatBlock"])
def test_antpairs_none_with_blocks_requires_full_coverage(method):
    """antpairs=None expands to all Nant^2 pairs, so a partial block set must raise.

    This is MatProd.__init__'s existing antpairs=None behaviour, unaffected by
    antenna_blocks.
    """
    nant, nfeed, precision = 4, 1, 1
    blocks = [(np.array([0, 1]), np.array([0, 1]))]  # misses antennas 2, 3
    cls = _get_cls(method)
    obj = cls(
        nchunks=1,
        nfeed=nfeed,
        nant=nant,
        antpairs=None,
        precision=precision,
        antenna_blocks=blocks,
    )
    with pytest.raises(ValueError, match="not covered"):
        obj.setup()

    # A fully-covering block set for the same antpairs=None case must succeed.
    all_idx = np.arange(nant)
    obj2 = cls(
        nchunks=1,
        nfeed=nfeed,
        nant=nant,
        antpairs=None,
        precision=precision,
        antenna_blocks=[(all_idx, all_idx)],
    )
    obj2.setup()  # must not raise


@pytest.mark.parametrize(
    "bad_block",
    [
        (np.array([0, 100]), np.array([0])),  # index >= nant
        (np.array([0, -1]), np.array([0])),  # negative index
        (np.array([], dtype=int), np.array([0])),  # empty row block
        (np.array([[0, 1]]), np.array([0])),  # 2D index array
        (np.array([0.0, 1.0]), np.array([0])),  # float dtype
    ],
    ids=["out_of_range", "negative", "empty_block", "not_1d", "non_integer_dtype"],
)
def test_invalid_block_indices_raise(bad_block):
    """Malformed antenna_blocks entries must raise at setup(), not deep in compute()."""
    from matvis.cpu.matprod import CPUMatBlock

    obj = CPUMatBlock(
        nchunks=1,
        nfeed=1,
        nant=5,
        antpairs=np.array([(0, 1)]),
        precision=1,
        antenna_blocks=[bad_block],
    )
    with pytest.raises(ValueError):
        obj.setup()


def test_antenna_blocks_required_for_matblock():
    """CPUMatBlock without any antenna_blocks should fail clearly at construction time.

    It must not silently no-op later.
    """
    from matvis.cpu.matprod import CPUMatBlock

    with pytest.raises(ValueError, match="antenna_blocks"):
        CPUMatBlock(
            nchunks=1,
            nfeed=1,
            nant=3,
            antpairs=np.array([(0, 1)]),
            antenna_blocks=None,
        )


@pytest.mark.parametrize(
    "method",
    ["CPUMatMul", "CPUVectorDot", pytest.param("GPUMatMul", marks=pytest.mark.gpu)],
)
def test_antenna_blocks_rejected_by_non_block_methods(method):
    """Passing antenna_blocks to a method that doesn't understand it is a hard error.

    It must raise at construction time, not be a silently-ignored footgun.
    """
    cls = _get_cls(method)
    blocks = [(np.array([0]), np.array([1]))]
    with pytest.raises(ValueError, match="antenna_blocks"):
        cls(
            nchunks=1,
            nfeed=1,
            nant=3,
            antpairs=np.array([(0, 1)]),
            precision=1,
            antenna_blocks=blocks,
        )


@pytest.mark.parametrize(
    "method",
    ["CPUMatMul", "CPUVectorDot", pytest.param("GPUMatMul", marks=pytest.mark.gpu)],
)
def test_antenna_blocks_none_is_unaffected_regression(method):
    """antenna_blocks=None (the default) must not change any existing class's behaviour."""
    cls = _get_cls(method)
    obj = cls(nchunks=1, nfeed=1, nant=3, antpairs=np.array([(0, 1)]), precision=1)
    obj.setup()
    assert obj.antenna_blocks is None


def test_positional_signature_unchanged():
    """antenna_blocks must be keyword-only, so existing positional calls keep working.

    A 5-positional-argument call site (nchunks, nfeed, nant, antpairs, precision)
    must keep meaning what it always meant.
    """
    from matvis.cpu.matprod import CPUMatMul

    obj = CPUMatMul(2, 1, 3, np.array([(0, 1)]), 1)
    assert obj.nchunks == 2
    assert obj.ctype == get_dtypes(1)[1]  # the 5th positional arg was `precision`
    assert obj.antenna_blocks is None


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("precision", [1, 2])
def test_output_dtype_and_value_match_precision(method, precision):
    """Output dtype must match the requested precision, and the value must be right.

    Checks the underlying buffer, not just a test-owned `out` array that trivially
    takes whatever dtype it's given.
    """
    nant, nfeed, nsrc = 3, 1, 5
    ctype = get_dtypes(precision)[1]
    z = _make_z(nant, nfeed, nsrc, precision)
    all_idx = np.arange(nant)
    blocks = [(all_idx, all_idx)]
    antpairs = np.array([(i, j) for i in range(nant) for j in range(nant)])

    obj = _construct(_get_cls(method), nant, nfeed, antpairs, blocks, precision)
    obj.setup()
    obj(_to_backend(z, method), chunk=0)

    buf = obj.vis[0] if method.startswith("GPU") else obj.vis
    assert buf.dtype == ctype

    out = np.zeros((obj.npairs, nfeed, nfeed), dtype=ctype)
    obj.sum_chunks(out)
    expected = _reference_vis(z, nant, nfeed, antpairs)
    np.testing.assert_allclose(out, expected, rtol=1e-4 if precision == 1 else 1e-10)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_vis_buffer_is_npairs_sized_not_nant_squared(method):
    """The allocated vis buffer must scale with npairs, not with Nant^2.

    Guards against copy-pasting e.g. GPUMatMul's ``allocate_vis`` (which is
    (Nfeed, Nant, Nfeed, Nant)-shaped) into the new class.
    """
    nant, nfeed = 50, 2
    # Only 3 requested pairs, covered by one small block.
    antpairs = np.array([(0, 1), (1, 2), (0, 2)])
    blocks = [(np.array([0, 1, 2]), np.array([0, 1, 2]))]

    obj = _construct(_get_cls(method), nant, nfeed, antpairs, blocks, precision=1)
    obj.setup()

    buf = obj.vis[0] if method.startswith("GPU") else obj.vis
    # However it's laid out, its total element count must be O(npairs), not
    # O(nant^2).
    assert buf.size < nant * nant * nfeed * nfeed / 10


def test_cpu_matblock_memory_does_not_scale_with_nant_squared():
    """Peak memory for a fixed, small block set must not grow with Nant^2.

    Guards against a dense ``(Nant, Nant, Nfeed, Nfeed)`` intermediate wherever it
    might sneak in -- ``compute()``, or a coverage-validation ``(Nant, Nant)`` mask
    in ``setup()``.

    Uses a same-process, warmed-up before/after comparison rather than an
    absolute byte threshold: an absolute threshold is fragile in a full test
    run, since unrelated one-time costs elsewhere in the process (library
    imports, first-use JIT-like paths) can dwarf a tiny problem's real
    footprint depending on what has already run; a growth *ratio* on an
    already-warm process isn't affected by that.
    """
    import tracemalloc

    from matvis.cpu.matprod import CPUMatBlock

    def peak_for(nant):
        nfeed, nsrc, precision = 1, 20, 1
        z = _make_z(nant, nfeed, nsrc, precision, seed=5)

        rng = np.random.default_rng(9)
        chosen = rng.choice(nant, size=40, replace=False)
        blocks = [(chosen[i : i + 4], chosen[i : i + 4]) for i in range(0, 40, 4)]
        antpairs = np.array(
            [(i, j) for rows, cols in blocks for i in rows for j in cols]
        )

        obj = CPUMatBlock(
            nchunks=1,
            nfeed=nfeed,
            nant=nant,
            antpairs=antpairs,
            precision=precision,
            antenna_blocks=blocks,
        )
        tracemalloc.start()
        try:
            obj.setup()
            obj(z, chunk=0)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        out = np.zeros((obj.npairs, nfeed, nfeed), dtype=z.dtype)
        obj.sum_chunks(out)
        expected = _reference_vis(z, nant, nfeed, antpairs)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)
        return peak

    peak_for(50)  # warm up imports/first-use costs before the real comparison

    small = peak_for(400)
    large = peak_for(3200)  # 8x nant -> 64x nant^2, but the same small block set

    # A dense (Nant, Nant, ...) intermediate would grow peak memory ~64x
    # between these two calls; a real O(block size) scatter should barely move.
    assert large < small * 4, (
        f"peak memory grew {large / small:.1f}x when nant grew 8x (400 -> 3200) "
        f"with the same small block set (small={small}, large={large} bytes) "
        "-- looks like an O(Nant^2) intermediate snuck into setup()/compute()"
    )
