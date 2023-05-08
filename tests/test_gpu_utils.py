"""Test GPU utilities."""
import pytest

from hypothesis import assume, given
from hypothesis import strategies as st

from vis_cpu.gpu import _get_3d_block_grid


class Test_GetBlockGrid:
    """Tests of the _get_3d_block_grid function."""

    @given(
        nthreads=st.integers(min_value=1),
        a=st.integers(min_value=1),
        b=st.integers(min_value=1),
        c=st.integers(min_value=1),
    )
    def test_get_3d_block_grid_happy(self, nthreads, a, b, c):
        """Test the function with valid input parameters."""
        block, grid = _get_3d_block_grid(nthreads, a, b, c)

        # Ensure we're hitting the whole axis for each dimension
        assert block[0] * grid[0] >= a
        assert block[1] * grid[1] >= b
        assert block[2] * grid[2] >= c

        # Ensure we're never using more threads than we have
        assert block[0] * block[1] * block[2] <= nthreads

        # Ensure we're using the maximum number of blocks possible
        ntasks = a * b * c
        assert (block[0] + 1) * block[1] * block[2] > min(nthreads, ntasks) or block[
            0
        ] + 1 > a
        assert (
            block[0] * (block[1] + 1) * block[2] > min(nthreads, ntasks)
            or block[1] + 1 > b
        )
        assert (
            block[0] * block[1] * (block[2] + 1) > min(nthreads, ntasks)
            or block[2] + 1 > c
        )

        # Ensure we're using more threads in the earlier dimensions
        assert (a // block[0]) <= (b // block[1]) or block[1] == 1
        assert (b // block[1]) <= (c // block[2]) or block[2] == 1

    def test_get_3d_block_grid_nthreads_zero(self):
        """Tests the function when nthreads is 0."""
        with pytest.raises(ValueError):
            _get_3d_block_grid(0, 128, 64, 32)

    def test_get_3d_block_grid_negative_input(self):
        """Tests the function when a, b, c, or nthreads is negative."""
        with pytest.raises(ValueError):
            _get_3d_block_grid(64, -128, 64, 32)
        with pytest.raises(ValueError):
            _get_3d_block_grid(64, 128, -64, 32)
        with pytest.raises(ValueError):
            _get_3d_block_grid(64, 128, 64, -32)
        with pytest.raises(ValueError):
            _get_3d_block_grid(-64, 128, 64, 32)

    @given(
        nthreads=st.integers(min_value=2),
        a=st.integers(min_value=1),
    )
    def test_get_3d_block_grid_nthreads_greater_than_a(self, nthreads, a):
        """Tests the function when nthreads is greater than a."""
        assume(nthreads > a)
        block, grid = _get_3d_block_grid(nthreads, a, 1, 1)
        assert block == (a, 1, 1)
        assert grid[1:] == (1, 1)
        assert block[0] * grid[0] >= a

    @given(
        nthreads=st.integers(min_value=1),
        b=st.integers(min_value=1),
        c=st.integers(min_value=1),
    )
    def test_get_3d_block_grid_a_equal_to_nthreads(self, nthreads, b, c):
        """Tests the function when a is equal to nthreads."""
        block, grid = _get_3d_block_grid(nthreads, nthreads, b, c)
        assert block == (nthreads, 1, 1)
        assert grid == (1, b, c)
