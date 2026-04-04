"""Unit tests for coherency matrix and M matrix computation."""

import numpy as np
import pytest

from matvis.core.coherency import (
    coherency_to_stokes,
    compute_m_matrix_eigen,
    compute_m_matrix_sign_split,
    stokes_to_coherency,
)


def _m_times_m_dagger(M):
    """Compute M @ M† per source. M shape (2, 2, Nsrc) → C shape (2, 2, Nsrc)."""
    # C[i,j,n] = sum_k M[i,k,n] * conj(M[j,k,n])
    return np.einsum("ikn,jkn->ijn", M, np.conj(M))


class TestStokesCoherencyConversion:
    def test_stokes_coherency_roundtrip(self):
        """I,Q,U,V → C → I,Q,U,V should return original values."""
        rng = np.random.default_rng(42)
        nsrc = 50
        I = rng.uniform(1.0, 10.0, nsrc)
        Q = rng.uniform(-0.5, 0.5, nsrc) * I
        U = rng.uniform(-0.5, 0.5, nsrc) * I
        V = rng.uniform(-0.5, 0.5, nsrc) * I

        C = stokes_to_coherency(I, Q, U, V)
        I2, Q2, U2, V2 = coherency_to_stokes(C)

        np.testing.assert_allclose(I2, I, atol=1e-12)
        np.testing.assert_allclose(Q2, Q, atol=1e-12)
        np.testing.assert_allclose(U2, U, atol=1e-12)
        np.testing.assert_allclose(V2, V, atol=1e-12)

    def test_coherency_shape(self):
        """Check output shape is (2, 2, Nsrc)."""
        I = np.array([1.0, 2.0, 3.0])
        Q = np.zeros(3)
        U = np.zeros(3)
        V = np.zeros(3)
        C = stokes_to_coherency(I, Q, U, V)
        assert C.shape == (2, 2, 3)

    def test_coherency_hermitian(self):
        """Coherency matrix should be Hermitian: C = C†."""
        rng = np.random.default_rng(123)
        nsrc = 20
        I = rng.uniform(1.0, 10.0, nsrc)
        Q = rng.uniform(-0.3, 0.3, nsrc) * I
        U = rng.uniform(-0.3, 0.3, nsrc) * I
        V = rng.uniform(-0.3, 0.3, nsrc) * I

        C = stokes_to_coherency(I, Q, U, V)
        # C† means conjugate transpose of the 2x2 part
        C_dagger = np.conj(C.transpose(1, 0, 2))
        np.testing.assert_allclose(C, C_dagger, atol=1e-15)


class TestMMatrixEigen:
    def test_m_times_m_dagger_equals_c(self):
        """M @ M† should equal the coherency matrix C for physical sources."""
        rng = np.random.default_rng(42)
        nsrc = 100
        I = rng.uniform(1.0, 10.0, nsrc)
        # Ensure I > T (physical): scale polarization to be < I
        frac = rng.uniform(0.0, 0.8, nsrc)
        direction = rng.normal(size=(3, nsrc))
        direction /= np.linalg.norm(direction, axis=0, keepdims=True)
        T_target = frac * I
        Q = direction[0] * T_target
        U = direction[1] * T_target
        V = direction[2] * T_target

        C = stokes_to_coherency(I, Q, U, V)
        M = compute_m_matrix_eigen(I, Q, U, V)
        C_reconstructed = _m_times_m_dagger(M)

        np.testing.assert_allclose(C_reconstructed, C, atol=1e-10)

    def test_m_unpolarized_matches_diagonal(self):
        """For Q=U=V=0, M should be diag(√(I/2), √(I/2))."""
        I = np.array([1.0, 4.0, 9.0, 16.0])
        Q = np.zeros(4)
        U = np.zeros(4)
        V = np.zeros(4)

        M = compute_m_matrix_eigen(I, Q, U, V)

        expected_diag = np.sqrt(0.5 * I)
        np.testing.assert_allclose(M[0, 0].real, expected_diag, atol=1e-15)
        np.testing.assert_allclose(M[1, 1].real, expected_diag, atol=1e-15)
        np.testing.assert_allclose(M[0, 1], 0.0, atol=1e-15)
        np.testing.assert_allclose(M[1, 0], 0.0, atol=1e-15)

    def test_eigenvalues_physical(self):
        """Both eigenvalues should be non-negative for physical sources (I > T)."""
        rng = np.random.default_rng(99)
        nsrc = 200
        I = rng.uniform(1.0, 100.0, nsrc)
        Q = rng.uniform(-0.3, 0.3, nsrc) * I
        U = rng.uniform(-0.3, 0.3, nsrc) * I
        V = rng.uniform(-0.3, 0.3, nsrc) * I

        T = np.sqrt(Q**2 + U**2 + V**2)
        lp = 0.5 * (I + T)
        lm = 0.5 * (I - T)

        assert np.all(lp >= 0)
        assert np.all(lm >= 0)

    def test_eigenvalues_negative_I_raises(self):
        """Negative I with T=0 should raise ValueError in compute_m_matrix_eigen."""
        I = np.array([-1.0, 2.0, 3.0])
        Q = np.zeros(3)
        U = np.zeros(3)
        V = np.zeros(3)

        with pytest.raises(ValueError, match="Negative eigenvalue"):
            compute_m_matrix_eigen(I, Q, U, V)

    def test_m_T_zero_no_nan(self):
        """No NaN or Inf should appear when T is exactly zero."""
        I = np.array([0.0, 1.0, 100.0, 1e-10])
        Q = np.zeros(4)
        U = np.zeros(4)
        V = np.zeros(4)

        M = compute_m_matrix_eigen(I, Q, U, V)
        assert not np.any(np.isnan(M))
        assert not np.any(np.isinf(M))

    def test_m_T_near_zero(self):
        """M @ M† should equal C when T is very small but nonzero."""
        I = np.array([10.0])
        Q = np.array([1e-14])
        U = np.array([1e-14])
        V = np.array([0.0])

        M = compute_m_matrix_eigen(I, Q, U, V)
        C = stokes_to_coherency(I, Q, U, V)
        C_recon = _m_times_m_dagger(M)

        # M @ M† must equal C regardless of which code path was taken
        np.testing.assert_allclose(C_recon, C, atol=1e-10)
        # No NaNs
        assert not np.any(np.isnan(M))

    def test_m_pure_Q(self):
        """For U=V=0, Q≠0, M should be real diagonal."""
        I = np.array([10.0, 10.0])
        Q = np.array([3.0, -3.0])
        U = np.zeros(2)
        V = np.zeros(2)

        M = compute_m_matrix_eigen(I, Q, U, V)

        # Off-diagonals should be zero
        np.testing.assert_allclose(M[0, 1], 0.0, atol=1e-15)
        np.testing.assert_allclose(M[1, 0], 0.0, atol=1e-15)

        # Diagonal should be real
        np.testing.assert_allclose(M[0, 0].imag, 0.0, atol=1e-15)
        np.testing.assert_allclose(M[1, 1].imag, 0.0, atol=1e-15)

        # Verify M @ M† = C
        C = stokes_to_coherency(I, Q, U, V)
        C_recon = _m_times_m_dagger(M)
        np.testing.assert_allclose(C_recon, C, atol=1e-10)

    def test_m_fully_polarized(self):
        """When I = T (fully polarized), one eigenvalue is zero."""
        I = np.array([5.0])
        Q = np.array([3.0])
        U = np.array([4.0])
        V = np.array([0.0])
        # T = sqrt(9+16) = 5 = I → λ- = 0

        M = compute_m_matrix_eigen(I, Q, U, V)
        C = stokes_to_coherency(I, Q, U, V)
        C_recon = _m_times_m_dagger(M)
        np.testing.assert_allclose(C_recon, C, atol=1e-10)

        # λ- = 0 means second column of M should be zero
        # (or very small)
        col1_norm = np.sqrt(abs(M[0, 1, 0]) ** 2 + abs(M[1, 1, 0]) ** 2)
        assert col1_norm < 1e-10


class TestMMatrixSignSplit:
    def test_sign_split_all_positive(self):
        """When all eigenvalues positive, has_neg should be False."""
        I = np.array([5.0, 10.0, 20.0])
        Q = np.array([1.0, 2.0, 3.0])
        U = np.array([0.5, 1.0, 1.5])
        V = np.array([0.3, 0.6, 0.9])

        M_pos, M_neg, has_neg = compute_m_matrix_sign_split(I, Q, U, V)

        assert not has_neg
        np.testing.assert_allclose(M_neg, 0.0, atol=1e-15)

    def test_sign_split_has_negative(self):
        """Sources with I < 0 should produce has_neg=True."""
        I = np.array([-1.0, 5.0, -3.0])
        Q = np.zeros(3)
        U = np.zeros(3)
        V = np.zeros(3)

        M_pos, M_neg, has_neg = compute_m_matrix_sign_split(I, Q, U, V)

        assert has_neg
        # Negative sources should have nonzero M_neg
        assert np.any(np.abs(M_neg[:, :, 0]) > 0)
        assert np.any(np.abs(M_neg[:, :, 2]) > 0)
        # Positive source should have zero M_neg
        np.testing.assert_allclose(M_neg[:, :, 1], 0.0, atol=1e-15)

    def test_sign_split_recombination(self):
        """V_pos - V_neg should reconstruct the correct coherency matrix.

        For each source: M_pos @ M_pos† - M_neg @ M_neg† should equal C.
        """
        I = np.array([-2.0, 5.0, -0.5, 10.0])
        Q = np.zeros(4)
        U = np.zeros(4)
        V = np.zeros(4)

        C = stokes_to_coherency(I, Q, U, V)
        M_pos, M_neg, has_neg = compute_m_matrix_sign_split(I, Q, U, V)

        C_pos = _m_times_m_dagger(M_pos)
        C_neg = _m_times_m_dagger(M_neg)
        C_recon = C_pos - C_neg

        np.testing.assert_allclose(C_recon, C, atol=1e-10)

    def test_sign_split_matches_eigen_for_positive(self):
        """For all-positive sky, sign-split M_pos should produce same C as eigen M."""
        rng = np.random.default_rng(77)
        nsrc = 30
        I = rng.uniform(1.0, 10.0, nsrc)
        Q = rng.uniform(-0.2, 0.2, nsrc) * I
        U = rng.uniform(-0.2, 0.2, nsrc) * I
        V = rng.uniform(-0.2, 0.2, nsrc) * I

        M_eigen = compute_m_matrix_eigen(I, Q, U, V)
        M_pos, M_neg, has_neg = compute_m_matrix_sign_split(I, Q, U, V)

        assert not has_neg

        C_eigen = _m_times_m_dagger(M_eigen)
        C_split = _m_times_m_dagger(M_pos)

        np.testing.assert_allclose(C_split, C_eigen, atol=1e-10)

    def test_sign_split_polarized_negative_flux(self):
        """Sign-split with polarized sources and negative I."""
        I = np.array([-5.0, 3.0])
        Q = np.array([1.0, 0.5])
        U = np.array([0.5, 1.0])
        V = np.array([0.3, 0.2])

        C = stokes_to_coherency(I, Q, U, V)
        M_pos, M_neg, has_neg = compute_m_matrix_sign_split(I, Q, U, V)

        assert has_neg

        C_pos = _m_times_m_dagger(M_pos)
        C_neg = _m_times_m_dagger(M_neg)
        C_recon = C_pos - C_neg

        np.testing.assert_allclose(C_recon, C, atol=1e-10)

    def test_sign_split_diagonal_negative(self):
        """Sign-split with pure Q polarization and negative C diagonal entries.

        Covers the diagonal case (U≈0, V≈0) where c00 or c11 can be negative.
        """
        # Source 1: I=-3, Q=2, U=V=0 → c00 = 0.5*(-3+2) = -0.5, c11 = 0.5*(-3-2) = -2.5
        # Source 2: I=10, Q=12, U=V=0 → c00 = 0.5*(10+12) = 11, c11 = 0.5*(10-12) = -1
        I = np.array([-3.0, 10.0])
        Q = np.array([2.0, 12.0])
        U = np.zeros(2)
        V = np.zeros(2)

        C = stokes_to_coherency(I, Q, U, V)
        M_pos, M_neg, has_neg = compute_m_matrix_sign_split(I, Q, U, V)

        assert has_neg

        C_pos = _m_times_m_dagger(M_pos)
        C_neg = _m_times_m_dagger(M_neg)
        C_recon = C_pos - C_neg

        np.testing.assert_allclose(C_recon, C, atol=1e-10)

    def test_sign_split_general_negative_both_eigenvalues(self):
        """Sign-split general case where both eigenvalues are negative.

        Covers the general path (U,V ≠ 0) with negative contributions in M_neg.
        """
        # Very negative I with polarization: both eigenvalues negative
        I = np.array([-10.0, -8.0])
        Q = np.array([1.0, 0.5])
        U = np.array([2.0, 1.5])
        V = np.array([1.0, 0.8])

        C = stokes_to_coherency(I, Q, U, V)
        M_pos, M_neg, has_neg = compute_m_matrix_sign_split(I, Q, U, V)

        assert has_neg
        # M_neg should be nonzero for both sources
        assert np.any(np.abs(M_neg[:, :, 0]) > 0)
        assert np.any(np.abs(M_neg[:, :, 1]) > 0)

        C_pos = _m_times_m_dagger(M_pos)
        C_neg = _m_times_m_dagger(M_neg)
        C_recon = C_pos - C_neg

        np.testing.assert_allclose(C_recon, C, atol=1e-10)


class TestEigendecompEdgeCases:
    def test_eigenvalue_tiny_negative_clamped(self):
        """Eigenvalues that are barely negative from float noise should be clamped.

        When I is just barely less than T due to float representation,
        lambda_minus becomes a tiny negative within machine eps.
        This should be clamped to 0, not raise ValueError.
        """
        # I = 4.999...9 (just under 5), T = sqrt(9+16) = 5.0 exactly
        # → λ- = 0.5*(I-T) ≈ -4.4e-16 (negative but within eps)
        I = np.array([4.999999999999999])
        Q = np.array([3.0])
        U = np.array([4.0])
        V = np.array([0.0])

        # Verify this actually produces a tiny negative eigenvalue
        T = np.sqrt(Q**2 + U**2 + V**2)
        lm = 0.5 * (I - T)
        assert lm[0] < 0, "Test setup: need tiny negative eigenvalue"

        # This should NOT raise — the tiny negative is within float eps
        M = compute_m_matrix_eigen(I, Q, U, V)
        assert not np.any(np.isnan(M))

        # Verify reconstruction
        C = stokes_to_coherency(I, Q, U, V)
        C_recon = _m_times_m_dagger(M)
        np.testing.assert_allclose(C_recon, C, atol=1e-8)
