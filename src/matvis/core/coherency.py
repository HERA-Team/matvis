"""Coherency matrix and M matrix computation for polarized sky models.

The coherency matrix C for a source with Stokes parameters (I, Q, U, V) is:

    C = 0.5 * [[I+Q, U+iV], [U-iV, I-Q]]

The matvis algorithm factorizes C = M @ M† and builds Z = Σ_k A·F·M
so that V = Z @ Z† recovers the full RIME.

For physical sources (I² > Q²+U²+V²), the eigenvalues of C are both
non-negative, and M can be constructed via eigendecomposition. For
EOR-like scenarios where I < 0, a sign-split approach separates
positive and negative eigenvalue contributions.

Reference: Murray et al. (2024), arXiv:2312.09763, Section 3.2 & Appendix B.
"""

import numpy as np

try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False


def stokes_to_coherency(I, Q, U, V, xp=np):
    """Convert Stokes parameters to 2x2 coherency matrices.

    Includes the factor of 0.5 so that the unpolarized case gives
    C = diag(I/2, I/2), matching the current matvis convention where
    flux is split equally between two feeds.

    Parameters
    ----------
    I, Q, U, V : array_like
        Stokes parameters, each shape (Nsrc,).
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    C : ndarray
        Coherency matrices, shape (2, 2, Nsrc), complex.
    """
    I = xp.asarray(I)
    Q = xp.asarray(Q)
    U = xp.asarray(U)
    V = xp.asarray(V)

    nsrc = len(I)
    ctype = xp.result_type(I, 1j)  # float32 → complex64, float64 → complex128
    C = xp.empty((2, 2, nsrc), dtype=ctype)
    C[0, 0] = 0.5 * (I + Q)
    C[0, 1] = 0.5 * (U + 1j * V)
    C[1, 0] = 0.5 * (U - 1j * V)
    C[1, 1] = 0.5 * (I - Q)
    return C


def coherency_to_stokes(C):
    """Extract Stokes parameters from 2x2 coherency matrices.

    Assumes C includes the 0.5 factor (i.e. C = 0.5 * [[I+Q, ...], ...]).
    Works with both numpy and cupy arrays via duck typing.

    Parameters
    ----------
    C : ndarray
        Coherency matrices, shape (2, 2, Nsrc), complex.

    Returns
    -------
    I, Q, U, V : ndarray
        Stokes parameters, each shape (Nsrc,), real.
    """
    I = (C[0, 0] + C[1, 1]).real
    Q = (C[0, 0] - C[1, 1]).real
    U = 2.0 * C[0, 1].real
    V = 2.0 * C[0, 1].imag
    return I, Q, U, V


def compute_m_matrix_eigen(I, Q, U, V, xp=np):
    """Compute the M matrix via eigendecomposition of the coherency matrix.

    The coherency matrix C = 0.5 * [[I+Q, U+iV], [U-iV, I-Q]] is Hermitian
    with eigenvalues λ± = 0.5 * (I ± T), where T = √(Q²+U²+V²).

    For physical sources (I > T), both eigenvalues are positive and
    M = U_mat @ diag(√λ+, √λ-) satisfies C = M @ M†.

    Three special cases are handled:
    - T ≈ 0 (unpolarized): M = diag(√(I/2), √(I/2))
    - U ≈ 0, V ≈ 0 (pure Q): M = diag(√λ+, √λ-)
    - General: full eigenvector decomposition

    Parameters
    ----------
    I, Q, U, V : array_like
        Stokes parameters, each shape (Nsrc,).
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    M : ndarray
        M matrix, shape (2, 2, Nsrc), complex. Satisfies C = M @ M†.

    Raises
    ------
    ValueError
        If any eigenvalue is negative (use compute_m_matrix_sign_split instead).
    """
    input_dtype = xp.result_type(I, Q, U, V)
    if not xp.issubdtype(input_dtype, xp.floating):
        input_dtype = xp.float64
    I = xp.asarray(I, dtype=input_dtype)
    Q = xp.asarray(Q, dtype=input_dtype)
    U = xp.asarray(U, dtype=input_dtype)
    V = xp.asarray(V, dtype=input_dtype)

    nsrc = len(I)
    T = xp.sqrt(Q**2 + U**2 + V**2)

    lambda_plus = 0.5 * (I + T)
    lambda_minus = 0.5 * (I - T)

    # Check for negative eigenvalues — convert to Python float once to avoid
    # multiple GPU→CPU syncs when xp is cupy.
    min_eigenvalue = float(xp.min(lambda_minus))
    if min_eigenvalue < 0:
        max_abs_I = float(xp.max(xp.abs(I)))
        eps = float(xp.finfo(input_dtype).eps) * max(max_abs_I, 1.0)
        if min_eigenvalue < -eps:
            raise ValueError(
                f"Negative eigenvalue detected (min={min_eigenvalue:.6e}). "
                "Use compute_m_matrix_sign_split for sky models with negative flux."
            )
        # Clamp small negative values from floating point errors
        lambda_minus = xp.maximum(lambda_minus, 0.0)

    sqrt_lp = xp.sqrt(lambda_plus)
    sqrt_lm = xp.sqrt(lambda_minus)

    # Masks for special cases
    eps = xp.finfo(I.dtype).eps * xp.maximum(xp.abs(I), 1.0)

    mask_unpolarized = T < eps
    mask_diagonal = (~mask_unpolarized) & ((xp.abs(U) + xp.abs(V)) < eps)
    mask_general = ~mask_unpolarized & ~mask_diagonal

    ctype = xp.result_type(input_dtype, 1j)  # float32 → complex64, float64 → complex128
    M = xp.zeros((2, 2, nsrc), dtype=ctype)

    # Case 1: Unpolarized (T ≈ 0) → M = diag(√(I/2), √(I/2))
    if xp.any(mask_unpolarized):
        sqrt_half_I = xp.sqrt(xp.maximum(0.5 * I[mask_unpolarized], 0.0))
        M[0, 0, mask_unpolarized] = sqrt_half_I
        M[1, 1, mask_unpolarized] = sqrt_half_I

    # Case 2: Pure Q polarization (U ≈ 0, V ≈ 0) → C is diagonal, so M is diagonal.
    # M = diag(√C[0,0], √C[1,1]) = diag(√(0.5*(I+Q)), √(0.5*(I-Q)))
    # Note: we use C diagonal entries directly, not λ± which don't preserve
    # the matrix ordering when Q < 0.
    if xp.any(mask_diagonal):
        md = mask_diagonal
        M[0, 0, md] = xp.sqrt(xp.maximum(0.5 * (I[md] + Q[md]), 0.0))
        M[1, 1, md] = xp.sqrt(xp.maximum(0.5 * (I[md] - Q[md]), 0.0))

    # Case 3: General polarization → full eigenvector decomposition
    if xp.any(mask_general):
        mg = mask_general
        T_g = T[mg]
        Q_g = Q[mg]
        U_g = U[mg]
        V_g = V[mg]

        # Eigenvectors of C (unnormalized):
        #   v+ = [U+iV, T-Q]   for eigenvalue λ+
        #   v- = [Q-T, U-iV]   for eigenvalue λ-
        # Norm: ||v+||² = (U²+V²) + (T-Q)² = 2T(T-Q)
        #        ||v-||² = (Q-T)² + (U²+V²) = 2T(T-Q)  [same]
        norm_sq = 2.0 * T_g * (T_g - Q_g)

        # Guard against norm_sq ≈ 0 (happens when T ≈ Q, i.e. U,V ≈ 0)
        # This should be caught by mask_diagonal, but add safety
        safe_norm = xp.sqrt(xp.maximum(norm_sq, xp.finfo(norm_sq.dtype).tiny))

        # Normalized eigenvectors
        # v+_hat = [U+iV, T-Q] / norm
        v_plus_0 = (U_g + 1j * V_g) / safe_norm
        v_plus_1 = (T_g - Q_g) / safe_norm

        # v-_hat = [Q-T, U-iV] / norm
        v_minus_0 = (Q_g - T_g) / safe_norm
        v_minus_1 = (U_g - 1j * V_g) / safe_norm

        # M = [v+_hat * √λ+, v-_hat * √λ-]  (columns)
        slp = sqrt_lp[mg]
        slm = sqrt_lm[mg]

        M[0, 0, mg] = v_plus_0 * slp
        M[1, 0, mg] = v_plus_1 * slp
        M[0, 1, mg] = v_minus_0 * slm
        M[1, 1, mg] = v_minus_1 * slm

    return M


def compute_m_matrix_sign_split(I, Q, U, V, xp=np):
    """Compute M matrices for sign-split approach (handles negative eigenvalues).

    Splits sources by eigenvalue sign. Positive eigenvalues contribute to
    M_pos, negative eigenvalues (in absolute value) contribute to M_neg.
    The caller computes V = Z_pos @ Z_pos† - Z_neg @ Z_neg†.

    Parameters
    ----------
    I, Q, U, V : array_like
        Stokes parameters, each shape (Nsrc,).
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    M_pos : ndarray
        M matrix for positive eigenvalue contributions, shape (2, 2, Nsrc).
    M_neg : ndarray
        M matrix for negative eigenvalue contributions, shape (2, 2, Nsrc).
    has_neg : bool
        True if any eigenvalue is negative.
    """
    input_dtype = xp.result_type(I, Q, U, V)
    if not xp.issubdtype(input_dtype, xp.floating):
        input_dtype = xp.float64
    I = xp.asarray(I, dtype=input_dtype)
    Q = xp.asarray(Q, dtype=input_dtype)
    U = xp.asarray(U, dtype=input_dtype)
    V = xp.asarray(V, dtype=input_dtype)

    nsrc = len(I)
    T = xp.sqrt(Q**2 + U**2 + V**2)

    lambda_plus = 0.5 * (I + T)
    lambda_minus = 0.5 * (I - T)

    # Determine sign of each eigenvalue — use element-wise OR and single
    # bool() conversion to avoid fragile Python `or` on CuPy 0-d arrays.
    neg_plus = lambda_plus < 0
    neg_minus = lambda_minus < 0
    has_neg = bool(xp.any(neg_plus | neg_minus))

    # Construct sqrt of absolute eigenvalues
    sqrt_lp_pos = xp.sqrt(xp.maximum(lambda_plus, 0.0))
    sqrt_lm_pos = xp.sqrt(xp.maximum(lambda_minus, 0.0))
    sqrt_lp_neg = xp.sqrt(xp.maximum(-lambda_plus, 0.0))
    sqrt_lm_neg = xp.sqrt(xp.maximum(-lambda_minus, 0.0))

    # Masks for special cases
    eps = xp.finfo(I.dtype).eps * xp.maximum(xp.abs(I), 1.0)
    mask_unpolarized = T < eps
    mask_diagonal = (~mask_unpolarized) & ((xp.abs(U) + xp.abs(V)) < eps)
    mask_general = ~mask_unpolarized & ~mask_diagonal

    ctype = xp.result_type(input_dtype, 1j)
    M_pos = xp.zeros((2, 2, nsrc), dtype=ctype)
    M_neg = xp.zeros((2, 2, nsrc), dtype=ctype)

    # Case 1: Unpolarized (T ≈ 0) → both eigenvalues equal I/2
    if xp.any(mask_unpolarized):
        mu = mask_unpolarized
        half_I = 0.5 * I[mu]
        sqrt_pos = xp.sqrt(xp.maximum(half_I, 0.0))
        sqrt_neg = xp.sqrt(xp.maximum(-half_I, 0.0))
        M_pos[0, 0, mu] = sqrt_pos
        M_pos[1, 1, mu] = sqrt_pos
        M_neg[0, 0, mu] = sqrt_neg
        M_neg[1, 1, mu] = sqrt_neg

    # Case 2: Pure Q (U ≈ 0, V ≈ 0) → C is diagonal, M is diagonal.
    # Use C diagonal entries directly: C[0,0] = 0.5*(I+Q), C[1,1] = 0.5*(I-Q)
    if xp.any(mask_diagonal):
        md = mask_diagonal
        c00 = 0.5 * (I[md] + Q[md])
        c11 = 0.5 * (I[md] - Q[md])
        M_pos[0, 0, md] = xp.sqrt(xp.maximum(c00, 0.0))
        M_pos[1, 1, md] = xp.sqrt(xp.maximum(c11, 0.0))
        M_neg[0, 0, md] = xp.sqrt(xp.maximum(-c00, 0.0))
        M_neg[1, 1, md] = xp.sqrt(xp.maximum(-c11, 0.0))

    # Case 3: General → full eigenvector decomposition
    if xp.any(mask_general):
        mg = mask_general
        T_g = T[mg]
        Q_g = Q[mg]
        U_g = U[mg]
        V_g = V[mg]

        norm_sq = 2.0 * T_g * (T_g - Q_g)
        safe_norm = xp.sqrt(xp.maximum(norm_sq, xp.finfo(norm_sq.dtype).tiny))

        v_plus_0 = (U_g + 1j * V_g) / safe_norm
        v_plus_1 = (T_g - Q_g) / safe_norm
        v_minus_0 = (Q_g - T_g) / safe_norm
        v_minus_1 = (U_g - 1j * V_g) / safe_norm

        # Positive contributions
        slp_p = sqrt_lp_pos[mg]
        slm_p = sqrt_lm_pos[mg]
        M_pos[0, 0, mg] = v_plus_0 * slp_p
        M_pos[1, 0, mg] = v_plus_1 * slp_p
        M_pos[0, 1, mg] = v_minus_0 * slm_p
        M_pos[1, 1, mg] = v_minus_1 * slm_p

        # Negative contributions
        slp_n = sqrt_lp_neg[mg]
        slm_n = sqrt_lm_neg[mg]
        M_neg[0, 0, mg] = v_plus_0 * slp_n
        M_neg[1, 0, mg] = v_plus_1 * slp_n
        M_neg[0, 1, mg] = v_minus_0 * slm_n
        M_neg[1, 1, mg] = v_minus_1 * slm_n

    return M_pos, M_neg, has_neg
