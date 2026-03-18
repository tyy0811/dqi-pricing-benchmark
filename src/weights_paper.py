"""Paper-derived DQI alpha utilities.

The paper-mode ablation uses the finite tridiagonal matrix

    A^(ell) in R^((ell+1) x (ell+1)),
    A[i, i] = 0,
    A[i, i+1] = A[i+1, i] = sqrt((i+1) * (ell-i))  for i = 0..ell-1.

The paper alpha vector is the principal eigenvector of A^(ell), with sign fixed
so the first component is non-negative. By default the returned vector is
L2-normalized, i.e., ||alpha||_2 = 1.

This module also keeps parameter-safety helpers used by benchmark diagnostics.
"""

from __future__ import annotations

from math import comb

import numpy as np


def max_safe_ell(m: int, n: int) -> int:
    """Compute maximum ell such that error patterns fit in syndrome space.

    Requires: sum(C(m, t), t=0..ell) <= 2^n.
    """
    syndrome_space = 2 ** n
    total = 0
    for ell in range(m + 1):
        total += comb(m, ell)
        if total > syndrome_space:
            return max(0, ell - 1)
    return m


def validate_dqi_parameters(m: int, n: int, ell: int) -> tuple[bool, str]:
    """Check if DQI parameters exceed syndrome-space capacity."""
    safe_ell = max_safe_ell(m, n)
    error_patterns = sum(comb(m, t) for t in range(ell + 1))
    syndrome_space = 2 ** n

    if ell <= safe_ell:
        return True, f"OK: {error_patterns} patterns fit in {syndrome_space} syndromes"
    return False, (
        f"WARNING: ell={ell} too large. {error_patterns} error patterns > "
        f"{syndrome_space} syndromes. Max safe ell={safe_ell}. "
        "Expect low success probability and poor results."
    )


def uniform_weights(ell: int) -> np.ndarray:
    """Return normalized uniform alpha vector of length ell+1."""
    alpha = np.ones(ell + 1, dtype=np.float64)
    alpha /= np.linalg.norm(alpha)
    return alpha


def paper_alpha_matrix(ell: int) -> np.ndarray:
    """Construct the paper-derived finite tridiagonal matrix A^(ell)."""
    if ell < 0:
        raise ValueError(f"ell must be non-negative, got {ell}")

    A = np.zeros((ell + 1, ell + 1), dtype=np.float64)
    for i in range(ell):
        coupling = float(np.sqrt((i + 1) * (ell - i)))
        A[i, i + 1] = coupling
        A[i + 1, i] = coupling
    return A


def paper_alpha_weights(ell: int, *, normalized: bool = True) -> np.ndarray:
    """Return paper-derived alpha via principal eigenvector of A^(ell)."""
    A = paper_alpha_matrix(ell)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    principal_idx = int(np.argmax(eigenvalues))
    alpha = np.asarray(eigenvectors[:, principal_idx].real, dtype=np.float64)

    # Eigenvectors are defined up to sign; keep deterministic orientation.
    if alpha[0] < 0:
        alpha = -alpha

    if normalized:
        norm = float(np.linalg.norm(alpha))
        if norm <= 0:
            raise ValueError("paper_alpha_weights produced zero-norm eigenvector")
        alpha = alpha / norm

    return alpha


def semicircle_prediction(m: int, ell: int, r_over_p: float = 0.5) -> float:
    """Compute semicircle-law prediction for expected satisfaction fraction."""
    if ell == 0:
        return r_over_p

    ratio = ell / m
    if r_over_p == 0.5:
        return 0.5 + np.sqrt(ratio * (1 - ratio))

    term1 = np.sqrt(ratio * (1 - r_over_p))
    term2 = np.sqrt(r_over_p * (1 - ratio))
    return (term1 + term2) ** 2
