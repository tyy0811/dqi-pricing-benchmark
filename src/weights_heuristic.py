"""Heuristic-proxy alpha-weight computation utilities.

These routines intentionally use a proxy matrix and are not the paper-derived
finite tridiagonal construction.
"""

from __future__ import annotations

import numpy as np

from src.weights_paper import semicircle_prediction, uniform_weights


def build_weight_matrix_heuristic(
    B: np.ndarray,
    v: np.ndarray,
    ell: int,
) -> np.ndarray:
    """Build a heuristic (ell+1)x(ell+1) symmetric weight matrix."""
    m = B.shape[0]
    A = np.zeros((ell + 1, ell + 1), dtype=np.float64)

    for t in range(ell + 1):
        A[t, t] = 0.5 + 0.1 * t / max(ell, 1)
        for tp in range(t + 1, ell + 1):
            A[t, tp] = A[t, t] * 0.5 ** (tp - t)
            A[tp, t] = A[t, tp]

    max_val = np.max(np.abs(A))
    if max_val > 0:
        A = A / max_val

    return A


def heuristic_alpha_weights(
    B: np.ndarray,
    v: np.ndarray,
    ell: int,
) -> np.ndarray:
    """Compute heuristic-proxy alpha via principal eigenvector."""
    A = build_weight_matrix_heuristic(B, v, ell)

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    principal_idx = np.argmax(eigenvalues)
    alpha = eigenvectors[:, principal_idx].real

    if alpha[0] < 0:
        alpha = -alpha

    alpha /= np.linalg.norm(alpha)
    return alpha


def optimal_weights(
    B: np.ndarray,
    v: np.ndarray,
    ell: int,
) -> np.ndarray:
    """Backward-compatible alias for heuristic_alpha_weights()."""
    return heuristic_alpha_weights(B, v, ell)


def compare_weights(
    B: np.ndarray,
    v: np.ndarray,
    ell: int,
) -> dict:
    """Compare uniform and heuristic-proxy alpha vectors."""
    m = B.shape[0]
    return {
        "uniform": uniform_weights(ell),
        "heuristic": heuristic_alpha_weights(B, v, ell),
        "optimal": heuristic_alpha_weights(B, v, ell),  # compatibility alias
        "semicircle_prediction": semicircle_prediction(m, ell),
        "ell": ell,
        "m": m,
    }
