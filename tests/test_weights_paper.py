"""Tests for paper-derived alpha weights."""

from __future__ import annotations

import numpy as np
import sys

sys.path.insert(0, ".")

from src.weights import heuristic_alpha_weights, paper_alpha_weights


def test_paper_alpha_weights_returns_normalized_length_ell_plus_one() -> None:
    """Paper alpha constructor should return a normalized vector of length ell+1."""
    ell = 4
    alpha = paper_alpha_weights(ell)

    assert alpha.shape == (ell + 1,)
    assert np.all(np.isfinite(alpha))
    np.testing.assert_allclose(np.linalg.norm(alpha), 1.0, rtol=1e-12, atol=1e-12)


def test_paper_and_heuristic_alpha_not_identical_for_representative_ell() -> None:
    """Paper and heuristic alpha paths should differ on at least one representative ell."""
    ell = 3
    B = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    v = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)

    alpha_paper = paper_alpha_weights(ell)
    alpha_heuristic = heuristic_alpha_weights(B, v, ell)

    assert alpha_paper.shape == alpha_heuristic.shape == (ell + 1,)
    assert not np.allclose(alpha_paper, alpha_heuristic, atol=1e-12, rtol=1e-10)
