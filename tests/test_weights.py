"""
test_weights.py — Tests for optimal DQI weight computation.

Run: python -m pytest tests/test_weights.py -v
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from src.weights import (
    uniform_weights,
    optimal_weights,
    semicircle_prediction,
)
from src import weights_paper, weights_heuristic


def test_uniform_weights_normalized():
    """Uniform weights should be normalized."""
    for ell in [1, 2, 3, 4]:
        alpha = uniform_weights(ell)
        assert len(alpha) == ell + 1
        np.testing.assert_allclose(np.sum(alpha ** 2), 1.0, rtol=1e-10)


def test_optimal_weights_normalized():
    """Optimal weights should be normalized."""
    m = 15
    # Simple B matrix for testing
    B = np.random.randint(0, 2, size=(m, 10), dtype=np.int8)
    v = np.random.randint(0, 2, size=m, dtype=np.int8)

    for ell in [1, 2, 3]:
        alpha = optimal_weights(B, v, ell)
        assert len(alpha) == ell + 1
        np.testing.assert_allclose(np.sum(alpha ** 2), 1.0, rtol=1e-10)


def test_optimal_weights_real():
    """Optimal weights should be real (not complex)."""
    m = 10
    B = np.eye(m, dtype=np.int8)[:, :5]  # m=10, n=5
    v = np.zeros(m, dtype=np.int8)

    alpha = optimal_weights(B, v, ell=2)
    assert np.isreal(alpha).all()


def test_semicircle_prediction():
    """Semicircle prediction should be in valid range."""
    for m in [10, 15, 20]:
        for ell in [1, 2, 3, 4]:
            if ell < m:
                pred = semicircle_prediction(m, ell)
                # Expected fraction is between 0.5 and 1.0
                assert 0.5 <= pred <= 1.0, f"m={m}, ell={ell}: {pred}"


def test_semicircle_at_boundaries():
    """Semicircle prediction at ell=0 should give 0.5 (random guess)."""
    pred = semicircle_prediction(m=20, ell=0)
    np.testing.assert_allclose(pred, 0.5, rtol=1e-10)


def test_weights_shim_uniform_matches_paper_module():
    """Compatibility shim should forward uniform weights to paper module."""
    for ell in [1, 2, 3]:
        np.testing.assert_allclose(
            uniform_weights(ell),
            weights_paper.uniform_weights(ell),
        )


def test_weights_shim_optimal_matches_heuristic_module():
    """Compatibility shim should forward heuristic optimal weights."""
    m = 12
    B = np.random.randint(0, 2, size=(m, 8), dtype=np.int8)
    v = np.random.randint(0, 2, size=m, dtype=np.int8)

    alpha_shim = optimal_weights(B, v, ell=3)
    alpha_new = weights_heuristic.optimal_weights(B, v, ell=3)
    np.testing.assert_allclose(alpha_shim, alpha_new, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
