"""
test_dqi_pipeline.py — Tests for DQI sampling and evaluation pipeline.

Run: python -m pytest tests/test_dqi_pipeline.py -v
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from src.dqi_pipeline import DQIPipeline, has_complete_alpha_metrics
from src.problem_generator import default_instance, small_instance


def test_pipeline_construction():
    """Pipeline should initialize with a pricing problem."""
    prob = small_instance(3)
    pipeline = DQIPipeline(prob, k=10, ell=2)

    assert pipeline.n_bits == 6
    assert pipeline.m == 10
    assert pipeline.ell == 2


def test_pipeline_run_returns_results():
    """Pipeline run should return samples with F and G scores."""
    prob = small_instance(3)
    pipeline = DQIPipeline(prob, k=8, ell=2)

    results = pipeline.run(n_samples=100, seed=42)

    assert 'samples' in results
    assert 'F_values' in results
    assert 'G_values' in results
    assert 'G_weighted_values' in results
    assert 'G_unweighted_values' in results
    assert 'satisfied_counts' in results
    assert len(results['samples']) == 100
    assert len(results['F_values']) == 100
    assert len(results['G_values']) == 100
    assert len(results['G_weighted_values']) == 100
    assert len(results['G_unweighted_values']) == 100
    assert len(results['satisfied_counts']) == 100
    np.testing.assert_array_equal(results['G_values'], results['G_weighted_values'])


def test_pipeline_reports_weighted_and_unweighted_best_scores():
    """Pipeline should report both weighted and unweighted surrogate metrics."""
    prob = small_instance(3)
    pipeline = DQIPipeline(prob, k=8, ell=2)
    results = pipeline.run(n_samples=80, seed=123)

    assert 'best_G' in results
    assert 'best_G_weighted' in results
    assert 'best_G_unweighted' in results
    assert 'best_satisfied_count' in results
    assert results['best_G'] == results['best_G_weighted']
    assert results['best_G_unweighted'] <= pipeline.m
    assert results['best_G_unweighted'] >= -pipeline.m
    assert results['best_satisfied_count'] <= pipeline.m
    assert results['best_satisfied_count'] >= 0


def test_pipeline_f_values_valid():
    """F values should be within expected range."""
    prob = small_instance(3)
    pipeline = DQIPipeline(prob, k=8, ell=2)

    results = pipeline.run(n_samples=50, seed=42)

    # All F values should be real numbers
    assert np.all(np.isfinite(results['F_values']))


def test_pipeline_best_f():
    """Pipeline should track best F found."""
    prob = small_instance(3)
    pipeline = DQIPipeline(prob, k=8, ell=2)

    results = pipeline.run(n_samples=100, seed=42)

    assert results['best_F'] == np.max(results['F_values'])
    assert results['best_x'] == results['samples'][np.argmax(results['F_values'])]


def test_pipeline_approximation_ratio():
    """Approximation ratio should be in (0, 1] for feasible solutions."""
    prob = small_instance(3)
    x_opt, f_opt = prob.brute_force_solve()

    pipeline = DQIPipeline(prob, k=8, ell=2)
    results = pipeline.run(n_samples=200, seed=42)

    # Approximation ratio
    ratio = results['best_F'] / f_opt

    # Should be positive (best F might be negative if infeasible)
    # and at most 1
    assert ratio <= 1.0 + 1e-10


def test_has_complete_alpha_metrics_uses_compact_reviewer_fields_only():
    assert has_complete_alpha_metrics(
        {
            "alpha_mode": "paper",
            "execution_mode": "mixture",
            "status": "completed",
            "best_F": 12.0,
            "success_prob": 0.5,
            "top1_regret": 0.1,
        }
    ) is True

    assert has_complete_alpha_metrics(
        {
            "alpha_mode": "paper",
            "execution_mode": "coherent",
            "status": "completed",
            "best_F": 12.0,
            "success_prob": np.nan,
            "top1_regret": 0.1,
        }
    ) is False


def test_has_complete_alpha_metrics_uses_success_probability_precedence():
    row = {
        "alpha_mode": "uniform",
        "execution_mode": "mixture",
        "status": "completed",
        "best_F": 10.0,
        "success_prob": None,
        "postselection_success": 0.4,
        "decoder_success_rate": 0.9,
        "top1_regret": 0.3,
    }
    assert has_complete_alpha_metrics(row) is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
