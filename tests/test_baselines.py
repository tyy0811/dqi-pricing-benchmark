"""
test_baselines.py — Tests for classical baseline solvers.

Run: python -m pytest tests/test_baselines.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

from src.classical_baselines import (
    brute_force_solver,
    cp_sat_solver,
    run_all_baselines,
    simulated_annealing_solver,
)
from src.problem_generator import small_instance
from src.benchmark import load_problem_from_json


def test_brute_force_finds_optimum():
    """Brute force should find the exact optimum."""
    prob = small_instance(3)
    x_opt_bf, f_opt_bf = brute_force_solver(prob)
    x_opt, f_opt = prob.brute_force_solve()

    assert x_opt_bf == x_opt
    assert f_opt_bf == f_opt


def test_simulated_annealing_returns_valid():
    """SA should return a valid configuration and objective."""
    prob = small_instance(3)
    x_sa, f_sa = simulated_annealing_solver(prob, max_iter=1000, seed=42)

    # Should be valid integer
    assert 0 <= x_sa < 2 ** prob.n_bits

    # Should evaluate correctly
    assert f_sa == prob.evaluate(x_sa)


def test_simulated_annealing_quality():
    """SA should find reasonable solution (at least better than random)."""
    prob = small_instance(3)
    table = prob.build_truth_table()
    mean_f = np.mean(table)

    x_sa, f_sa = simulated_annealing_solver(prob, max_iter=5000, seed=42)

    # SA should beat mean
    assert f_sa > mean_f


def test_simulated_annealing_deterministic_with_seed():
    """SA with same seed should give same result."""
    prob = small_instance(3)

    x1, f1 = simulated_annealing_solver(prob, max_iter=500, seed=123)
    x2, f2 = simulated_annealing_solver(prob, max_iter=500, seed=123)

    assert x1 == x2
    assert f1 == f2


def test_run_all_baselines_cp_sat_status_and_optimality():
    """CP-SAT baseline should be exact when available, else cleanly unavailable."""
    prob = small_instance(3)  # n_bits=6 (within intended CP-SAT range)
    results = run_all_baselines(prob, sa_iter=300, random_samples=100, seed=11)

    assert 'cp_sat' in results
    cp_sat = results['cp_sat']
    assert cp_sat.get('status') in {'ok', 'unavailable'}

    if cp_sat.get('status') == 'ok':
        x_opt, f_opt = brute_force_solver(prob)
        assert cp_sat.get('optimal') is True
        assert cp_sat['f'] == f_opt
        assert cp_sat['x'] is not None
        assert cp_sat.get('solver') == 'ortools_cp_sat_one_hot_exact'
        assert cp_sat.get('matches_bruteforce_objective') is True
    else:
        assert cp_sat.get('optimal') is False
        assert cp_sat.get('x') is None
        assert cp_sat.get('f') is None
        assert isinstance(cp_sat.get('reason', ''), str)


def test_run_all_baselines_cp_sat_uses_standard_contract():
    """CP-SAT payload should expose pricing-model contract fields."""
    prob = small_instance(3)
    results = run_all_baselines(prob, sa_iter=300, random_samples=100, seed=11)
    cp_sat = results['cp_sat']

    assert cp_sat.get('status') in {'ok', 'unavailable'}
    if cp_sat.get('status') == 'ok':
        assert 'objective_value' in cp_sat
        assert 'optimized_objective_value' in cp_sat
        assert cp_sat['objective_value'] == prob.evaluate(cp_sat['x'])


def test_cp_sat_matches_bruteforce_when_ortools_available():
    """CP-SAT should return the exact same optimal F as brute force."""
    pytest.importorskip("ortools.sat.python.cp_model")
    prob = small_instance(3)  # n_bits=6

    x_cp, f_cp = cp_sat_solver(prob)
    x_bf, f_bf = brute_force_solver(prob)

    assert f_cp == f_bf
    assert prob.evaluate(x_cp) == f_bf
    assert prob.evaluate(x_bf) == f_bf


def test_cp_sat_matches_bruteforce_on_frozen_instances_when_ortools_available():
    """CP-SAT should match brute-force objective on frozen pricing instances."""
    pytest.importorskip("ortools.sat.python.cp_model")

    for path in [
        "data/instances/pricing_3feat_6bit.json",
        "data/instances/pricing_4feat_8bit.json",
        "data/instances/pricing_5feat_10bit.json",
        "data/instances/pricing_6feat_12bit.json",
        "data/instances/pricing_7feat_14bit.json",
    ]:
        prob = load_problem_from_json(path)
        x_cp, f_cp = cp_sat_solver(prob)
        x_bf, f_bf = brute_force_solver(prob)
        assert f_cp == f_bf
        assert prob.evaluate(x_cp) == f_bf
        assert prob.evaluate(x_bf) == f_bf


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
