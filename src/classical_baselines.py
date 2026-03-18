"""
classical_baselines.py — Classical baseline solvers for pricing optimization.

Provides brute-force, CP-SAT, and simulated annealing solvers that operate on
the true objective F(x), for comparison with DQI.

This module remains the single-period baseline surface. For the separate fixed
3-period extension, see ``src.classical_baselines_multiperiod``.

Reference: Jordan et al. (2024) Table 2 uses SA as baseline.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional

from src.pricing_model_ortools import solve_pricing_with_cp_sat

if TYPE_CHECKING:
    from src.problem_generator import PricingProblem


def brute_force_solver(prob: "PricingProblem") -> tuple[int, float]:
    """Solve by exhaustive enumeration.

    Args:
        prob: PricingProblem instance.

    Returns:
        (x_opt, f_opt): Optimal configuration and objective value.
    """
    return prob.brute_force_solve()


def simulated_annealing_solver(
    prob: "PricingProblem",
    max_iter: int = 10000,
    initial_temp: float = 1000.0,
    cooling_rate: float = 0.995,
    seed: Optional[int] = None,
) -> tuple[int, float]:
    """Solve via simulated annealing.

    Uses single-bit flip moves and geometric cooling schedule.

    Args:
        prob: PricingProblem instance.
        max_iter: Maximum number of iterations.
        initial_temp: Starting temperature.
        cooling_rate: Multiplicative cooling factor per iteration.
        seed: Random seed for reproducibility.

    Returns:
        (x_best, f_best): Best configuration found and its objective value.
    """
    rng = np.random.default_rng(seed)
    n_bits = prob.n_bits

    # Initialize with random configuration
    x_current = rng.integers(0, 2 ** n_bits)
    f_current = prob.evaluate(x_current)

    x_best = x_current
    f_best = f_current

    temp = initial_temp

    for _ in range(max_iter):
        # Propose move: flip random bit
        bit_to_flip = rng.integers(0, n_bits)
        x_proposal = x_current ^ (1 << bit_to_flip)
        f_proposal = prob.evaluate(x_proposal)

        # Accept or reject
        delta = f_proposal - f_current

        if delta > 0:
            # Better solution: always accept
            accept = True
        else:
            # Worse solution: accept with probability exp(delta/T)
            accept = rng.random() < np.exp(delta / temp)

        if accept:
            x_current = x_proposal
            f_current = f_proposal

            # Update best
            if f_current > f_best:
                x_best = x_current
                f_best = f_current

        # Cool down (with floor to avoid numerical issues)
        temp = max(temp * cooling_rate, 1e-8)

    return x_best, f_best


def cp_sat_solver_result(
    prob: "PricingProblem",
    instance_id: str | None = None,
) -> dict:
    """Solve exactly with CP-SAT and return standardized result schema."""
    return solve_pricing_with_cp_sat(prob, instance_id=instance_id)


def cp_sat_solver(prob: "PricingProblem") -> tuple[int, float]:
    """Backward-compatible CP-SAT API returning ``(x_opt, objective_value)``."""
    result = cp_sat_solver_result(prob)
    if result["status"] not in {"optimal", "feasible"}:
        raise RuntimeError(
            f"CP-SAT failed with status '{result['status']}' "
            f"(raw={result['metadata'].get('raw_status_code')})"
        )
    return int(result["x_opt"]), float(result["objective_value"])


def random_search_solver(
    prob: "PricingProblem",
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> tuple[int, float]:
    """Solve by random sampling.

    Args:
        prob: PricingProblem instance.
        n_samples: Number of random samples.
        seed: Random seed.

    Returns:
        (x_best, f_best): Best configuration found and its objective value.
    """
    rng = np.random.default_rng(seed)
    n_bits = prob.n_bits

    x_best = 0
    f_best = prob.evaluate(0)

    for _ in range(n_samples):
        x = rng.integers(0, 2 ** n_bits)
        f = prob.evaluate(x)
        if f > f_best:
            x_best = x
            f_best = f

    return x_best, f_best


def run_all_baselines(
    prob: "PricingProblem",
    sa_iter: int = 10000,
    random_samples: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """Run all classical baselines and return results.

    Args:
        prob: PricingProblem instance.
        sa_iter: SA iterations.
        random_samples: Random search samples.
        seed: Random seed.

    Returns:
        Dictionary with results for each solver.
    """
    import time

    results = {}

    # Brute force
    t0 = time.perf_counter()
    x_bf, f_bf = brute_force_solver(prob)
    t_bf = time.perf_counter() - t0
    results['brute_force'] = {
        'x': x_bf,
        'f': f_bf,
        'time': t_bf,
        'optimal': True,
    }

    # Helper for approximation ratio (handles negative optima from penalties)
    def approx_ratio(f: float, f_opt: float) -> float:
        if f_opt > 0:
            return f / f_opt
        elif f_opt < 0:
            # Both negative: ratio > 1 means f is closer to 0 (worse)
            return f / f_opt if f < 0 else 0.0
        else:
            return 1.0 if f == 0 else 0.0

    # Simulated annealing
    t0 = time.perf_counter()
    x_sa, f_sa = simulated_annealing_solver(prob, max_iter=sa_iter, seed=seed)
    t_sa = time.perf_counter() - t0
    results['simulated_annealing'] = {
        'x': x_sa,
        'f': f_sa,
        'time': t_sa,
        'approx_ratio': approx_ratio(f_sa, f_bf),
    }

    # CP-SAT (exact, if OR-Tools is available and instance is small enough)
    t0 = time.perf_counter()
    try:
        cp_result = cp_sat_solver_result(prob)
        if cp_result["status"] not in {"optimal", "feasible"}:
            raise RuntimeError(f"CP-SAT failed with status {cp_result['status']}")
        x_cp = int(cp_result["x_opt"])
        f_cp = float(cp_result["objective_value"])
        t_cp = time.perf_counter() - t0
        results['cp_sat'] = {
            'x': x_cp,
            'f': f_cp,
            'objective_value': f_cp,
            'optimized_objective_value': float(cp_result["optimized_objective_value"]),
            'decoded_config': cp_result["decoded_config"],
            'constraint_summary': cp_result["constraint_summary"],
            'instance_id': cp_result["instance_id"],
            'metadata': cp_result["metadata"],
            'time': t_cp,
            'approx_ratio': approx_ratio(f_cp, f_bf),
            'optimal': True,
            'status': 'ok',
            'solver': 'ortools_cp_sat_one_hot_exact',
            'matches_bruteforce_objective': bool(np.isclose(f_cp, f_bf)),
            'matches_bruteforce_x': bool(x_cp == x_bf),
        }
    except (ImportError, ValueError, RuntimeError) as exc:
        t_cp = time.perf_counter() - t0
        results['cp_sat'] = {
            'x': None,
            'f': None,
            'time': t_cp,
            'approx_ratio': None,
            'optimal': False,
            'status': 'unavailable',
            'reason': str(exc),
        }

    # Random search
    t0 = time.perf_counter()
    x_rs, f_rs = random_search_solver(prob, n_samples=random_samples, seed=seed)
    t_rs = time.perf_counter() - t0
    results['random_search'] = {
        'x': x_rs,
        'f': f_rs,
        'time': t_rs,
        'approx_ratio': approx_ratio(f_rs, f_bf),
    }

    return results
