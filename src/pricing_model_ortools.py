"""OR-Tools CP-SAT model for pricing instances.

This module is the single source of truth for CP-SAT solve semantics used by
notebooks and baseline orchestration.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from src.problem_generator import decode_bitstring

if TYPE_CHECKING:
    from src.problem_generator import PricingProblem


def _normalize_status(raw_status: int, cp_model_module) -> str:
    """Map OR-Tools status codes to stable string values."""
    cp_model = cp_model_module
    if raw_status == cp_model.OPTIMAL:
        return "optimal"
    if raw_status == cp_model.FEASIBLE:
        return "feasible"
    if raw_status == cp_model.INFEASIBLE:
        return "infeasible"
    if raw_status == cp_model.MODEL_INVALID:
        return "model_invalid"
    return "error"


def _constraint_summary(prob: "PricingProblem", x_opt: int) -> dict[str, int | bool]:
    """Build compact constraint diagnostics at the selected optimum."""
    tiers = decode_bitstring(x_opt, prob.n_features)
    n_invalid = int(prob.count_invalid(tiers))
    n_bundle_violations = int(prob.bundle_violations(tiers))
    n_luxury = int(prob.count_luxury(tiers))
    return {
        "n_invalid": n_invalid,
        "n_bundle_violations": n_bundle_violations,
        "n_luxury": n_luxury,
        "max_luxury": int(prob.max_luxury),
        "is_feasible": bool(
            n_invalid == 0
            and n_bundle_violations == 0
            and n_luxury <= int(prob.max_luxury)
        ),
    }


def solve_pricing_with_cp_sat(
    prob: "PricingProblem",
    instance_id: str | None = None,
) -> dict:
    """Solve a pricing instance with one-hot CP-SAT and return standard contract.

    Contract semantics:
    - ``objective_value`` is always raw pricing objective ``F(x_opt)``.
    - ``optimized_objective_value`` is the exact scalar optimized by CP-SAT.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError(
            "OR-Tools is required for pricing CP-SAT modeling. Install with: pip install ortools"
        ) from exc

    table = prob.build_truth_table()
    unique_vals = np.unique(table)
    ranks = np.searchsorted(unique_vals, table).astype(np.int64)

    model = cp_model.CpModel()
    selectors = [model.NewBoolVar(f"select_{x}") for x in range(prob.n_configurations)]
    model.Add(sum(selectors) == 1)
    model.Maximize(sum(int(ranks[x]) * selectors[x] for x in range(prob.n_configurations)))

    solver = cp_model.CpSolver()
    t0 = time.perf_counter()
    raw_status = solver.Solve(model)
    solve_time_s = time.perf_counter() - t0

    status = _normalize_status(raw_status, cp_model)
    base = {
        "status": status,
        "objective_value": None,
        "optimized_objective_value": None,
        "x_opt": None,
        "decoded_config": None,
        "solve_time_s": float(solve_time_s),
        "constraint_summary": None,
        "instance_id": instance_id,
        "metadata": {
            "solver_name": "ortools_cp_sat_one_hot_exact",
            "raw_status_code": int(raw_status),
            "objective_encoding": "rank_by_unique_F",
            "model_type": "one_hot_exact",
            "n_variables": int(prob.n_configurations),
            "n_decision_variables": int(prob.n_configurations),
        },
    }

    if status not in {"optimal", "feasible"}:
        return base

    x_opt = int(next(x for x in range(prob.n_configurations) if solver.Value(selectors[x]) == 1))
    objective_value = float(table[x_opt])

    base.update(
        {
            "objective_value": objective_value,
            "optimized_objective_value": float(solver.ObjectiveValue()),
            "x_opt": x_opt,
            "decoded_config": prob.solution_summary(x_opt),
            "constraint_summary": _constraint_summary(prob, x_opt),
        }
    )
    return base
