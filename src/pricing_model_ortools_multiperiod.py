"""OR-Tools CP-SAT solver for the fixed 3-period pricing extension."""

from __future__ import annotations

import time


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


def solve_multiperiod_pricing_with_cp_sat(
    prob,
    instance_id: str | None = None,
) -> dict:
    """Solve a fixed 3-period pricing horizon with CP-SAT."""
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError(
            "OR-Tools is required for multiperiod pricing CP-SAT modeling. "
            "Install with: pip install ortools"
        ) from exc

    scale = 1000
    period_tables = [prob.build_period_truth_table(period_idx) for period_idx in range(3)]

    model = cp_model.CpModel()
    selectors: list[list] = []
    for period_idx in range(3):
        period_selectors = [
            model.NewBoolVar(f"select_p{period_idx}_x{x}")
            for x in range(prob.n_configurations)
        ]
        model.Add(sum(period_selectors) == 1)
        selectors.append(period_selectors)

    total_inventory_terms = []
    for period_idx in range(3):
        if prob.per_period_capacity_caps is not None:
            model.Add(
                sum(
                    int(prob.inventory_consumption_by_config[x]) * selectors[period_idx][x]
                    for x in range(prob.n_configurations)
                ) <= int(prob.per_period_capacity_caps[period_idx])
            )
        total_inventory_terms.extend(
            int(prob.inventory_consumption_by_config[x]) * selectors[period_idx][x]
            for x in range(prob.n_configurations)
        )
    model.Add(sum(total_inventory_terms) <= int(prob.initial_inventory))

    reward_terms = []
    for period_idx in range(3):
        reward_terms.extend(
            int(round(float(period_tables[period_idx][x]) * scale)) * selectors[period_idx][x]
            for x in range(prob.n_configurations)
        )

    smoothing_terms = []
    smoothing_scaled = int(round(float(prob.smoothing_penalty) * scale))
    if smoothing_scaled > 0:
        for period_idx in range(2):
            same_choice_vars = []
            for x in range(prob.n_configurations):
                same_choice = model.NewBoolVar(f"same_p{period_idx}_x{x}")
                model.Add(same_choice <= selectors[period_idx][x])
                model.Add(same_choice <= selectors[period_idx + 1][x])
                model.Add(same_choice >= selectors[period_idx][x] + selectors[period_idx + 1][x] - 1)
                same_choice_vars.append(same_choice)
            change_indicator = model.NewBoolVar(f"change_after_p{period_idx}")
            model.Add(change_indicator + sum(same_choice_vars) == 1)
            smoothing_terms.append(smoothing_scaled * change_indicator)

    model.Maximize(sum(reward_terms) - sum(smoothing_terms))

    solver = cp_model.CpSolver()
    t0 = time.perf_counter()
    raw_status = solver.Solve(model)
    solve_time_s = time.perf_counter() - t0

    status = _normalize_status(raw_status, cp_model)
    out = {
        "status": status,
        "instance_id": instance_id,
        "horizon_revenue": None,
        "per_period_configuration": [],
        "remaining_inventory": None,
        "smoothing_penalty_total": None,
        "capacity_usage": {
            "inventory_consumed_per_period": [],
            "remaining_inventory_after_period": [],
        },
        "solve_time_s": float(solve_time_s),
        "objective_value_scaled": None,
        "metadata": {
            "solver_name": "ortools_cp_sat_multiperiod_one_hot_exact",
            "raw_status_code": int(raw_status),
            "n_periods": 3,
            "objective_scale": scale,
        },
    }
    if status not in {"optimal", "feasible"}:
        return out

    selected = []
    for period_idx in range(3):
        chosen = next(
            x for x in range(prob.n_configurations)
            if solver.Value(selectors[period_idx][x]) == 1
        )
        selected.append(int(chosen))

    evaluated = prob.evaluate_horizon(selected)
    out.update(
        {
            "horizon_revenue": float(evaluated["horizon_revenue"]),
            "per_period_configuration": evaluated["per_period_configuration"],
            "remaining_inventory": evaluated["capacity_usage"]["remaining_inventory_after_period"][-1],
            "smoothing_penalty_total": float(evaluated["smoothing_penalty_total"]),
            "capacity_usage": evaluated["capacity_usage"],
            "objective_value_scaled": float(solver.ObjectiveValue()),
        }
    )
    return out
