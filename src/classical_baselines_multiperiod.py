"""Classical baselines for the fixed 3-period pricing extension."""

from __future__ import annotations

import numpy as np

from src.problem_generator import (
    BundleConstraint,
    CustomerSegment,
    Feature,
    TIER_PREMIUM,
)
from src.problem_generator_multiperiod import MultiPeriodPricingProblem
from src.pricing_model_ortools_multiperiod import solve_multiperiod_pricing_with_cp_sat


def solve_static_multiperiod_baseline(prob) -> dict:
    """Return the best fixed configuration repeated across all 3 periods."""
    best = None
    for x in range(prob.n_configurations):
        evaluated = prob.evaluate_horizon([x, x, x])
        candidate = {"x": int(x), **evaluated}
        if best is None or (
            candidate["horizon_revenue"] > best["horizon_revenue"]
            or (
                candidate["horizon_revenue"] == best["horizon_revenue"]
                and candidate["x"] < best["x"]
            )
        ):
            best = candidate

    return {
        "status": "ok",
        "horizon_revenue": float(best["horizon_revenue"]),
        "per_period_configuration": best["per_period_configuration"],
        "capacity_usage": best["capacity_usage"],
        "remaining_inventory": best["capacity_usage"]["remaining_inventory_after_period"][-1],
        "smoothing_penalty_total": float(best.get("smoothing_penalty_total", 0.0)),
        "configuration": int(best["x"]),
        "solve_time_s": 0.0,
    }


def run_multiperiod_baselines(prob, **kwargs) -> dict:
    """Run the minimal static-vs-dynamic baseline set."""
    dynamic = solve_multiperiod_pricing_with_cp_sat(prob, **kwargs)
    static = solve_static_multiperiod_baseline(prob)
    remaining_inventory = dynamic.get("remaining_inventory")
    if remaining_inventory is None:
        remaining_after_period = (
            dynamic.get("capacity_usage", {}).get("remaining_inventory_after_period", [])
        )
        remaining_inventory = remaining_after_period[-1] if remaining_after_period else None

    return {
        "dynamic_cp_sat": dynamic,
        "static_fixed_config": static,
        "dynamic_horizon_revenue": dynamic.get("horizon_revenue"),
        "static_horizon_revenue": static.get("horizon_revenue"),
        "static_vs_dynamic_delta": (
            None
            if dynamic.get("horizon_revenue") is None or static.get("horizon_revenue") is None
            else float(dynamic["horizon_revenue"] - static["horizon_revenue"])
        ),
        "dynamic_per_period_configuration": dynamic.get("per_period_configuration"),
        "static_configuration": static.get("configuration"),
        "capacity_usage": dynamic.get("capacity_usage"),
        "remaining_inventory": remaining_inventory,
        "solve_time_s": dynamic.get("solve_time_s"),
        "status": dynamic.get("status"),
    }


def _default_multiperiod_components() -> dict:
    """Return the shared tiny reviewer-facing 3-period pricing components."""
    return {
        "features": [
            Feature("Seats", 400, 700, 1000),
            Feature("Roof", 300, 600, 900),
        ],
        "segments": [
            CustomerSegment("Budget", 900, 0.6),
            CustomerSegment("Premium", 1500, 0.4),
        ],
        "bundle_constraints": [
            BundleConstraint(
                name="Premium roof requires premium seats",
                feature_a=1,
                tier_a=TIER_PREMIUM,
                feature_b=0,
                allowed_tiers_b={TIER_PREMIUM, 2},
            )
        ],
        "inventory_consumption": np.array([1, 1], dtype=np.int64),
    }


def make_multiperiod_scenario_problem(
    *,
    demand_shift: str,
    inventory_regime: str,
    smoothing_penalty: float,
) -> MultiPeriodPricingProblem:
    """Build one curated 3-period pricing scenario for reviewer-facing comparison."""
    demand_shift_multipliers = {
        "low": np.array(
            [
                [1.0, 1.0],
                [1.05, 0.95],
                [0.95, 1.05],
            ],
            dtype=np.float64,
        ),
        "medium": np.array(
            [
                [1.0, 1.0],
                [1.2, 0.8],
                [0.7, 1.3],
            ],
            dtype=np.float64,
        ),
        "high": np.array(
            [
                [1.0, 1.0],
                [1.4, 0.6],
                [0.5, 1.5],
            ],
            dtype=np.float64,
        ),
    }
    inventory_regimes = {
        "tight": {"initial_inventory": 2, "per_period_capacity_caps": [1, 1, 1]},
        "buffered": {"initial_inventory": 3, "per_period_capacity_caps": [2, 2, 2]},
        "loose": {"initial_inventory": 4, "per_period_capacity_caps": [2, 2, 2]},
    }

    components = _default_multiperiod_components()
    return MultiPeriodPricingProblem(
        features=components["features"],
        segments=components["segments"],
        bundle_constraints=components["bundle_constraints"],
        period_segment_multipliers=demand_shift_multipliers[demand_shift],
        initial_inventory=inventory_regimes[inventory_regime]["initial_inventory"],
        inventory_consumption=components["inventory_consumption"],
        per_period_capacity_caps=inventory_regimes[inventory_regime]["per_period_capacity_caps"],
        smoothing_penalty=float(smoothing_penalty),
    )


def build_multiperiod_scenario_matrix() -> list[dict]:
    """Return a compact reviewer-facing scenario matrix for Notebook 09."""
    selected_scenarios = [
        ("low", "tight", 0.0),
        ("low", "buffered", 5.0),
        ("medium", "tight", 0.0),
        ("medium", "tight", 20.0),
        ("medium", "buffered", 0.0),
        ("medium", "loose", 5.0),
        ("high", "tight", 0.0),
        ("high", "tight", 5.0),
        ("high", "buffered", 20.0),
    ]

    zero_smoothing_reference: dict[tuple[str, str], dict] = {}
    for demand_shift, inventory_regime in {
        (demand_shift, inventory_regime)
        for demand_shift, inventory_regime, _ in selected_scenarios
    }:
        prob = make_multiperiod_scenario_problem(
            demand_shift=demand_shift,
            inventory_regime=inventory_regime,
            smoothing_penalty=0.0,
        )
        zero_smoothing_reference[(demand_shift, inventory_regime)] = run_multiperiod_baselines(prob)

    rows: list[dict] = []
    for demand_shift, inventory_regime, smoothing_penalty in selected_scenarios:
        prob = make_multiperiod_scenario_problem(
            demand_shift=demand_shift,
            inventory_regime=inventory_regime,
            smoothing_penalty=smoothing_penalty,
        )
        baselines = run_multiperiod_baselines(prob)
        baseline_zero = zero_smoothing_reference[(demand_shift, inventory_regime)]
        dynamic_cfg = baselines["dynamic_per_period_configuration"]
        dynamic_x = [int(cfg["x"]) for cfg in dynamic_cfg]
        zero_dynamic_x = [
            int(cfg["x"]) for cfg in baseline_zero["dynamic_per_period_configuration"]
        ]
        remaining_after = baselines["capacity_usage"]["remaining_inventory_after_period"]
        consumed = baselines["capacity_usage"]["inventory_consumed_per_period"]
        inventory_used = int(sum(int(value) for value in consumed))

        if baselines["static_vs_dynamic_delta"] <= 1e-9:
            verdict = "near-static-match"
        elif len(set(dynamic_x)) > 1:
            verdict = "dynamic-uplift-via-package-change"
        else:
            verdict = "dynamic-uplift-without-package-change"

        rows.append(
            {
                "scenario_label": (
                    f"{demand_shift}-{inventory_regime}-smooth-{int(smoothing_penalty)}"
                ),
                "demand_shift": demand_shift,
                "inventory_regime": inventory_regime,
                "smoothing_penalty": float(smoothing_penalty),
                "dynamic_horizon_revenue": float(baselines["dynamic_horizon_revenue"]),
                "static_horizon_revenue": float(baselines["static_horizon_revenue"]),
                "static_vs_dynamic_delta": float(baselines["static_vs_dynamic_delta"]),
                "remaining_inventory": int(baselines["remaining_inventory"]),
                "inventory_used": inventory_used,
                "inventory_usage_summary": (
                    f"used {consumed} -> remaining {remaining_after}"
                ),
                "dynamic_changes_across_periods": len(set(dynamic_x)) > 1,
                "dynamic_choice_pattern": " -> ".join(str(x) for x in dynamic_x),
                "delta_reduction_vs_zero_smoothing": float(
                    baseline_zero["static_vs_dynamic_delta"] - baselines["static_vs_dynamic_delta"]
                ),
                "choices_change_vs_zero_smoothing": dynamic_x != zero_dynamic_x,
                "verdict": verdict,
            }
        )

    return rows
