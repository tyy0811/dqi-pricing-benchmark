"""Tests for the fixed 3-period baseline wrappers."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.classical_baselines_multiperiod import (
    build_multiperiod_scenario_matrix,
    run_multiperiod_baselines,
)
from tests._multiperiod_helpers import (
    make_identical_multiplier_multiperiod_problem,
    make_tiny_multiperiod_problem,
)


def test_multiperiod_baselines_compare_static_and_dynamic():
    prob = make_tiny_multiperiod_problem()
    out = run_multiperiod_baselines(prob)

    assert set(out) == {
        "dynamic_cp_sat",
        "static_fixed_config",
        "dynamic_horizon_revenue",
        "static_horizon_revenue",
        "static_vs_dynamic_delta",
        "dynamic_per_period_configuration",
        "static_configuration",
        "capacity_usage",
        "remaining_inventory",
        "solve_time_s",
        "status",
    }
    assert len(out["dynamic_cp_sat"]["per_period_configuration"]) == 3
    assert len(out["static_fixed_config"]["per_period_configuration"]) == 3
    assert (
        out["static_fixed_config"]["per_period_configuration"][0]["x"]
        == out["static_fixed_config"]["per_period_configuration"][1]["x"]
    )
    assert out["dynamic_horizon_revenue"] >= out["static_horizon_revenue"]
    assert out["static_vs_dynamic_delta"] == (
        out["dynamic_horizon_revenue"] - out["static_horizon_revenue"]
    )
    assert out["dynamic_horizon_revenue"] == out["dynamic_cp_sat"]["horizon_revenue"]
    assert out["static_horizon_revenue"] == out["static_fixed_config"]["horizon_revenue"]
    assert out["dynamic_per_period_configuration"] == out["dynamic_cp_sat"]["per_period_configuration"]
    assert out["static_configuration"] == out["static_fixed_config"]["configuration"]
    assert out["capacity_usage"] == out["dynamic_cp_sat"]["capacity_usage"]
    assert out["solve_time_s"] == out["dynamic_cp_sat"]["solve_time_s"]
    assert out["status"] == out["dynamic_cp_sat"]["status"]
    assert (
        out["remaining_inventory"]
        == out["dynamic_cp_sat"]["capacity_usage"]["remaining_inventory_after_period"][-1]
    )


def test_multiperiod_baselines_match_when_period_multipliers_are_identical():
    prob = make_identical_multiplier_multiperiod_problem()
    out = run_multiperiod_baselines(prob)

    assert out["dynamic_horizon_revenue"] == out["static_horizon_revenue"]
    assert out["static_vs_dynamic_delta"] == 0.0


def test_multiperiod_baselines_respect_inventory_and_capacity_limits():
    prob = make_tiny_multiperiod_problem()
    out = run_multiperiod_baselines(prob)

    dynamic_cfg = out["dynamic_per_period_configuration"]
    static_cfg = out["static_fixed_config"]["per_period_configuration"]
    for cfgs in [dynamic_cfg, static_cfg]:
        total_inventory = 0
        for period_idx, cfg in enumerate(cfgs):
            used = int(cfg["inventory_consumed"])
            total_inventory += used
            assert used <= prob.per_period_capacity_caps[period_idx]
        assert total_inventory <= prob.initial_inventory


def test_multiperiod_scenario_matrix_returns_small_reviewer_facing_rows():
    rows = build_multiperiod_scenario_matrix()

    assert 6 <= len(rows) <= 9
    required = {
        "scenario_label",
        "demand_shift",
        "inventory_regime",
        "smoothing_penalty",
        "dynamic_horizon_revenue",
        "static_horizon_revenue",
        "static_vs_dynamic_delta",
        "remaining_inventory",
        "inventory_usage_summary",
        "dynamic_changes_across_periods",
        "delta_reduction_vs_zero_smoothing",
        "choices_change_vs_zero_smoothing",
        "inventory_used",
        "verdict",
    }
    assert required <= set(rows[0])
    assert any(row["static_vs_dynamic_delta"] > 0 for row in rows)
    assert any(abs(row["static_vs_dynamic_delta"]) < 1e-9 for row in rows)
    assert any(row["smoothing_penalty"] > 0 for row in rows)
    assert all(isinstance(row["inventory_used"], int) for row in rows)
    assert all(isinstance(row["verdict"], str) and row["verdict"] for row in rows)
    assert any(
        row["delta_reduction_vs_zero_smoothing"] > 0
        or row["choices_change_vs_zero_smoothing"]
        for row in rows
        if row["smoothing_penalty"] > 0
    )
