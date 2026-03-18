"""Tests for the fixed 3-period CP-SAT pricing solver."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.pricing_model_ortools_multiperiod import solve_multiperiod_pricing_with_cp_sat
from tests._multiperiod_helpers import make_tiny_multiperiod_problem


def test_multiperiod_cp_sat_returns_compact_contract():
    prob = make_tiny_multiperiod_problem()
    out = solve_multiperiod_pricing_with_cp_sat(prob, instance_id="tiny_3period")

    assert out["status"] in {"optimal", "feasible"}
    assert out["horizon_revenue"] is not None
    assert len(out["per_period_configuration"]) == 3
    assert out["capacity_usage"]["inventory_consumed_per_period"]
    assert out["capacity_usage"]["remaining_inventory_after_period"]
    assert out["solve_time_s"] >= 0.0
