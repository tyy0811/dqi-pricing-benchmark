"""Tests for OR-Tools pricing model contract."""

from __future__ import annotations

import pytest

from src.problem_generator import scaling_instance, small_instance
from src.pricing_model_ortools import solve_pricing_with_cp_sat


def test_cp_sat_result_schema_when_available():
    pytest.importorskip("ortools.sat.python.cp_model")

    prob = small_instance(3)
    result = solve_pricing_with_cp_sat(prob, instance_id="pricing_3feat_6bit")

    required = {
        "status",
        "objective_value",
        "optimized_objective_value",
        "x_opt",
        "decoded_config",
        "solve_time_s",
        "constraint_summary",
        "instance_id",
        "metadata",
    }
    assert required.issubset(result.keys())
    assert result["status"] in {"optimal", "feasible"}


def test_cp_sat_objective_semantics():
    pytest.importorskip("ortools.sat.python.cp_model")

    prob = small_instance(3)
    result = solve_pricing_with_cp_sat(prob)

    assert result["objective_value"] == prob.evaluate(result["x_opt"])
    assert result["optimized_objective_value"] >= 0
    assert result["metadata"]["solver_name"] == "ortools_cp_sat_one_hot_exact"


def test_cp_sat_instance_id_is_passthrough():
    pytest.importorskip("ortools.sat.python.cp_model")

    prob = small_instance(3)
    result = solve_pricing_with_cp_sat(prob, instance_id="demo-id")
    assert result["instance_id"] == "demo-id"


@pytest.mark.parametrize(
    ("n_features", "instance_id"),
    [
        (6, "pricing_6feat_12bit"),
        (7, "pricing_7feat_14bit"),
    ],
)
def test_cp_sat_supports_p6_and_p7_exactly(n_features: int, instance_id: str):
    pytest.importorskip("ortools.sat.python.cp_model")

    prob = scaling_instance(n_features)
    x_bf, f_bf = prob.brute_force_solve()
    result = solve_pricing_with_cp_sat(prob, instance_id=instance_id)

    assert result["status"] in {"optimal", "feasible"}
    assert abs(result["objective_value"] - f_bf) < 1e-9
    assert prob.evaluate(result["x_opt"]) == f_bf
    assert prob.evaluate(x_bf) == f_bf
    assert result["metadata"]["solver_name"] == "ortools_cp_sat_one_hot_exact"
    assert result["metadata"]["model_type"] == "one_hot_exact"
    assert result["metadata"]["n_decision_variables"] == prob.n_configurations
