"""Stage E canonical master/unavailable builder tests."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


def _base_row() -> dict:
    return {
        "run_id": "run_1",
        "manifest_slot": "slot-1",
        "matrix": "matrix_c",
        "stage": "decoder_study",
        "family": "S",
        "instance_id": "S1",
        "encoding": "synthetic_structured_bv",
        "decoder": "bp1",
        "alpha_mode": "paper",
        "execution_mode": "mixture",
        "trial_seed": 7,
        "canonical_trial_seed": 7,
        "seed": 7,
        "run_status": "completed",
        "status_reason_code": "completed",
        "status_reason": "completed",
        "has_verified_provenance": True,
        "best_F": 10.5,
        "topk_regret": 0.1,
        "decision_distortion": 0.2,
        "spearman_rho_F_G": 0.8,
        "retained_energy_eta": 0.6,
        "resource_total_qubits_est_with_decode": 5,
        "resource_total_gate_est_with_decode": 120,
        "resource_total_depth_est_with_decode": 35,
        "resource_decode_uncompute_confidence": "implementation_proxy",
    }


def test_builder_returns_master_unavailable_and_summary() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    winner = _base_row()
    loser = _base_row()
    loser["run_id"] = "run_2"
    loser["manifest_slot"] = "slot-2"

    out = build_stage_e_quality_cost_tables([loser, winner])
    assert set(out.keys()) == {"quality_cost_master", "quality_cost_unavailable", "summary"}
    assert len(out["quality_cost_master"]) == 1
    assert len(out["quality_cost_unavailable"]) == 1
    assert out["quality_cost_unavailable"][0]["unavailable_class"] == "duplicate_key"
    assert out["summary"]["admitted_count"] == 1
    assert out["summary"]["duplicate_count"] == 1
    assert out["summary"]["unavailable_count_by_class"]["duplicate_key"] == 1


def test_builder_routes_missing_resource_fields_with_field_list() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    row = _base_row()
    row["resource_total_gate_est_with_decode"] = None

    out = build_stage_e_quality_cost_tables([row])
    assert out["quality_cost_master"] == []
    assert len(out["quality_cost_unavailable"]) == 1
    unr = out["quality_cost_unavailable"][0]
    assert unr["unavailable_class"] == "missing_resource_fields"
    assert "gates" in unr["missing_resource_fields"]


def test_schema_validators_accept_builder_rows() -> None:
    from src.report_schema import (
        validate_quality_cost_master_row,
        validate_quality_cost_unavailable_row,
    )
    from src.resources_compare import build_stage_e_quality_cost_tables

    winner = _base_row()
    loser = _base_row()
    loser["run_id"] = "run_2"
    loser["manifest_slot"] = "slot-2"

    out = build_stage_e_quality_cost_tables([winner, loser])
    validate_quality_cost_master_row(out["quality_cost_master"][0])
    validate_quality_cost_unavailable_row(out["quality_cost_unavailable"][0])


def test_unavailable_schema_validator_rejects_unknown_class() -> None:
    from src.report_schema import validate_quality_cost_unavailable_row

    row = {
        "run_id": "r",
        "instance_id": "S1",
        "family": "S",
        "encoding": "enc",
        "decoder": "bp1",
        "alpha_mode": "paper",
        "execution_mode": "mixture",
        "unavailable_class": "unknown_class",
        "unavailable_reason": "x",
        "run_status": "failed",
        "resource_status": "not_comparable",
        "has_verified_provenance": False,
        "missing_required_fields": [],
        "missing_quality_fields": [],
        "missing_resource_fields": [],
    }

    try:
        validate_quality_cost_unavailable_row(row)
    except ValueError as exc:
        assert "unavailable_class" in str(exc)
    else:
        raise AssertionError("expected unavailable validator to reject unknown class")


def test_builder_ignores_boundary_only_rows_for_comparable_master() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    row = _base_row()
    row["run_status"] = "unavailable"
    row["boundary_reason"] = "qubit_count_exceeded"
    row["attempted_execution"] = False
    row["estimated_qubits"] = 31
    row["estimated_memory_gb"] = 16.0
    row["wall_clock_s"] = 0.0
    row["peak_memory_mb"] = 0.0

    out = build_stage_e_quality_cost_tables([row])
    assert out["quality_cost_master"] == []
    assert len(out["quality_cost_unavailable"]) == 1
