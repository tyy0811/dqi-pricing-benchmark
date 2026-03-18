"""Stage E regression tests for explicit schema validation."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from tests._stage_e_fixtures import stage_e_base_row


def _valid_master_row() -> dict:
    return {
        "run_id": "r1",
        "manifest_slot": "slot-1",
        "instance_id": "S1",
        "family": "S",
        "encoding": "synthetic_structured_bv",
        "decoder": "bp1",
        "alpha_mode": "paper",
        "execution_mode": "mixture",
        "run_status": "completed",
        "best_F": 10.0,
        "topk_regret": 0.1,
        "decision_distortion": 0.2,
        "spearman_rho_F_G": 0.8,
        "retained_energy_eta": 0.6,
        "qubits": 5,
        "gates": 120,
        "depth": 35,
        "resource_status": "comparable",
        "resource_confidence": "implementation_proxy",
        "has_verified_provenance": True,
    }


def _valid_unavailable_row() -> dict:
    row = stage_e_base_row()
    return {
        "run_id": row["run_id"],
        "instance_id": row["instance_id"],
        "family": row["family"],
        "encoding": row["encoding"],
        "decoder": row["decoder"],
        "alpha_mode": row["alpha_mode"],
        "execution_mode": row["execution_mode"],
        "unavailable_class": "missing_quality_fields",
        "unavailable_reason": "missing quality fields: topk_regret",
        "run_status": "completed",
        "resource_status": "comparable",
        "has_verified_provenance": True,
        "missing_required_fields": [],
        "missing_quality_fields": ["topk_regret"],
        "missing_resource_fields": [],
    }


def test_invalid_unavailable_class_fails() -> None:
    from src.report_schema import validate_quality_cost_unavailable_row

    row = _valid_unavailable_row()
    row["unavailable_class"] = "invalid_class"

    try:
        validate_quality_cost_unavailable_row(row)
    except ValueError as exc:
        assert "unavailable_class" in str(exc)
    else:
        raise AssertionError("expected invalid unavailable_class to fail")


def test_master_missing_required_field_fails() -> None:
    from src.report_schema import validate_quality_cost_master_row

    row = _valid_master_row()
    row.pop("best_F")

    try:
        validate_quality_cost_master_row(row)
    except ValueError as exc:
        assert "missing required fields" in str(exc) or "best_F" in str(exc)
    else:
        raise AssertionError("expected master validator to fail on missing required field")


def test_unavailable_missing_reason_fails() -> None:
    from src.report_schema import validate_quality_cost_unavailable_row

    row = _valid_unavailable_row()
    row.pop("unavailable_reason")

    try:
        validate_quality_cost_unavailable_row(row)
    except ValueError as exc:
        assert "missing required fields" in str(exc) or "unavailable_reason" in str(exc)
    else:
        raise AssertionError("expected unavailable validator to fail when reason is missing")
