"""Stage E resource comparability normalization tests."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


def test_normalize_resources_emits_qubits_gates_depth_fields() -> None:
    from src.resources_compare import normalize_resource_row_for_stage_e

    row = {
        "resource_total_qubits_est_with_decode": 12,
        "resource_total_gate_est_with_decode": 345,
        "resource_total_depth_est_with_decode": 89,
        "resource_decode_uncompute_confidence": "implementation_proxy",
    }

    out = normalize_resource_row_for_stage_e(row)
    assert out["qubits"] == 12
    assert out["gates"] == 345
    assert out["depth"] == 89
    assert out["resource_status"] == "comparable"


def test_estimated_comparable_requires_comparability_basis() -> None:
    from src.resources_compare import normalize_resource_row_for_stage_e

    row = {
        "resource_total_qubits_est_with_decode": 8,
        "resource_total_gate_est_with_decode": 111,
        "resource_total_depth_est_with_decode": 22,
        "resource_decode_uncompute_confidence": "analytic_upper_bound",
        "resource_comparability_basis": "invalid_basis",
    }

    out = normalize_resource_row_for_stage_e(row)
    assert out["resource_status"] == "not_comparable"
    assert out["resource_not_comparable_reason"] == "missing_comparability_basis"


def test_non_comparable_resource_status_flagged() -> None:
    from src.resources_compare import normalize_resource_row_for_stage_e

    row = {
        "resource_total_gate_est_with_decode": None,
        "resource_total_depth_est_with_decode": 10,
        "resource_total_qubits_est_with_decode": 4,
        "resource_decode_uncompute_confidence": "analytic_upper_bound",
        "resource_comparability_basis": "same_estimator",
    }

    out = normalize_resource_row_for_stage_e(row)
    assert out["resource_status"] == "not_comparable"
    assert out["resource_not_comparable_reason"] == "missing_resource_fields"


def _stage_e_base_row() -> dict:
    return {
        "run_id": "r1",
        "manifest_slot": "slot-1",
        "matrix": "matrix_a",
        "stage": "benchmark_matrix",
        "family": "P",
        "instance_id": "P3",
        "encoding": "wht_truncated",
        "decoder": "bp1",
        "alpha_mode": "paper",
        "execution_mode": "mixture",
        "trial_seed": 42,
        "canonical_trial_seed": 42,
        "seed": 42,
        "run_status": "completed",
        "status_reason_code": "completed",
        "status_reason": "completed",
        "has_verified_provenance": True,
        "best_F": 12.0,
        "topk_regret": 0.5,
        "decision_distortion": 0.2,
        "spearman_rho_F_G": 0.93,
        "retained_energy_eta": 0.81,
        "resource_total_qubits_est_with_decode": 8,
        "resource_total_gate_est_with_decode": 123,
        "resource_total_depth_est_with_decode": 45,
        "resource_decode_uncompute_confidence": "implementation_proxy",
    }


def test_stage_e_required_quality_field_set_is_explicit() -> None:
    from src.resources_compare import STAGE_E_REQUIRED_QUALITY_FIELDS

    assert tuple(STAGE_E_REQUIRED_QUALITY_FIELDS) == (
        "best_F",
        "topk_regret",
        "decision_distortion",
        "spearman_rho_F_G",
        "retained_energy_eta",
    )


def test_normalize_stage_e_candidate_row_admits_completed_with_canonical_status() -> None:
    from src.resources_compare import normalize_stage_e_candidate_row

    row = _stage_e_base_row()
    row["run_status"] = "ok"

    out = normalize_stage_e_candidate_row(row)
    assert out["identity_fields"]["run_status"] == "completed"
    assert out["admission"]["admitted_to_master"] is True
    assert out["admission"]["route_to_unavailable"] is False
    assert out["unavailable_row_candidate"] is None


def test_paired_encoding_validation_is_opt_in_only() -> None:
    from src.resources_compare import normalize_stage_e_candidate_row
    from src.verify_encoding import PAIR_VALIDATION_FLAG

    row = _stage_e_base_row()
    row[PAIR_VALIDATION_FLAG] = False
    out = normalize_stage_e_candidate_row(row, paired_artifact=None)
    assert out["admission"]["admitted_to_master"] is True

    row2 = _stage_e_base_row()
    row2[PAIR_VALIDATION_FLAG] = True
    out2 = normalize_stage_e_candidate_row(row2, paired_artifact=None)
    assert out2["admission"]["admitted_to_master"] is False
    assert out2["admission"]["unavailable_class"] == "encoding_validation_failed"


def test_insufficient_provenance_routes_to_unavailable() -> None:
    from src.resources_compare import normalize_stage_e_candidate_row

    row = _stage_e_base_row()
    row["has_verified_provenance"] = False
    row["surrogate_best_vs_true_best_sampled_gap"] = None
    row["best_top_G_sampled_F"] = None
    row["best_sampled_F"] = None

    out = normalize_stage_e_candidate_row(row)
    assert out["admission"]["admitted_to_master"] is False
    assert out["admission"]["unavailable_class"] == "insufficient_provenance"
    assert out["unavailable_row_candidate"] is not None


def test_estimated_comparable_without_basis_routes_unavailable() -> None:
    from src.resources_compare import normalize_stage_e_candidate_row

    row = _stage_e_base_row()
    row["resource_decode_uncompute_confidence"] = "analytic_upper_bound"
    row["resource_comparability_basis"] = "invalid_basis"

    out = normalize_stage_e_candidate_row(row)
    assert out["admission"]["admitted_to_master"] is False
    assert out["admission"]["unavailable_class"] == "resource_not_comparable"
    assert out["admission"]["unavailable_reason_code"] == "missing_comparability_basis"


def test_manifest_slot_is_required_for_stage_e_admission() -> None:
    from src.resources_compare import normalize_stage_e_candidate_row

    row = _stage_e_base_row()
    row["manifest_slot"] = None

    out = normalize_stage_e_candidate_row(row)
    assert out["admission"]["admitted_to_master"] is False
    assert out["admission"]["unavailable_class"] == "missing_required_fields"
    assert "manifest_slot" in out["admission"]["missing_identity_fields"]


def test_stage_e_duplicate_key_definition_is_locked() -> None:
    from src.resources_compare import STAGE_E_DUPLICATE_KEY_FIELDS, stage_e_duplicate_key

    row = _stage_e_base_row()
    assert tuple(STAGE_E_DUPLICATE_KEY_FIELDS) == (
        "instance_id",
        "encoding",
        "decoder",
        "alpha_mode",
        "execution_mode",
    )
    assert stage_e_duplicate_key(row) == ("P3", "wht_truncated", "bp1", "paper", "mixture")


def test_partition_stage_e_rows_deduplicates_deterministically() -> None:
    from src.resources_compare import partition_stage_e_quality_cost_rows

    winner = _stage_e_base_row()
    winner.update(
        {
            "run_id": "run_a",
            "manifest_slot": "slot-1",
            "has_verified_provenance": True,
            "run_status": "completed",
        }
    )
    loser = _stage_e_base_row()
    loser.update(
        {
            "run_id": "run_b",
            "manifest_slot": "slot-2",
            "has_verified_provenance": True,
            "run_status": "completed",
        }
    )

    master, unavailable = partition_stage_e_quality_cost_rows([loser, winner])
    assert len(master) == 1
    assert master[0]["run_id"] == "run_a"
    assert len(unavailable) == 1
    assert unavailable[0]["run_id"] == "run_b"
    assert unavailable[0]["unavailable_class"] == "duplicate_key"
    assert unavailable[0]["unavailable_reason_code"] == "duplicate_key"


def test_partition_stage_e_rows_routes_non_completed_to_unavailable() -> None:
    from src.resources_compare import partition_stage_e_quality_cost_rows

    row = _stage_e_base_row()
    row["run_status"] = "failed"
    row["status_reason_code"] = "upstream_failed"

    master, unavailable = partition_stage_e_quality_cost_rows([row])
    assert master == []
    assert len(unavailable) == 1
    assert unavailable[0]["unavailable_class"] == "missing_required_fields"
    assert unavailable[0]["unavailable_reason_code"] == "upstream_failed"


def test_partition_stage_e_rows_routes_missing_resource_values_to_missing_resource_fields() -> None:
    from src.resources_compare import partition_stage_e_quality_cost_rows

    row = _stage_e_base_row()
    row["resource_total_gate_est_with_decode"] = None

    master, unavailable = partition_stage_e_quality_cost_rows([row])
    assert master == []
    assert len(unavailable) == 1
    assert unavailable[0]["unavailable_class"] == "missing_resource_fields"
    assert unavailable[0]["unavailable_reason_code"] == "missing_resource_fields"
