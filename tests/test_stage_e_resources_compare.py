"""Stage E regression tests for admission/routing and paired/provenance scope."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from tests._stage_e_fixtures import invalid_paired_artifact, stage_e_base_row


def test_comparable_row_is_admitted() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    row = stage_e_base_row()
    row["resource_status"] = "comparable"

    out = build_stage_e_quality_cost_tables([row])
    assert len(out["quality_cost_master"]) == 1
    assert out["quality_cost_unavailable"] == []


def test_estimated_comparable_with_valid_basis_is_admitted() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    row = stage_e_base_row()
    row["resource_status"] = "estimated_comparable"
    row["resource_comparability_basis"] = "same_estimator"

    out = build_stage_e_quality_cost_tables([row])
    assert len(out["quality_cost_master"]) == 1
    assert out["quality_cost_unavailable"] == []


def test_estimated_comparable_missing_basis_routes_resource_not_comparable() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    row = stage_e_base_row()
    row["resource_status"] = "estimated_comparable"
    row["resource_comparability_basis"] = "invalid_basis"

    out = build_stage_e_quality_cost_tables([row])
    assert out["quality_cost_master"] == []
    assert out["quality_cost_unavailable"][0]["unavailable_class"] == "resource_not_comparable"


def test_missing_required_quality_routes_missing_quality_fields() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    row = stage_e_base_row()
    row["topk_regret"] = None

    out = build_stage_e_quality_cost_tables([row])
    assert out["quality_cost_master"] == []
    assert out["quality_cost_unavailable"][0]["unavailable_class"] == "missing_quality_fields"


def test_missing_qubits_gates_depth_routes_missing_resource_fields() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    row = stage_e_base_row()
    row["resource_total_qubits_est_with_decode"] = None
    row["resource_total_gate_est_with_decode"] = None
    row["resource_total_depth_est_with_decode"] = None

    out = build_stage_e_quality_cost_tables([row])
    assert out["quality_cost_master"] == []
    unavailable = out["quality_cost_unavailable"][0]
    assert unavailable["unavailable_class"] == "missing_resource_fields"
    assert set(unavailable["missing_resource_fields"]) == {"qubits", "gates", "depth"}


def test_paired_validation_is_skipped_when_not_expected() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables
    from src.verify_encoding import PAIR_VALIDATION_FLAG

    row = stage_e_base_row()
    row[PAIR_VALIDATION_FLAG] = False

    out = build_stage_e_quality_cost_tables([row], paired_artifacts_by_instance={})
    assert len(out["quality_cost_master"]) == 1
    assert out["quality_cost_unavailable"] == []


def test_paired_validation_fails_when_expected_and_artifact_invalid() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables
    from src.verify_encoding import PAIR_VALIDATION_FLAG

    row = stage_e_base_row()
    row[PAIR_VALIDATION_FLAG] = True

    out = build_stage_e_quality_cost_tables(
        [row],
        paired_artifacts_by_instance={"S1": invalid_paired_artifact()},
    )
    assert out["quality_cost_master"] == []
    assert out["quality_cost_unavailable"][0]["unavailable_class"] == "encoding_validation_failed"


def test_unverified_provenance_does_not_fabricate_gap_metric() -> None:
    from src.resources_compare import normalize_stage_e_candidate_row

    row = stage_e_base_row()
    row["has_verified_provenance"] = False
    row["surrogate_best_vs_true_best_sampled_gap"] = None
    row["best_sampled_F"] = None
    row["best_top_G_sampled_F"] = None

    out = normalize_stage_e_candidate_row(row)
    faith = out["faithfulness_fields"]
    assert faith["surrogate_best_vs_true_best_sampled_gap"] is None


def test_insufficient_provenance_routes_when_gap_is_required() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    row = stage_e_base_row()
    row["has_verified_provenance"] = False
    row["surrogate_best_vs_true_best_sampled_gap"] = None
    row["best_sampled_F"] = None
    row["best_top_G_sampled_F"] = None

    out = build_stage_e_quality_cost_tables([row])
    assert out["quality_cost_master"] == []
    assert out["quality_cost_unavailable"][0]["unavailable_class"] == "insufficient_provenance"
