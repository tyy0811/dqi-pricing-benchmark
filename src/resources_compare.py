"""Stage 3 quality-vs-cost join helpers.

This module is intentionally a pure join/normalization layer. It does not run
new benchmark experiments; it only assembles existing benchmark outputs into a
stable, JSON-serializable summary contract.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from src.faithfulness import can_compute_faithfulness, normalize_faithfulness_fields
from src.report_schema import (
    STAGE_E_COMPARABLE_RESOURCE_STATUS_VALUES,
    STAGE_E_UNAVAILABLE_CLASS_VALUES,
    validate_quality_cost_master,
    validate_quality_cost_unavailable,
)
from src.verify_encoding import (
    PAIR_VALIDATION_FLAG,
    should_validate_paired_encoding,
    validate_paired_encoding_row,
)


SCHEMA_VERSION = "1"
ESTIMATED_COMPARABILITY_BASIS = {
    "same_estimator",
    "same_calibration",
    "same_method_version",
}
STAGE_E_REQUIRED_QUALITY_FIELDS = (
    "best_F",
    "topk_regret",
    "decision_distortion",
    "spearman_rho_F_G",
    "retained_energy_eta",
)
STAGE_E_REQUIRED_IDENTITY_FIELDS = (
    "run_id",
    "manifest_slot",
    "matrix",
    "stage",
    "instance_id",
    "encoding",
    "decoder",
    "alpha_mode",
    "execution_mode",
    "canonical_trial_seed",
)
STAGE_E_COMPARABLE_RESOURCE_STATUSES = set(STAGE_E_COMPARABLE_RESOURCE_STATUS_VALUES)
STAGE_E_DUPLICATE_KEY_FIELDS = (
    "instance_id",
    "encoding",
    "decoder",
    "alpha_mode",
    "execution_mode",
)
STAGE_E_UNAVAILABLE_CLASS_ENUM = set(STAGE_E_UNAVAILABLE_CLASS_VALUES)
STAGE_E_STATUS_ALIASES = {
    "ok": "completed",
    "completed": "completed",
    "error": "failed",
    "failed": "failed",
    "skipped": "not_applicable",
    "unsupported": "not_applicable",
    "not_applicable": "not_applicable",
    "unavailable": "unavailable",
    "not_in_experiment_matrix": "not_in_experiment_matrix",
}

_SKIP_DQI_KEYS = {
    "dqi_uniform",
    "dqi_paper",
    "dqi_heuristic",
    "dqi_optimal",
    "dqi_heuristic_optimal",
}


def _to_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_stage_e_run_status(value: Any) -> str:
    if value is None:
        return "unavailable"
    raw = str(value).strip().lower()
    if not raw:
        return "unavailable"
    return STAGE_E_STATUS_ALIASES.get(raw, raw)


def _is_missing_identity_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def normalize_resource_row_for_stage_e(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize resource fields to canonical Stage E comparability columns."""
    qubits = _to_number(
        _first_present(
            row,
            [
                "qubits",
                "resource_total_qubits_est_with_decode",
                "total_qubits_est_with_decode",
            ],
        )
    )
    gates = _to_number(
        _first_present(
            row,
            [
                "gates",
                "resource_total_gate_est_with_decode",
                "total_gate_est_with_decode",
            ],
        )
    )
    depth = _to_number(
        _first_present(
            row,
            [
                "depth",
                "resource_total_depth_est_with_decode",
                "total_depth_est_with_decode",
            ],
        )
    )
    resource_confidence = _first_present(
        row,
        [
            "resource_confidence",
            "resource_decode_uncompute_confidence",
            "decode_uncompute_confidence",
        ],
    )
    status = _first_present(row, ["resource_status"])
    reason = None
    basis = _first_present(row, ["resource_comparability_basis"])

    if qubits is None or gates is None or depth is None:
        status = "not_comparable"
        reason = "missing_resource_fields"
    elif status not in (STAGE_E_COMPARABLE_RESOURCE_STATUSES | {"not_comparable"}):
        if resource_confidence in {"exact_formula", "implementation_proxy"}:
            status = "comparable"
        elif resource_confidence == "analytic_upper_bound":
            status = "estimated_comparable"
        else:
            status = "not_comparable"
            reason = "unknown_resource_confidence"

    if status == "estimated_comparable":
        if basis not in ESTIMATED_COMPARABILITY_BASIS:
            status = "not_comparable"
            reason = "missing_comparability_basis"
    elif status == "comparable":
        basis = None

    if status == "not_comparable" and reason is None:
        reason = "resource_not_comparable"

    return {
        "qubits": qubits,
        "gates": gates,
        "depth": depth,
        "resource_status": status,
        "resource_confidence": resource_confidence,
        "resource_comparability_basis": basis,
        "resource_not_comparable_reason": reason,
        "qiskit_subset_resource_report": row.get("qiskit_subset_resource_report"),
        "qiskit_transpiled_structural_count": row.get("qiskit_transpiled_structural_count"),
        "qiskit_subset_transpiled_depth": _to_number(row.get("qiskit_subset_transpiled_depth")),
        "qiskit_subset_cx_count": _to_number(row.get("qiskit_subset_cx_count")),
        "qiskit_subset_qubit_count": _to_number(row.get("qiskit_subset_qubit_count")),
        "qiskit_subset_status": row.get("qiskit_subset_status"),
    }


def normalize_stage_e_candidate_row(
    candidate_row: dict[str, Any],
    *,
    paired_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical Stage E row normalization and admission routing.

    This function is the single normalization path for Stage E inputs and
    reuses existing helpers for resource normalization, faithfulness
    normalization, and optional paired-encoding validation.
    """
    row = dict(candidate_row)
    row["run_status"] = _normalize_stage_e_run_status(row.get("run_status"))
    row["canonical_trial_seed"] = _first_non_null(
        row.get("canonical_trial_seed"),
        row.get("trial_seed"),
        row.get("seed"),
    )
    row["has_verified_provenance"] = bool(
        _first_non_null(row.get("has_verified_provenance"), row.get("observed_run"), False)
    )

    identity_fields = {
        "run_id": row.get("run_id"),
        "manifest_slot": _first_non_null(row.get("manifest_slot"), row.get("slot_id")),
        "matrix": row.get("matrix"),
        "stage": row.get("stage"),
        "family": row.get("family"),
        "instance_id": row.get("instance_id"),
        "encoding": row.get("encoding"),
        "decoder": row.get("decoder"),
        "alpha_mode": row.get("alpha_mode"),
        "execution_mode": row.get("execution_mode"),
        "trial_seed": row.get("trial_seed"),
        "canonical_trial_seed": row.get("canonical_trial_seed"),
        "seed": row.get("seed"),
        "run_status": row.get("run_status"),
        "status_reason_code": row.get("status_reason_code"),
        "status_reason": row.get("status_reason"),
        "run_error": row.get("run_error"),
    }

    quality_fields = {
        "best_F": _to_number(_first_non_null(row.get("best_F"), row.get("best_sampled_F"))),
        "best_sampled_F": _to_number(row.get("best_sampled_F")),
        "best_sampled_G": _to_number(row.get("best_sampled_G")),
        "top1_regret": _to_number(row.get("top1_regret")),
        "topk_regret": _to_number(row.get("topk_regret")),
        "optimum_recall_at_k": _to_number(row.get("optimum_recall_at_k")),
        "decode_success": _to_number(row.get("decode_success")),
        "postselection_success": _to_number(row.get("postselection_success")),
    }

    faithfulness_input = dict(row)
    faithfulness_input.update(quality_fields)
    faithfulness_fields = normalize_faithfulness_fields(faithfulness_input)

    resource_fields = normalize_resource_row_for_stage_e(row)

    expects_encoding_validation = should_validate_paired_encoding(row)
    if expects_encoding_validation:
        encoding_ok, encoding_reason_code, encoding_reason = validate_paired_encoding_row(
            row=row,
            paired_artifact=paired_artifact,
        )
    else:
        encoding_ok, encoding_reason_code, encoding_reason = (True, None, None)

    faithfulness_ok, faithfulness_reason = can_compute_faithfulness(faithfulness_input)
    faithfulness_blocked = (
        faithfulness_fields.get("surrogate_best_vs_true_best_sampled_gap") is None
        and not faithfulness_ok
    )

    required_quality_view = {
        "best_F": quality_fields.get("best_F"),
        "topk_regret": quality_fields.get("topk_regret"),
        "decision_distortion": faithfulness_fields.get("decision_distortion"),
        "spearman_rho_F_G": faithfulness_fields.get("spearman_rho_F_G"),
        "retained_energy_eta": faithfulness_fields.get("retained_energy_eta"),
    }

    # Check if energy_eta is applicable for this encoding type
    # Non-WHT encodings (e.g., synthetic_structured_bv) set this to False
    energy_eta_applicable = row.get("energy_eta_applicable", True)

    missing_identity_fields = [
        field for field in STAGE_E_REQUIRED_IDENTITY_FIELDS if _is_missing_identity_value(identity_fields.get(field))
    ]
    # Only require retained_energy_eta if energy_eta is applicable for this encoding
    required_quality_fields_effective = [
        field for field in STAGE_E_REQUIRED_QUALITY_FIELDS
        if not (field == "retained_energy_eta" and not energy_eta_applicable)
    ]
    missing_quality_fields = [
        field for field in required_quality_fields_effective if _to_number(required_quality_view.get(field)) is None
    ]

    unavailable_class: str | None = None
    unavailable_reason_code: str | None = None
    unavailable_reason: str | None = None

    if missing_identity_fields:
        unavailable_class = "missing_required_fields"
        unavailable_reason_code = "missing_required_fields"
        unavailable_reason = f"missing required fields: {', '.join(missing_identity_fields)}"
    elif identity_fields.get("run_status") != "completed":
        missing_identity_fields = list(missing_identity_fields)
        if "run_status" not in missing_identity_fields:
            missing_identity_fields.append("run_status")
        unavailable_class = "missing_required_fields"
        unavailable_reason_code = str(identity_fields.get("status_reason_code") or "not_completed_status")
        unavailable_reason = str(identity_fields.get("status_reason") or "row is not completed")
    elif not encoding_ok:
        unavailable_class = "encoding_validation_failed"
        unavailable_reason_code = str(encoding_reason_code or "encoding_validation_failed")
        unavailable_reason = str(encoding_reason or "paired encoding validation failed")
    elif faithfulness_blocked:
        unavailable_class = "insufficient_provenance"
        unavailable_reason_code = str(faithfulness_reason or "insufficient_provenance")
        unavailable_reason = "faithfulness computation blocked by provenance"
    elif missing_quality_fields:
        unavailable_class = "missing_quality_fields"
        unavailable_reason_code = "missing_quality_fields"
        unavailable_reason = f"missing quality fields: {', '.join(missing_quality_fields)}"
    else:
        resource_status = str(resource_fields.get("resource_status") or "")
        if resource_status not in STAGE_E_COMPARABLE_RESOURCE_STATUSES:
            reason_code = str(resource_fields.get("resource_not_comparable_reason") or "resource_not_comparable")
            if reason_code == "missing_resource_fields":
                unavailable_class = "missing_resource_fields"
            else:
                unavailable_class = "resource_not_comparable"
            unavailable_reason_code = reason_code
            unavailable_reason = f"resource status is {resource_status or 'unknown'}"
        elif resource_status == "estimated_comparable":
            basis = resource_fields.get("resource_comparability_basis")
            if basis not in ESTIMATED_COMPARABILITY_BASIS:
                unavailable_class = "resource_not_comparable"
                unavailable_reason_code = "missing_comparability_basis"
                unavailable_reason = "estimated resources missing valid comparability basis"

    admitted_to_master = unavailable_class is None
    route_to_unavailable = not admitted_to_master

    provenance_validation_fields = {
        "has_verified_provenance": bool(faithfulness_input.get("has_verified_provenance", False)),
        "in_experiment_matrix": row.get("in_experiment_matrix"),
        "observed_run": row.get("observed_run"),
        "source_artifact": row.get("source_artifact"),
        PAIR_VALIDATION_FLAG: expects_encoding_validation,
        "encoding_validation_ok": bool(encoding_ok),
        "encoding_validation_reason_code": encoding_reason_code,
        "encoding_validation_reason": encoding_reason,
        "faithfulness_provenance_ok": bool(faithfulness_ok),
        "faithfulness_provenance_reason": None if faithfulness_ok else faithfulness_reason,
    }

    normalized_row_candidate = {
        **identity_fields,
        **quality_fields,
        **faithfulness_fields,
        **resource_fields,
        **provenance_validation_fields,
        "energy_eta_applicable": energy_eta_applicable,
    }

    unavailable_row_candidate = None
    if route_to_unavailable:
        unavailable_row_candidate = {
            **normalized_row_candidate,
            "unavailable_class": unavailable_class,
            "unavailable_reason_code": unavailable_reason_code,
            "unavailable_reason": unavailable_reason,
            "missing_identity_fields": list(missing_identity_fields),
            "missing_quality_fields": list(missing_quality_fields),
            "resource_status_context": str(resource_fields.get("resource_status")),
            "resource_confidence_context": resource_fields.get("resource_confidence"),
            "resource_comparability_basis_context": resource_fields.get("resource_comparability_basis"),
            "encoding_validation_expected": bool(expects_encoding_validation),
        }

    return {
        "identity_fields": identity_fields,
        "quality_fields": quality_fields,
        "faithfulness_fields": faithfulness_fields,
        "resource_fields": resource_fields,
        "provenance_validation_fields": provenance_validation_fields,
        "admission": {
            "admitted_to_master": admitted_to_master,
            "route_to_unavailable": route_to_unavailable,
            "unavailable_class": unavailable_class,
            "unavailable_reason_code": unavailable_reason_code,
            "unavailable_reason": unavailable_reason,
            "missing_identity_fields": list(missing_identity_fields),
            "missing_quality_fields": list(missing_quality_fields),
        },
        "normalized_row_candidate": normalized_row_candidate,
        "unavailable_row_candidate": unavailable_row_candidate,
    }


def stage_e_duplicate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in STAGE_E_DUPLICATE_KEY_FIELDS)


def _manifest_slot_sort_key(value: Any) -> tuple[int, str]:
    if value is None:
        return (1, "~")
    return (0, str(value))


def stage_e_duplicate_winner_key(row: dict[str, Any]) -> tuple[Any, ...]:
    completed_rank = 0 if str(row.get("run_status")) == "completed" else 1
    verified_rank = 0 if bool(row.get("has_verified_provenance", False)) else 1
    manifest_slot_rank = _manifest_slot_sort_key(row.get("manifest_slot"))
    run_id = str(row.get("run_id") or "~")
    return (completed_rank, verified_rank, manifest_slot_rank, run_id)


def _stage_e_unavailable_row(
    row: dict[str, Any],
    *,
    unavailable_class: str,
    unavailable_reason_code: str,
    unavailable_reason: str,
) -> dict[str, Any]:
    cls = str(unavailable_class)
    if cls not in STAGE_E_UNAVAILABLE_CLASS_ENUM:
        cls = "missing_required_fields"
    out = dict(row)
    out["unavailable_class"] = cls
    out["unavailable_reason_code"] = str(unavailable_reason_code)
    out["unavailable_reason"] = str(unavailable_reason)
    return out


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        out.append(str(item))
    return out


def partition_stage_e_quality_cost_rows(
    candidate_rows: list[dict[str, Any]],
    *,
    paired_artifacts_by_instance: dict[str, dict[str, Any] | None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition candidate Stage E rows into comparable and unavailable lists.

    Policy:
    - Normalize each candidate via ``normalize_stage_e_candidate_row``.
    - For rows that pass admission, perform deterministic deduplication by the
      frozen comparable key (instance_id/encoding/decoder/alpha_mode/execution_mode).
    - Winners remain comparable; non-winners are preserved in unavailable as
      ``duplicate_key``.
    - Rows missing ``manifest_slot`` are not considered in dedup ranking and are
      routed to unavailable with explicit required-fields reason.
    """
    paired_artifacts_by_instance = paired_artifacts_by_instance or {}

    admitted_candidates: list[dict[str, Any]] = []
    unavailable_rows: list[dict[str, Any]] = []

    for candidate in candidate_rows:
        row = dict(candidate)
        paired_artifact = paired_artifacts_by_instance.get(str(row.get("instance_id")))
        normalized = normalize_stage_e_candidate_row(row, paired_artifact=paired_artifact)
        passthrough = {
            field: row.get(field)
            for field in _PROJECTED_OPTIONAL_IDENTITY_COLUMNS
            if field in row
        }
        normalized_row = {**passthrough, **dict(normalized["normalized_row_candidate"])}
        admission = normalized["admission"]

        if bool(admission.get("admitted_to_master", False)):
            if _is_missing_identity_value(normalized_row.get("manifest_slot")):
                missing_required_fields = _as_string_list(
                    normalized.get("admission", {}).get("missing_identity_fields")
                )
                if "manifest_slot" not in missing_required_fields:
                    missing_required_fields.append("manifest_slot")
                unavailable_rows.append(
                    _stage_e_unavailable_row(
                        {**normalized_row, "missing_required_fields": missing_required_fields},
                        unavailable_class="missing_required_fields",
                        unavailable_reason_code="missing_required_fields",
                        unavailable_reason="missing required fields: manifest_slot",
                    )
                )
                continue
            admitted_candidates.append(normalized_row)
            continue

        unavailable_payload = {
            **passthrough,
            **dict(normalized.get("unavailable_row_candidate") or normalized_row),
        }
        unavailable_rows.append(
            _stage_e_unavailable_row(
                unavailable_payload,
                unavailable_class=str(admission.get("unavailable_class") or "missing_required_fields"),
                unavailable_reason_code=str(admission.get("unavailable_reason_code") or "missing_required_fields"),
                unavailable_reason=str(admission.get("unavailable_reason") or "row is unavailable"),
            )
        )

    duplicate_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in admitted_candidates:
        duplicate_groups[stage_e_duplicate_key(row)].append(row)

    comparable_rows: list[dict[str, Any]] = []
    for group_rows in duplicate_groups.values():
        ranked = sorted(group_rows, key=stage_e_duplicate_winner_key)
        winner = ranked[0]
        comparable_rows.append(winner)
        for loser in ranked[1:]:
            unavailable_rows.append(
                _stage_e_unavailable_row(
                    loser,
                    unavailable_class="duplicate_key",
                    unavailable_reason_code="duplicate_key",
                    unavailable_reason=f"duplicate key loser; winner={winner.get('run_id')}",
                )
            )

    comparable_rows = sorted(comparable_rows, key=stage_e_duplicate_winner_key)
    unavailable_rows = sorted(
        unavailable_rows,
        key=lambda row: (
            str(row.get("unavailable_class")),
            stage_e_duplicate_winner_key(row),
        ),
    )
    return comparable_rows, unavailable_rows


STAGE_E_MASTER_REQUIRED_COLUMNS = (
    "run_id",
    "manifest_slot",
    "instance_id",
    "family",
    "encoding",
    "decoder",
    "alpha_mode",
    "execution_mode",
    "best_F",
    "topk_regret",
    "decision_distortion",
    "spearman_rho_F_G",
    "retained_energy_eta",  # Can be None if energy_eta_applicable=False
    "energy_eta_applicable",  # False for non-WHT encodings where energy_eta is N/A
    "qubits",
    "gates",
    "depth",
    "resource_status",
    "resource_confidence",
    "has_verified_provenance",
)

STAGE_E_UNAVAILABLE_REQUIRED_COLUMNS = (
    "run_id",
    "instance_id",
    "family",
    "encoding",
    "decoder",
    "alpha_mode",
    "execution_mode",
    "unavailable_class",
    "unavailable_reason",
    "run_status",
    "resource_status",
    "has_verified_provenance",
    "missing_required_fields",
    "missing_quality_fields",
    "missing_resource_fields",
)

_PROJECTED_OPTIONAL_IDENTITY_COLUMNS = (
    "run_key",
    "matrix",
    "stage",
    "matrix_norm",
    "stage_norm",
    "family_norm",
    "encoding_norm",
    "decoder_norm",
    "alpha_mode_norm",
    "execution_mode_norm",
    "comparison_key",
    "counterpart_key",
    "stage_e_admission_key",
)


def _project_stage_e_master_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "run_id": row.get("run_id"),
        "manifest_slot": row.get("manifest_slot"),
        "instance_id": row.get("instance_id"),
        "family": row.get("family"),
        "encoding": row.get("encoding"),
        "decoder": row.get("decoder"),
        "alpha_mode": row.get("alpha_mode"),
        "execution_mode": row.get("execution_mode"),
        "run_status": row.get("run_status"),
        "best_F": _to_number(row.get("best_F")),
        "topk_regret": _to_number(row.get("topk_regret")),
        "decision_distortion": _to_number(row.get("decision_distortion")),
        "spearman_rho_F_G": _to_number(row.get("spearman_rho_F_G")),
        "retained_energy_eta": _to_number(row.get("retained_energy_eta")),
        "energy_eta_applicable": bool(row.get("energy_eta_applicable", True)),
        "qubits": _to_number(row.get("qubits")),
        "gates": _to_number(row.get("gates")),
        "depth": _to_number(row.get("depth")),
        "resource_status": row.get("resource_status"),
        "resource_confidence": row.get("resource_confidence"),
        "resource_comparability_basis": row.get("resource_comparability_basis"),
        "has_verified_provenance": bool(row.get("has_verified_provenance", False)),
        "qiskit_subset_resource_report": row.get("qiskit_subset_resource_report"),
        "qiskit_transpiled_structural_count": row.get("qiskit_transpiled_structural_count"),
        "qiskit_subset_transpiled_depth": row.get("qiskit_subset_transpiled_depth"),
        "qiskit_subset_cx_count": row.get("qiskit_subset_cx_count"),
        "qiskit_subset_qubit_count": row.get("qiskit_subset_qubit_count"),
        "qiskit_subset_status": row.get("qiskit_subset_status"),
    }
    for field in _PROJECTED_OPTIONAL_IDENTITY_COLUMNS:
        if field in row:
            out[field] = row.get(field)
    return out


def _missing_resource_fields_from_row(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for col in ("qubits", "gates", "depth"):
        if _to_number(row.get(col)) is None:
            missing.append(col)
    return missing


def _project_stage_e_unavailable_row(row: dict[str, Any]) -> dict[str, Any]:
    unavailable_class = str(row.get("unavailable_class") or "missing_required_fields")
    missing_required_fields = _as_string_list(
        _first_non_null(row.get("missing_required_fields"), row.get("missing_identity_fields"), [])
    )
    missing_quality_fields = _as_string_list(row.get("missing_quality_fields"))
    missing_resource_fields = _as_string_list(row.get("missing_resource_fields"))
    if unavailable_class == "missing_resource_fields" and not missing_resource_fields:
        missing_resource_fields = _missing_resource_fields_from_row(row)

    out = {
        "run_id": row.get("run_id"),
        "instance_id": row.get("instance_id"),
        "family": row.get("family"),
        "encoding": row.get("encoding"),
        "decoder": row.get("decoder"),
        "alpha_mode": row.get("alpha_mode"),
        "execution_mode": row.get("execution_mode"),
        "unavailable_class": unavailable_class,
        "unavailable_reason": row.get("unavailable_reason"),
        "unavailable_reason_code": row.get("unavailable_reason_code"),
        "run_status": row.get("run_status"),
        "resource_status": row.get("resource_status"),
        "resource_confidence": row.get("resource_confidence"),
        "has_verified_provenance": bool(row.get("has_verified_provenance", False)),
        "missing_required_fields": missing_required_fields,
        "missing_quality_fields": missing_quality_fields,
        "missing_resource_fields": missing_resource_fields,
        "qiskit_subset_resource_report": row.get("qiskit_subset_resource_report"),
        "qiskit_transpiled_structural_count": row.get("qiskit_transpiled_structural_count"),
        "qiskit_subset_transpiled_depth": row.get("qiskit_subset_transpiled_depth"),
        "qiskit_subset_cx_count": row.get("qiskit_subset_cx_count"),
        "qiskit_subset_qubit_count": row.get("qiskit_subset_qubit_count"),
        "qiskit_subset_status": row.get("qiskit_subset_status"),
    }
    for field in _PROJECTED_OPTIONAL_IDENTITY_COLUMNS:
        if field in row:
            out[field] = row.get(field)
    return out


def build_stage_e_quality_cost_tables(
    normalized_candidate_rows: list[dict[str, Any]],
    *,
    paired_artifacts_by_instance: dict[str, dict[str, Any] | None] | None = None,
    strict_validate: bool = True,
) -> dict[str, Any]:
    """Build canonical Stage E master/unavailable tables from candidate rows.

    Returns a payload with:
    - ``quality_cost_master``: admitted, deduplicated comparable rows
    - ``quality_cost_unavailable``: all rejected rows + dedup losers
    - ``summary``: counts by unavailable_class and admission/duplicate totals
    """
    comparable_rows, unavailable_rows = partition_stage_e_quality_cost_rows(
        normalized_candidate_rows,
        paired_artifacts_by_instance=paired_artifacts_by_instance,
    )

    master_rows = [_project_stage_e_master_row(row) for row in comparable_rows]
    unavailable_projected = [_project_stage_e_unavailable_row(row) for row in unavailable_rows]

    unavailable_count_by_class: dict[str, int] = {}
    for row in unavailable_projected:
        cls = str(row.get("unavailable_class") or "missing_required_fields")
        unavailable_count_by_class[cls] = int(unavailable_count_by_class.get(cls, 0) + 1)
    duplicate_count = int(unavailable_count_by_class.get("duplicate_key", 0))

    summary = {
        "input_candidate_count": int(len(normalized_candidate_rows)),
        "admitted_count": int(len(master_rows)),
        "duplicate_count": duplicate_count,
        "unavailable_count": int(len(unavailable_projected)),
        "unavailable_count_by_class": dict(sorted(unavailable_count_by_class.items())),
    }

    if strict_validate:
        validate_quality_cost_master(master_rows)
        validate_quality_cost_unavailable(unavailable_projected)

    return {
        "quality_cost_master": master_rows,
        "quality_cost_unavailable": unavailable_projected,
        "summary": summary,
    }


def _get_path(payload: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _infer_instance_id(results: dict[str, Any]) -> str:
    n_features = results.get("problem", {}).get("n_features")
    if isinstance(n_features, int):
        return f"P{n_features}"
    return "unknown_instance"


def _infer_instance_family(instance_id: str) -> str:
    if instance_id.startswith("P"):
        return "P"
    if instance_id.startswith("C"):
        return "C"
    if instance_id.startswith("D"):
        return "D"
    return "unknown"


def _make_row_id(
    *,
    instance_id: str,
    encoding_artifact_id: str,
    decoder: str,
    alpha_mode: str,
    execution_mode: str,
    run_label: str,
) -> str:
    return "|".join(
        [
            str(instance_id),
            str(encoding_artifact_id),
            str(decoder),
            str(alpha_mode),
            str(execution_mode),
            str(run_label),
        ]
    )


def _quality_defaults() -> dict[str, Any]:
    return {
        "best_F": None,
        "F_star": None,
        "best_F_over_F_star": None,
        "best_G": None,
        "top1_regret": None,
        "topk_regret": None,
        "optimum_recall_at_k": None,
        "best_top_G_sampled_F": None,
        "decision_distortion": None,
        "decode_success": None,
        "postselection_success": None,
    }


def _faithfulness_defaults() -> dict[str, Any]:
    return {
        "spearman_rho_F_G": None,
        "retained_energy_eta": None,
    }


def _collision_defaults() -> dict[str, Any]:
    return {
        "n": None,
        "m": None,
        "rank": None,
        "rank_deficiency": None,
        "row_weight_mean": None,
        "row_weight_max": None,
        "col_weight_mean": None,
        "col_weight_max": None,
        "syndrome_density": None,
        "ambiguous_syndrome_count": None,
        "collision_rate": None,
        "decodable_syndrome_fraction": None,
    }


def _coherent_overhead_template() -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "weight_register_qubits": None,
        "coherent_prep_overhead": None,
        "controlled_sector_overhead": None,
    }


def _resource_defaults() -> dict[str, Any]:
    return {
        "implemented_resources": {},
        "analytic_estimates": {},
        "total_estimated_resources": {},
        "prefix_gate_est": None,
        "prefix_depth_est": None,
        "decode_model_gate_est": None,
        "decode_model_depth_est": None,
        "total_gate_est_with_decode": None,
        "total_depth_est_with_decode": None,
        "total_qubits_est_with_decode": None,
        "coherent_alpha_overhead": _coherent_overhead_template(),
        "resource_status": {
            "decode_included": False,
            "decode_model_status": "not_included",
            "weight_register_status": "not_applicable",
            "transpiled_subset_available": False,
        },
        "qiskit_subset_resource_report": None,
        "qiskit_transpiled_structural_count": None,
        "qiskit_subset_transpiled_depth": None,
        "qiskit_subset_cx_count": None,
        "qiskit_subset_qubit_count": None,
        "qiskit_subset_status": None,
    }


def _candidate_run_keys(results: dict[str, Any]) -> list[str]:
    preferred = [
        "dqi_uniform_mixture",
        "dqi_paper_mixture",
        "dqi_heuristic_mixture",
        "dqi_paper_mixture_bp1",
        "dqi_paper_coherent",
    ]
    keys: list[str] = [k for k in preferred if k in results]

    dynamic = sorted(
        k
        for k, v in results.items()
        if k.startswith("dqi_")
        and isinstance(v, dict)
        and k not in _SKIP_DQI_KEYS
        and k not in keys
        and (
            "decision_metrics" in v
            or v.get("status") == "unsupported"
        )
    )
    keys.extend(dynamic)
    return keys


def _decoder_from_run(key: str, run: dict[str, Any]) -> str:
    if "decoder_mode" in run and run.get("decoder_mode") is not None:
        return str(run["decoder_mode"])
    if "bp1" in key:
        return "bp1"
    return "bruteforce"


def _execution_mode_from_run(key: str, run: dict[str, Any]) -> str:
    if "execution_mode" in run and run.get("execution_mode") is not None:
        return str(run["execution_mode"])
    if "coherent" in key:
        return "coherent"
    return "mixture"


def _alpha_mode_from_run(run: dict[str, Any]) -> str:
    return str(run.get("alpha_mode", "unknown"))


def _resource_summary_for_decoder(results: dict[str, Any], decoder: str) -> dict[str, Any] | None:
    by_model = results.get("resources_by_decoder_model")
    if isinstance(by_model, dict) and isinstance(by_model.get(decoder), dict):
        return by_model[decoder]

    if decoder == "bruteforce":
        legacy = results.get("resources")
        if isinstance(legacy, dict):
            return legacy
    return None


def _build_quality_block(run: dict[str, Any], f_star: float | None) -> tuple[dict[str, Any], bool]:
    quality = _quality_defaults()
    metrics = run.get("decision_metrics") if isinstance(run, dict) else None
    core = metrics.get("core_decision", {}) if isinstance(metrics, dict) else {}
    pipeline = metrics.get("pipeline", {}) if isinstance(metrics, dict) else {}

    has_quality = False

    best_f = run.get("best_F")
    if best_f is not None:
        quality["best_F"] = float(best_f)
        has_quality = True

    if f_star is not None:
        quality["F_star"] = float(f_star)

    if quality["best_F"] is not None and quality["F_star"] not in (None, 0.0):
        quality["best_F_over_F_star"] = float(quality["best_F"] / quality["F_star"])

    if run.get("best_G_weighted") is not None:
        quality["best_G"] = float(run["best_G_weighted"])
    elif run.get("best_G") is not None:
        quality["best_G"] = float(run["best_G"])

    for src, dst in [
        ("top1_regret", "top1_regret"),
        ("topk_regret", "topk_regret"),
        ("optimum_recall_at_k", "optimum_recall_at_k"),
        ("best_top_G_sampled_F", "best_top_G_sampled_F"),
        ("decision_distortion", "decision_distortion"),
    ]:
        if core.get(src) is not None:
            quality[dst] = float(core[src])

    decode_success = pipeline.get("decode_success")
    if decode_success is None:
        decode_success = run.get("decoder_success_rate")
    if decode_success is not None:
        quality["decode_success"] = float(decode_success)

    postselection_success = pipeline.get("postselection_success")
    if postselection_success is None:
        postselection_success = run.get("success_prob")
    if postselection_success is not None:
        quality["postselection_success"] = float(postselection_success)

    return quality, has_quality


def _build_faithfulness_block(run: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    faith = _faithfulness_defaults()
    metrics = run.get("decision_metrics") if isinstance(run, dict) else None
    faith_block = metrics.get("faithfulness", {}) if isinstance(metrics, dict) else {}

    rho = faith_block.get("spearman_rho_F_G")
    if rho is None:
        rho = run.get("spearman_rho_weighted")
    if rho is None:
        rho = results.get("surrogate", {}).get("spearman_rho_weighted")
    if rho is not None:
        faith["spearman_rho_F_G"] = float(rho)

    eta = faith_block.get("retained_energy_eta")
    if eta is None:
        eta = run.get("energy_fraction")
    if eta is None:
        eta = results.get("surrogate", {}).get("energy_fraction")
    if eta is not None:
        faith["retained_energy_eta"] = float(eta)

    return faith


def _build_collision_block(results: dict[str, Any]) -> dict[str, Any]:
    struct = results.get("structure", {}) if isinstance(results.get("structure"), dict) else {}
    out = _collision_defaults()

    mappings = {
        "n": "n",
        "m": "m",
        "rank": "rank",
        "rank_deficiency": "rank_deficiency",
        "row_weight_mean": "row_weight_mean",
        "row_weight_max": "row_weight_max",
        "col_weight_mean": "col_weight_mean",
        "col_weight_max": "col_weight_max",
        "syndrome_density": "syndrome_density",
        "ambiguous_syndrome_count": "ambiguous_syndrome_count",
        "collision_rate": "decoder_collision_rate",
        "decodable_syndrome_fraction": "decodable_syndrome_fraction",
    }
    for dst, src in mappings.items():
        val = struct.get(src)
        out[dst] = val if val is None else (float(val) if isinstance(val, float) else int(val))

    if out["rank"] is None and struct.get("rank_B_gf2") is not None:
        out["rank"] = int(struct["rank_B_gf2"])

    if out["rank_deficiency"] is None and out["n"] is not None and out["rank"] is not None:
        out["rank_deficiency"] = int(max(0, int(out["n"]) - int(out["rank"])))

    if out["decodable_syndrome_fraction"] is None:
        decodable = struct.get("number_of_unique_syndromes_decodable_at_ell")
        space = struct.get("syndrome_space_size")
        if decodable is not None and space:
            out["decodable_syndrome_fraction"] = float(float(decodable) / float(space))

    return out


def _build_resources_block(
    *,
    resource_summary: dict[str, Any] | None,
    execution_mode: str,
) -> tuple[dict[str, Any], bool]:
    out = _resource_defaults()
    if resource_summary is None:
        return out, False

    implemented = resource_summary.get("implemented_resources") or {}
    analytic = resource_summary.get("analytic_estimates") or resource_summary.get("estimated_resources") or {}
    total = resource_summary.get("total_estimated_resources") or resource_summary.get("total_resources") or {}

    out["implemented_resources"] = implemented
    out["analytic_estimates"] = analytic
    out["total_estimated_resources"] = total

    out["prefix_gate_est"] = resource_summary.get("prefix_gate_est", resource_summary.get("total_gate_est"))
    out["prefix_depth_est"] = resource_summary.get("prefix_depth_est", resource_summary.get("total_depth_est"))
    out["decode_model_gate_est"] = resource_summary.get(
        "decode_model_gate_est", resource_summary.get("decode_uncompute_gate_est")
    )
    out["decode_model_depth_est"] = resource_summary.get(
        "decode_model_depth_est", resource_summary.get("decode_uncompute_depth_est")
    )
    out["total_gate_est_with_decode"] = resource_summary.get("total_gate_est_with_decode")
    out["total_depth_est_with_decode"] = resource_summary.get("total_depth_est_with_decode")
    out["total_qubits_est_with_decode"] = resource_summary.get("total_qubits_est_with_decode")

    qiskit_subset_resource_report = resource_summary.get("qiskit_subset_resource_report")
    if not isinstance(qiskit_subset_resource_report, dict):
        qiskit_subset_resource_report = None

    transpiled = resource_summary.get("qiskit_transpiled_structural_count") or {}
    if not isinstance(transpiled, dict):
        transpiled = None

    qiskit_subset_depth = None
    qiskit_subset_cx = None
    qiskit_subset_qubits = None
    qiskit_subset_status = None
    if isinstance(qiskit_subset_resource_report, dict):
        qiskit_subset_depth = qiskit_subset_resource_report.get("transpiled_depth")
        qiskit_subset_cx = qiskit_subset_resource_report.get("cx_count")
        qiskit_subset_qubits = qiskit_subset_resource_report.get("qubit_count")
        qiskit_subset_status = qiskit_subset_resource_report.get("status")

    subset_available = str(transpiled.get("status", "") if isinstance(transpiled, dict) else "").lower() in {"ok", "completed"}

    out["qiskit_subset_resource_report"] = qiskit_subset_resource_report
    out["qiskit_transpiled_structural_count"] = transpiled
    out["qiskit_subset_transpiled_depth"] = qiskit_subset_depth
    out["qiskit_subset_cx_count"] = qiskit_subset_cx
    out["qiskit_subset_qubit_count"] = qiskit_subset_qubits
    out["qiskit_subset_status"] = qiskit_subset_status

    out["resource_status"] = {
        "decode_included": out["total_gate_est_with_decode"] is not None,
        "decode_model_status": str(resource_summary.get("decode_uncompute_status", "not_included")),
        "weight_register_status": (
            "not_applicable"
            if execution_mode != "coherent"
            else str(resource_summary.get("weight_register_overhead_status", "not_included"))
        ),
        "transpiled_subset_available": subset_available,
    }

    if execution_mode == "coherent":
        out["coherent_alpha_overhead"] = {
            "status": str(resource_summary.get("weight_register_overhead_status", "not_included")),
            "weight_register_qubits": resource_summary.get("weight_register_qubits"),
            "coherent_prep_overhead": resource_summary.get("coherent_weight_register_gate_est"),
            "controlled_sector_overhead": resource_summary.get("coherent_weight_register_depth_est"),
        }

    return out, True


def _row_status(*, has_quality: bool, has_resources: bool, run_status: str | None) -> str:
    if run_status == "unsupported":
        return "skipped"
    if has_quality and has_resources:
        return "complete"
    if has_resources and not has_quality:
        return "missing_quality"
    if has_quality and not has_resources:
        return "missing_resources"
    return "skipped"


def _compute_pareto_frontier(
    rows: list[dict[str, Any]],
    *,
    maximize_keys: list[str],
    minimize_keys: list[str],
) -> list[str]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        valid = True
        for key in maximize_keys + minimize_keys:
            if _get_path(row, key) is None:
                valid = False
                break
        if valid:
            candidates.append(row)

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        a_ge_all = True
        a_strict = False

        for key in maximize_keys:
            av = float(_get_path(a, key))
            bv = float(_get_path(b, key))
            if av < bv:
                a_ge_all = False
                break
            if av > bv:
                a_strict = True

        if a_ge_all:
            for key in minimize_keys:
                av = float(_get_path(a, key))
                bv = float(_get_path(b, key))
                if av > bv:
                    a_ge_all = False
                    break
                if av < bv:
                    a_strict = True

        return a_ge_all and a_strict

    frontier: list[dict[str, Any]] = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            if dominates(other, row):
                dominated = True
                break
        if not dominated:
            frontier.append(row)

    return sorted(str(r.get("row_id", "")) for r in frontier)


def _build_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    instance_id = _infer_instance_id(results)
    instance_family = _infer_instance_family(instance_id)
    encoding_artifact_id = str(
        results.get("resources", {}).get("encoding_artifact_id", "wht_default")
    )
    f_star_raw = results.get("ground_truth", {}).get("f_opt")
    f_star = float(f_star_raw) if f_star_raw is not None else None

    for key in _candidate_run_keys(results):
        run = results.get(key)
        if not isinstance(run, dict):
            continue

        decoder = _decoder_from_run(key, run)
        execution_mode = _execution_mode_from_run(key, run)
        alpha_mode = _alpha_mode_from_run(run)

        quality, has_quality = _build_quality_block(run, f_star)
        faithfulness = _build_faithfulness_block(run, results)
        collision = _build_collision_block(results)
        resource_summary = _resource_summary_for_decoder(results, decoder)
        resources, has_resources = _build_resources_block(
            resource_summary=resource_summary,
            execution_mode=execution_mode,
        )

        run_label = f"{instance_id} / wht_truncated / {decoder} / {alpha_mode} / {execution_mode}"
        row_id = _make_row_id(
            instance_id=instance_id,
            encoding_artifact_id=encoding_artifact_id,
            decoder=decoder,
            alpha_mode=alpha_mode,
            execution_mode=execution_mode,
            run_label=run_label,
        )

        row = {
            "row_id": row_id,
            "row_status": _row_status(
                has_quality=has_quality,
                has_resources=has_resources,
                run_status=run.get("status"),
            ),
            "instance_id": instance_id,
            "instance_family": instance_family,
            "encoding_artifact_id": encoding_artifact_id,
            "run_label": run_label,
            "provenance": {
                "encoding_backend": "wht_truncated",
                "decoder": decoder,
                "decoder_resource_model": (
                    None
                    if resource_summary is None
                    else resource_summary.get("decode_uncompute_model")
                ),
                "alpha_mode": alpha_mode,
                "execution_mode": execution_mode,
            },
            "quality": quality,
            "faithfulness": faithfulness,
            "collision_structure": collision,
            "resources": resources,
            "status": {
                "decode_confidence": (
                    None
                    if resource_summary is None
                    else resource_summary.get("decode_uncompute_confidence")
                ),
                "coherent_supported": run.get("coherent_supported"),
            },
        }

        # Keep rows with explicit quality/resources blocks only.
        if "quality" in row and "resources" in row:
            rows.append(row)

    rows.sort(key=lambda r: str(r["row_id"]))
    return rows


def build_quality_vs_cost_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Build canonical Stage 3 quality-vs-cost summary from benchmark payload."""
    rows = _build_rows(results)

    objective = {
        "maximize": ["quality.best_F_over_F_star"],
        "minimize": ["resources.total_gate_est_with_decode"],
    }

    pareto_row_ids = _compute_pareto_frontier(
        rows,
        maximize_keys=list(objective["maximize"]),
        minimize_keys=list(objective["minimize"]),
    )
    instance_id = _infer_instance_id(results)
    metadata = {
        "instance_id": instance_id,
        "instance_family": _infer_instance_family(instance_id),
        "n_bits": results.get("problem", {}).get("n_bits"),
        "k": results.get("parameters", {}).get("k"),
        "ell": results.get("parameters", {}).get("ell"),
        "canonical_source": "embedded_run_benchmark_payload",
        "has_decode_inclusive_totals": any(
            _get_path(row, "resources.total_gate_est_with_decode") is not None
            for row in rows
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_from_benchmark_version": str(results.get("repo_commit", "unknown_or_commit")),
        "metadata": metadata,
        "row_count": int(len(rows)),
        "objectives": objective,
        "rows": rows,
        "pareto": {
            "objective": objective,
            "row_ids": pareto_row_ids,
        },
        "notes": [
            "Embedded quality_vs_cost payload is canonical; no auto-export side effects.",
            "Rows with missing pareto objective values are excluded from pareto evaluation.",
        ],
    }


def export_quality_cost_summary(results: dict[str, Any], out_path: str | Path) -> None:
    """Manual-only export helper for quality_vs_cost summary."""
    payload = build_quality_vs_cost_summary(results)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
