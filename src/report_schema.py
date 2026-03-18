"""Reporting-layer schema validators."""

from __future__ import annotations

from typing import Any


_STAGE_D_REQUIRED_IDENTITY = {
    "run_id",
    "matrix",
    "stage",
    "family",
    "instance_id",
    "encoding",
    "decoder",
    "alpha_mode",
    "execution_mode",
    "ell",
    "m_active",
    "trial_seed",
    "canonical_trial_seed",
    "seed",
}

_STAGE_D_REQUIRED_STATUS = {
    "run_status",
    "status_reason_code",
    "status_reason",
    "in_experiment_matrix",
    "run_error",
    "coherent_supported",
    "oracle_exact",
}

_STAGE_D_REQUIRED_RAW_METRICS = {
    "success_probability",
    "best_F",
    "best_sampled_F",
    "best_top_G_sampled_F",
    "F_star",
    "top1_regret",
    "postselection_success",
}

_STAGE_D_RUN_STATUS_ALLOWED = {"completed", "failed", "not_applicable"}
_STAGE_D_RUN_STATUS_LEGACY = {"ok", "error", "skipped"}
STAGE_E_RUN_STATUS_VALUES = frozenset(
    {
        "completed",
        "failed",
        "not_applicable",
        "unavailable",
        "not_in_experiment_matrix",
    }
)
STAGE_E_COMPARABLE_RESOURCE_STATUS_VALUES = frozenset({"comparable", "estimated_comparable"})
STAGE_E_UNAVAILABLE_CLASS_VALUES = frozenset(
    {
        "duplicate_key",
        "missing_required_fields",
        "missing_quality_fields",
        "missing_resource_fields",
        "resource_not_comparable",
        "encoding_validation_failed",
        "insufficient_provenance",
    }
)
_STAGE_E_RUN_STATUS_ALLOWED = set(STAGE_E_RUN_STATUS_VALUES)
_STAGE_E_COMPARABLE_RESOURCE_STATUSES = set(STAGE_E_COMPARABLE_RESOURCE_STATUS_VALUES)
_STAGE_E_UNAVAILABLE_CLASS_ENUM = set(STAGE_E_UNAVAILABLE_CLASS_VALUES)
_STAGE_E_MASTER_REQUIRED = {
    "run_id",
    "manifest_slot",
    "instance_id",
    "family",
    "encoding",
    "decoder",
    "alpha_mode",
    "execution_mode",
    "run_status",
    "best_F",
    "topk_regret",
    "decision_distortion",
    "spearman_rho_F_G",
    "retained_energy_eta",
    "qubits",
    "gates",
    "depth",
    "resource_status",
    "resource_confidence",
    "has_verified_provenance",
}
_STAGE_E_UNAVAILABLE_REQUIRED = {
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
}


def _ensure_required_fields(row: dict[str, Any], required: set[str], *, label: str) -> None:
    missing = sorted(field for field in required if field not in row)
    if missing:
        raise ValueError(f"{label} row missing required fields: {missing}")


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_list_of_str(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(isinstance(item, str) for item in value)


def _ensure_fields(row: dict[str, Any], fields: set[str], *, label: str) -> None:
    missing = sorted(field for field in fields if field not in row)
    if missing:
        raise ValueError(f"alpha_validation row missing required {label} fields: {missing}")


def _is_nullable_int(value: Any) -> bool:
    return value is None or isinstance(value, int)


def _is_nullable_number(value: Any) -> bool:
    return value is None or isinstance(value, (int, float))


def validate_alpha_validation_row(row: dict[str, Any]) -> None:
    """Validate one Stage D alpha-validation row."""
    if not isinstance(row, dict):
        raise ValueError("alpha_validation row must be a dict")

    _ensure_fields(row, _STAGE_D_REQUIRED_IDENTITY, label="identity")
    _ensure_fields(row, _STAGE_D_REQUIRED_STATUS, label="status")
    _ensure_fields(row, _STAGE_D_REQUIRED_RAW_METRICS, label="raw metric")

    if row.get("matrix") != "matrix_b":
        raise ValueError("alpha_validation rows must have matrix='matrix_b'")
    if row.get("stage") != "alpha_validation":
        raise ValueError("alpha_validation rows must have stage='alpha_validation'")
    if row.get("family") != "C":
        raise ValueError("alpha_validation rows must have family='C'")
    if row.get("decoder") != "oracle":
        raise ValueError("alpha_validation rows must have decoder='oracle'")

    if row.get("execution_mode") not in {"coherent", "mixture"}:
        raise ValueError("alpha_validation rows must use execution_mode in {'coherent','mixture'}")
    run_status = row.get("run_status")
    if run_status in _STAGE_D_RUN_STATUS_LEGACY:
        raise ValueError(
            "alpha_validation rows must not use legacy run_status literals {'ok','error','skipped'}; "
            "use {'completed','failed','not_applicable'}"
        )
    if run_status not in _STAGE_D_RUN_STATUS_ALLOWED:
        raise ValueError(
            "alpha_validation rows must use run_status in {'completed','failed','not_applicable'}"
        )

    if not _is_nullable_int(row.get("trial_seed")):
        raise ValueError("trial_seed must be nullable int")
    if not _is_nullable_int(row.get("canonical_trial_seed")):
        raise ValueError("canonical_trial_seed must be nullable int")
    if not _is_nullable_int(row.get("seed")):
        raise ValueError("seed must be nullable int")
    if not isinstance(row.get("ell"), int) or int(row["ell"]) < 0:
        raise ValueError("ell must be a non-negative integer")
    if not isinstance(row.get("m_active"), int) or int(row["m_active"]) < 1:
        raise ValueError("m_active must be a positive integer")

    if not isinstance(row.get("in_experiment_matrix"), bool):
        raise ValueError("in_experiment_matrix must be bool")
    if not isinstance(row.get("coherent_supported"), bool):
        raise ValueError("coherent_supported must be bool")
    if row.get("oracle_exact") is not True:
        raise ValueError("oracle_exact must be true")

    run_error = row.get("run_error")
    if run_error is not None and not isinstance(run_error, str):
        raise ValueError("run_error must be nullable string")

    if not isinstance(row.get("status_reason"), str) or not row.get("status_reason"):
        raise ValueError("status_reason must be non-empty string")
    if not isinstance(row.get("status_reason_code"), str) or not row.get("status_reason_code"):
        raise ValueError("status_reason_code must be non-empty string")

    for metric in _STAGE_D_REQUIRED_RAW_METRICS:
        if not _is_nullable_number(row.get(metric)):
            raise ValueError(f"{metric} must be nullable number")

    if row.get("run_status") == "completed":
        for metric in _STAGE_D_REQUIRED_RAW_METRICS:
            if row.get(metric) is None:
                raise ValueError(f"{metric} must be populated for completed rows")


def validate_quality_cost_master_row(row: dict[str, Any]) -> None:
    """Validate one Stage E quality_cost_master row."""
    if not isinstance(row, dict):
        raise ValueError("quality_cost_master row must be a dict")
    _ensure_required_fields(row, _STAGE_E_MASTER_REQUIRED, label="quality_cost_master")

    for field in [
        "run_id",
        "manifest_slot",
        "instance_id",
        "family",
        "encoding",
        "decoder",
        "alpha_mode",
        "execution_mode",
        "resource_status",
    ]:
        if not _is_non_empty_str(row.get(field)):
            raise ValueError(f"{field} must be non-empty string in quality_cost_master row")

    if row.get("run_status") != "completed":
        raise ValueError("quality_cost_master rows must have run_status='completed'")
    if row.get("resource_status") not in _STAGE_E_COMPARABLE_RESOURCE_STATUSES:
        raise ValueError("quality_cost_master rows must have comparable resource_status")

    # Check if energy_eta is applicable for this encoding type
    energy_eta_applicable = row.get("energy_eta_applicable", True)

    required_numeric_fields = [
        "best_F",
        "topk_regret",
        "decision_distortion",
        "spearman_rho_F_G",
        "qubits",
        "gates",
        "depth",
    ]
    # Only require retained_energy_eta if applicable for this encoding
    if energy_eta_applicable:
        required_numeric_fields.append("retained_energy_eta")

    for field in required_numeric_fields:
        if not isinstance(row.get(field), (int, float)):
            raise ValueError(f"{field} must be numeric in quality_cost_master row")

    # retained_energy_eta can be None only if not applicable
    if not energy_eta_applicable and row.get("retained_energy_eta") is not None:
        if not isinstance(row.get("retained_energy_eta"), (int, float)):
            raise ValueError("retained_energy_eta must be numeric or None (if not applicable)")

    if row.get("resource_confidence") is not None and not isinstance(row.get("resource_confidence"), str):
        raise ValueError("resource_confidence must be nullable string in quality_cost_master row")

    if not isinstance(row.get("has_verified_provenance"), bool):
        raise ValueError("has_verified_provenance must be bool in quality_cost_master row")


def validate_quality_cost_unavailable_row(row: dict[str, Any]) -> None:
    """Validate one Stage E quality_cost_unavailable row."""
    if not isinstance(row, dict):
        raise ValueError("quality_cost_unavailable row must be a dict")
    _ensure_required_fields(row, _STAGE_E_UNAVAILABLE_REQUIRED, label="quality_cost_unavailable")

    for field in [
        "run_id",
        "instance_id",
        "family",
        "encoding",
        "decoder",
        "alpha_mode",
        "execution_mode",
        "unavailable_class",
        "unavailable_reason",
    ]:
        if not _is_non_empty_str(row.get(field)):
            raise ValueError(f"{field} must be non-empty string in quality_cost_unavailable row")

    unavailable_class = str(row.get("unavailable_class"))
    if unavailable_class not in _STAGE_E_UNAVAILABLE_CLASS_ENUM:
        raise ValueError(
            f"unavailable_class must be one of {sorted(_STAGE_E_UNAVAILABLE_CLASS_ENUM)}"
        )

    run_status = row.get("run_status")
    if run_status is not None and run_status not in _STAGE_E_RUN_STATUS_ALLOWED:
        raise ValueError(
            f"run_status must be nullable or one of {sorted(_STAGE_E_RUN_STATUS_ALLOWED)}"
        )

    if row.get("resource_status") is not None and not isinstance(row.get("resource_status"), str):
        raise ValueError("resource_status must be nullable string in quality_cost_unavailable row")

    if not isinstance(row.get("has_verified_provenance"), bool):
        raise ValueError("has_verified_provenance must be bool in quality_cost_unavailable row")

    for field in [
        "missing_required_fields",
        "missing_quality_fields",
        "missing_resource_fields",
    ]:
        if not _is_list_of_str(row.get(field)):
            raise ValueError(f"{field} must be list[str] in quality_cost_unavailable row")

    if unavailable_class == "missing_required_fields" and not row.get("missing_required_fields"):
        raise ValueError("missing_required_fields class requires non-empty missing_required_fields list")
    if unavailable_class == "missing_quality_fields" and not row.get("missing_quality_fields"):
        raise ValueError("missing_quality_fields class requires non-empty missing_quality_fields list")
    if unavailable_class == "missing_resource_fields" and not row.get("missing_resource_fields"):
        raise ValueError("missing_resource_fields class requires non-empty missing_resource_fields list")


def validate_quality_cost_master(rows: Any) -> None:
    """Validate Stage E quality_cost_master table payload."""
    if not isinstance(rows, list):
        raise ValueError("quality_cost_master must be a list of rows")
    for idx, row in enumerate(rows):
        try:
            validate_quality_cost_master_row(row)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"quality_cost_master row[{idx}] failed schema validation: {exc}") from exc


def validate_quality_cost_unavailable(rows: Any) -> None:
    """Validate Stage E quality_cost_unavailable table payload."""
    if not isinstance(rows, list):
        raise ValueError("quality_cost_unavailable must be a list of rows")
    for idx, row in enumerate(rows):
        try:
            validate_quality_cost_unavailable_row(row)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"quality_cost_unavailable row[{idx}] failed schema validation: {exc}") from exc


def validate_stage_e_quality_cost_payload(payload: Any) -> None:
    """Validate Stage E payload with explicit master/unavailable tables."""
    if not isinstance(payload, dict):
        raise ValueError("Stage E quality-cost payload must be a dict")
    if "quality_cost_master" not in payload:
        raise ValueError("Stage E payload missing key: quality_cost_master")
    if "quality_cost_unavailable" not in payload:
        raise ValueError("Stage E payload missing key: quality_cost_unavailable")

    validate_quality_cost_master(payload.get("quality_cost_master"))
    validate_quality_cost_unavailable(payload.get("quality_cost_unavailable"))
