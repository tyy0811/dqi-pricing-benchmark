from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure repo-root imports (e.g., `src.*`) work when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.coherent_reference import load_coherent_family
from src.family_s_manifest import load_family_s_manifest
from src.faithfulness import normalize_faithfulness_fields
from src.resources_compare import (
    ESTIMATED_COMPARABILITY_BASIS,
    build_stage_e_quality_cost_tables,
    normalize_resource_row_for_stage_e,
    normalize_stage_e_candidate_row,
    stage_e_duplicate_key,
    stage_e_duplicate_winner_key,
)
from src.report_schema import validate_quality_cost_master, validate_quality_cost_unavailable
from src.verify_encoding import (
    PAIR_VALIDATION_FLAG,
    validate_paired_encoding_row,
)


REPORT_SCHEMA_VERSION = "stage_b_1.0"
MANIFEST_VERSION = "1.0"
NORMALIZATION_VERSION = "stage_b_norm_v1"
PRODUCER = "scripts.make_report"

IDENTITY_FIELDS = [
    "slot_id",
    "run_id",
    "stage",
    "matrix",
    "family",
    "instance_id",
    "encoding",
    "decoder",
    "alpha_mode",
    "execution_mode",
    "trial_seed",
    "canonical_trial_seed",
    "seed",
]
STATUS_FIELDS = [
    "run_status",
    "status_reason_code",
    "status_reason",
    "in_experiment_matrix",
    "observed_run",
    "run_error",
]

QUALITY_REQUIRED_FIELDS = [
    "best_F",
    "topk_regret",
    "decision_distortion",
    "spearman_rho_F_G",
    "retained_energy_eta",
]
QUALITY_COMPARABLE_RESOURCE_STATUSES = {"comparable", "estimated_comparable"}
QUALITY_UNAVAILABLE_CLASS_ENUM = {
    "duplicate_key",
    "missing_required_fields",
    "missing_quality_fields",
    "missing_resource_fields",
    "resource_not_comparable",
    "encoding_validation_failed",
    "insufficient_provenance",
    "not_completed_status",
}

QUALITY_COST_ADMISSION_AUDIT_COLUMNS = [
    "run_id",
    "run_key",
    "comparison_key",
    "counterpart_key",
    "stage_e_admission_key",
    "matrix_norm",
    "stage_norm",
    "family_norm",
    "encoding_norm",
    "decoder_norm",
    "alpha_mode_norm",
    "execution_mode_norm",
    "admitted_to_master",
    "route_to_unavailable",
    "unavailable_class",
    "unavailable_reason_code",
    "unavailable_reason",
    "missing_required_fields",
    "missing_quality_fields",
    "missing_resource_fields",
    "run_status",
    "resource_status",
    "resource_confidence",
    "resource_comparability_basis",
    "has_verified_provenance",
]

QUALITY_COST_DUPLICATE_AUDIT_COLUMNS = [
    "duplicate_group_key",
    "winner_run_id",
    "winner_run_key",
    "loser_run_id",
    "loser_run_ids",
    "loser_disposition",
    "winner_ranking_completed_rank",
    "winner_ranking_verified_rank",
    "winner_ranking_manifest_slot",
    "winner_ranking_run_id",
]

DISPLAY_NOT_IN_MATRIX = "not_in_experiment_matrix"
AVAIL_AVAILABLE = "available"
AVAIL_FAILED = "failed_upstream"
AVAIL_NA = "not_applicable"

EXCLUDED_FROM_METRIC_COLUMNS = {
    "schema_version",
    "matrix",
    "run_id",
    "run_timestamp_utc",
    "run_status",
    "run_error",
    "instance_id",
    "family",
    "encoding",
    "decoder",
    "alpha_mode",
    "execution_mode",
    "stage",
    "trial_seed",
    "canonical_trial_seed",
    "seed",
    "metric_availability",
    "paired_comparison_id",
    "decoder_internal_mode",
    "decoder_alias_of",
    "coherent_mode_record",
    "coherent_vs_mixture_distribution_metric",
    "coherent_supported",
    "oracle_exact",
    "ell",
    "m_active",
    "source_artifact",
    "candidate_probabilities",
    "status_reason_code",
    "status_reason",
}

STAGE_D_BASELINE_MIN_PAIRS_DEFAULT = 8
STAGE_D_BASELINE_CANDIDATES: list[tuple[str, str, int]] = [
    ("wht_exact", "oracle", 1),
    ("ilp_derived_exact", "oracle", 2),
]
STAGE_D_SELECTED_REASON_PREFERRED = "preferred_available_meets_threshold"
STAGE_D_SELECTED_REASON_FALLBACK_BELOW = "preferred_below_threshold_used_fallback"
STAGE_D_SELECTED_REASON_FALLBACK_MISSING = "preferred_missing_used_fallback"
STAGE_D_SELECTED_REASON_NONE = "no_baseline_meets_threshold"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "<na>"}:
        return True
    return False


def _normalize_token(value: Any, *, missing: str = "unavailable") -> str:
    if value is None:
        return missing
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<na>"}:
        return missing
    return text


def _normalize_run_status_token(value: Any) -> str:
    raw = _normalize_token(value, missing="unavailable")
    mapping = {
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
    return mapping.get(raw, raw)


def _ensure_canonical_norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    field_map = [
        ("matrix", "matrix_norm"),
        ("stage", "stage_norm"),
        ("family", "family_norm"),
        ("encoding", "encoding_norm"),
        ("decoder", "decoder_norm"),
        ("alpha_mode", "alpha_mode_norm"),
        ("execution_mode", "execution_mode_norm"),
        ("resource_status", "resource_status_norm"),
    ]
    for src, dst in field_map:
        if src in out.columns:
            out[dst] = out[src].apply(_normalize_token)
        elif dst not in out.columns:
            out[dst] = "unavailable"

    if "run_status" in out.columns:
        out["run_status_norm"] = out["run_status"].apply(_normalize_run_status_token)
    elif "run_status_norm" not in out.columns:
        out["run_status_norm"] = "unavailable"

    return out


def _normalize_run_status_for_artifact(value: Any) -> str:
    normalized = _normalize_run_status_token(value)
    if normalized in {"skipped", "unsupported"}:
        return "not_applicable"
    return normalized


def _format_key_part(value: Any, *, missing: str = "unavailable") -> str:
    if _is_missing_value(value):
        return missing
    return str(value)


def _make_stage_e_admission_key(row: dict[str, Any] | pd.Series) -> str:
    key = stage_e_duplicate_key(dict(row))
    labels = ["instance_id", "encoding", "decoder", "alpha_mode", "execution_mode"]
    return "|".join(f"{label}={_format_key_part(value)}" for label, value in zip(labels, key))


def _make_matrix_a_comparison_key(row: dict[str, Any] | pd.Series) -> str | None:
    matrix_norm = _normalize_token(row.get("matrix_norm", row.get("matrix")))
    family_norm = _normalize_token(row.get("family_norm", row.get("family")))
    encoding_norm = _normalize_token(row.get("encoding_norm", row.get("encoding")))
    if matrix_norm != "matrix_a" or family_norm != "p":
        return None
    if encoding_norm not in {"wht_truncated", "ilp_derived"}:
        return None
    seed = normalize_seed_for_hash(row.get("canonical_trial_seed"))
    return "|".join(
        [
            matrix_norm,
            _format_key_part(row.get("stage_norm", row.get("stage"))),
            family_norm,
            _format_key_part(row.get("instance_id"), missing=""),
            _format_key_part(row.get("decoder_norm", row.get("decoder"))),
            _format_key_part(row.get("alpha_mode_norm", row.get("alpha_mode"))),
            _format_key_part(row.get("execution_mode_norm", row.get("execution_mode"))),
            f"canonical_trial_seed={seed}",
        ]
    )


def _persist_reporting_identity_columns(master_df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_canonical_norm_columns(master_df)
    if "run_key" not in out.columns:
        out["run_key"] = out.apply(
            lambda r: str(_first_non_null(r.get("slot_id"), r.get("run_id"), "")),
            axis=1,
        )
    out["comparison_key"] = out.apply(_make_matrix_a_comparison_key, axis=1)
    out["stage_e_admission_key"] = out.apply(_make_stage_e_admission_key, axis=1)
    return out


def _artifact_export_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_status" in out.columns:
        out["run_status"] = out["run_status"].apply(_normalize_run_status_for_artifact)
    if "run_status_norm" in out.columns:
        out["run_status_norm"] = out["run_status_norm"].apply(_normalize_run_status_for_artifact)
    return out


def normalize_seed_for_hash(seed: Any) -> str:
    s = _safe_int(seed)
    if s is None:
        return "NULL"
    return str(s)


def normalize_seed_value(seed: Any) -> int | None:
    return _safe_int(seed)


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def normalize_trial_seed_value(
    *,
    row: dict[str, Any] | pd.Series | None,
    run_meta: dict[str, Any] | None = None,
    report_meta: dict[str, Any] | None = None,
    summary_meta: dict[str, Any] | None = None,
) -> int | None:
    def _get(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, pd.Series):
            return obj.get(key)
        if isinstance(obj, dict):
            return obj.get(key)
        return None

    raw = _first_non_null(
        _get(row, "canonical_trial_seed"),
        _get(row, "trial_seed"),
        _get(row, "seed"),
        _get(run_meta, "canonical_trial_seed"),
        _get(run_meta, "trial_seed"),
        _get(run_meta, "seed"),
        _get(report_meta, "canonical_trial_seed"),
        _get(report_meta, "trial_seed"),
        _get(report_meta, "seed"),
        _get(summary_meta, "canonical_trial_seed"),
        _get(summary_meta, "trial_seed"),
        _get(summary_meta, "seed"),
        None,
    )
    return _safe_int(raw)


def _infer_stage(
    *,
    row: dict[str, Any] | pd.Series | None,
    run_meta: dict[str, Any] | None = None,
    report_meta: dict[str, Any] | None = None,
    summary_meta: dict[str, Any] | None = None,
    path_hint: str | None = None,
) -> str:
    def _get(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, pd.Series):
            return obj.get(key)
        if isinstance(obj, dict):
            return obj.get(key)
        return None

    for candidate in [
        _get(row, "stage"),
        _get(run_meta, "stage"),
        _get(report_meta, "stage"),
        _get(summary_meta, "stage"),
    ]:
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()

    # Path-based inference is a last resort only.
    if path_hint and "decoder_study" in str(path_hint):
        return "decoder_study"
    return "benchmark_matrix"


def _slot_hash_input(
    *,
    stage: str,
    matrix: str,
    family: str,
    instance_id: str,
    encoding: str,
    decoder: str,
    alpha_mode: str,
    execution_mode: str,
    trial_seed: Any,
    ell: Any = None,
    m_active: Any = None,
    run_id: str | None = None,
) -> str:
    seed_norm = normalize_seed_for_hash(trial_seed)
    base = (
        f"{stage}|{matrix}|{family}|{instance_id}|{encoding}|{decoder}|"
        f"{alpha_mode}|{execution_mode}|trial_seed={seed_norm}"
    )
    if _safe_int(ell) is not None:
        base += f"|ell={_safe_int(ell)}"
    if _safe_int(m_active) is not None:
        base += f"|m_active={_safe_int(m_active)}"
    if run_id is None or str(run_id).strip() == "":
        return base
    return f"{base}|run_id={run_id}"


def build_slot_id(**kwargs: Any) -> str:
    raw = _slot_hash_input(**kwargs)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_matrix_status(raw: Any, run_error: Any, observed: bool) -> tuple[str, str, str]:
    if not observed:
        return (
            "skipped",
            "planned_not_executed",
            "planned slot had no observed run row",
        )

    raw_str = str(raw).strip().lower() if raw is not None else ""
    err = "" if run_error is None else str(run_error)
    err_l = err.lower()

    if raw_str in {"ok", "completed"}:
        return "completed", "completed", "slot completed"

    if raw_str in {"error", "failed"}:
        return "failed", "upstream_failed", "observed run failed"

    if raw_str in {"not_applicable", "not-applicable", "na"}:
        return "not_applicable", "not_applicable", "observed not-applicable slot"

    if raw_str == "unavailable":
        return "unavailable", "upstream_unavailable", "observed unavailable slot"

    if raw_str in {"unsupported", "skipped"}:
        if any(tok in err_l for tok in ["not_supported", "not implemented", "not_implemented", "unsupported"]):
            return "skipped", "unsupported_combination", "observed unsupported combination"
        return "skipped", "intentionally_skipped", "observed skipped slot"

    return "failed", "parse_error", f"unrecognized raw run_status={raw!r}"


def _parse_metric_availability_map(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    return {}


def _normalize_availability_token(token: str | None) -> str:
    t = (token or "").strip().lower()
    if t == "available":
        return AVAIL_AVAILABLE
    if t in {"failed_upstream", "failed"}:
        return AVAIL_FAILED
    if t in {"not_applicable", "unavailable", "unsupported", "skipped"}:
        return AVAIL_NA
    return AVAIL_NA


def _infer_metric_columns(observed_df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in observed_df.columns:
        if col in EXCLUDED_FROM_METRIC_COLUMNS:
            continue
        if col.startswith("resource_"):
            continue
        if col.startswith("availability_"):
            continue
        if col.startswith("source_"):
            continue
        if col in {
            "_source_file_mtime",
            "_source_summary_path",
            "_source_report_path",
            "_source_row_index",
            "_normalized_seed",
            "_stage",
            "_slot_id",
            "_canonical_status",
            "_status_reason_code",
            "_status_reason",
        }:
            continue
        cols.append(col)
    return cols


def _to_serializable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    out: list[dict[str, Any]] = []
    for rec in records:
        clean: dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, (list, tuple, dict)):
                clean[k] = v
            elif isinstance(v, float) and math.isnan(v):
                clean[k] = None
            else:
                try:
                    if pd.isna(v):
                        clean[k] = None
                    else:
                        clean[k] = v
                except Exception:
                    clean[k] = v
        out.append(clean)
    return out


def _make_manifest_slots_matrix_a(config: dict[str, Any]) -> list[dict[str, Any]]:
    stage = str(config.get("stage") or "benchmark_matrix")
    matrix = "matrix_a"
    features = [int(x) for x in config.get("features", [3, 4, 5])]
    decoders = [str(x) for x in config.get("decoders", ["bruteforce", "bp1"])]
    alpha_modes = [str(x) for x in config.get("alpha_modes", ["uniform", "paper", "heuristic"])]
    seed = config.get("seed")

    slots: list[dict[str, Any]] = []
    for n_feat in features:
        instance_id = f"P{n_feat}"
        for decoder in decoders:
            for alpha_mode in alpha_modes:
                slots.append(
                    {
                        "stage": stage,
                        "matrix": matrix,
                        "family": "P",
                        "instance_id": instance_id,
                        "encoding": "wht_truncated",
                        "decoder": decoder,
                        "alpha_mode": alpha_mode,
                        "execution_mode": "mixture",
                        "trial_seed": normalize_seed_value(seed),
                        "seed": normalize_seed_value(seed),
                    }
                )
        slots.append(
            {
                "stage": stage,
                "matrix": matrix,
                "family": "P",
                "instance_id": instance_id,
                "encoding": "wht_truncated",
                "decoder": "bruteforce",
                "alpha_mode": "paper",
                "execution_mode": "coherent",
                "trial_seed": normalize_seed_value(seed),
                "seed": normalize_seed_value(seed),
            }
        )
        for decoder in decoders:
            for alpha_mode in alpha_modes:
                slots.append(
                    {
                        "stage": stage,
                        "matrix": matrix,
                        "family": "P",
                        "instance_id": instance_id,
                        "encoding": "ilp_derived",
                        "decoder": decoder,
                        "alpha_mode": alpha_mode,
                        "execution_mode": "mixture",
                        "trial_seed": normalize_seed_value(seed),
                        "seed": normalize_seed_value(seed),
                    }
                )
    return slots


def _ilp_feasible_c_instances() -> set[str]:
    feasible: set[str] = set()
    for rec in load_coherent_family():
        meta = rec.get("meta", {})
        instance_id = str(meta.get("instance_id"))
        ilp_ok = (
            str(meta.get("source_type", "")) == "pricing_ilp_exact"
            or bool(meta.get("exact_encoding_source_path"))
        )
        if ilp_ok:
            feasible.add(instance_id)
    return feasible


def _make_manifest_slots_matrix_b(config: dict[str, Any]) -> list[dict[str, Any]]:
    stage = str(config.get("stage") or "benchmark_matrix")
    matrix = "matrix_b"
    instance_ids = [str(x) for x in config.get("instance_ids", ["C1", "C2", "C3"])]
    alpha_modes = [str(x) for x in config.get("alpha_modes", ["uniform", "paper", "heuristic"])]
    execution_modes = [str(x) for x in config.get("execution_modes", ["mixture", "coherent"])]
    decoder_labels = [str(x) for x in config.get("decoder_labels") or config.get("decoders", ["oracle", "bp1"])]
    seed = config.get("seed")
    feasible_ilp = _ilp_feasible_c_instances()

    slots: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        encodings = ["wht_exact"]
        if instance_id in feasible_ilp:
            encodings.append("ilp_derived_exact")
        for encoding in encodings:
            for alpha_mode in alpha_modes:
                for execution_mode in execution_modes:
                    for d_idx, decoder in enumerate(decoder_labels):
                        slot_seed = None if seed is None else normalize_seed_value(seed) + d_idx
                        slots.append(
                            {
                                "stage": stage,
                                "matrix": matrix,
                                "family": "C",
                                "instance_id": instance_id,
                                "encoding": encoding,
                                "decoder": decoder,
                                "alpha_mode": alpha_mode,
                                "execution_mode": execution_mode,
                                "trial_seed": slot_seed,
                                "seed": slot_seed,
                            }
                        )
    return slots


def _make_manifest_slots_matrix_c(config: dict[str, Any]) -> list[dict[str, Any]]:
    stage = str(config.get("stage") or "benchmark_matrix")
    matrix = "matrix_c"
    alpha_modes = [str(x) for x in config.get("alpha_modes", ["paper", "uniform"])]
    decoders = [str(x) for x in config.get("decoders", ["bruteforce", "bp1", "bp+osd-lite", "oracle"])]
    seed = config.get("seed")

    family_s_manifest = load_family_s_manifest()
    if config.get("instance_ids") == "all" or config.get("instance_ids") is None:
        s_instance_ids = [str(x) for x in family_s_manifest.get("canonical_instance_order", [])]
    else:
        s_instance_ids = [str(x) for x in config.get("instance_ids", [])]

    slots: list[dict[str, Any]] = []
    for instance_id in s_instance_ids:
        for alpha_mode in alpha_modes:
            for d_idx, decoder in enumerate(decoders):
                slot_seed = None if seed is None else normalize_seed_value(seed) + d_idx
                slots.append(
                    {
                        "stage": stage,
                        "matrix": matrix,
                        "family": "S",
                        "instance_id": instance_id,
                        "encoding": "synthetic_structured_bv",
                        "decoder": decoder,
                        "alpha_mode": alpha_mode,
                        "execution_mode": "mixture",
                        "trial_seed": slot_seed,
                        "seed": slot_seed,
                    }
                )

    if bool(config.get("include_p_family", False)):
        p_features = [int(x) for x in config.get("p_features", [3, 4, 5])]
        for n_feat in p_features:
            instance_id = f"P{n_feat}"
            for alpha_mode in alpha_modes:
                for d_idx, decoder in enumerate(decoders):
                    slot_seed = None if seed is None else normalize_seed_value(seed) + d_idx
                    slots.append(
                        {
                            "stage": stage,
                            "matrix": matrix,
                            "family": "P",
                            "instance_id": instance_id,
                            "encoding": "wht_truncated_pricing",
                            "decoder": decoder,
                            "alpha_mode": alpha_mode,
                            "execution_mode": "mixture",
                            "trial_seed": slot_seed,
                            "seed": slot_seed,
                        }
                    )
    return slots


def _build_manifest_slots(results_root: Path, matrices: list[str]) -> tuple[pd.DataFrame, list[str]]:
    builders = {
        "matrix_a": _make_manifest_slots_matrix_a,
        "matrix_b": _make_manifest_slots_matrix_b,
        "matrix_c": _make_manifest_slots_matrix_c,
    }
    all_slots: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for matrix in matrices:
        manifest_path = results_root / matrix / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing core input: {manifest_path}")
        manifest_payload = json.loads(manifest_path.read_text())
        config = manifest_payload.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"missing manifest config for {matrix}")
        source_paths.append(str(manifest_path))
        if matrix not in builders:
            raise ValueError(f"unsupported matrix for Stage B report build: {matrix}")
        all_slots.extend(builders[matrix](config))

    if not all_slots:
        raise ValueError("manifest grid generation produced no slots")

    slot_df = pd.DataFrame(all_slots)
    for field in ["stage", "matrix", "family", "instance_id", "encoding", "decoder", "alpha_mode", "execution_mode"]:
        if field not in slot_df.columns:
            raise ValueError(f"malformed identity fields for planned slots: missing {field}")
        if slot_df[field].isna().any() or (slot_df[field].astype(str).str.strip() == "").any():
            raise ValueError(f"malformed identity fields for planned slots: empty {field}")

    if "trial_seed" not in slot_df.columns:
        slot_df["trial_seed"] = slot_df.get("seed")
    slot_df["seed"] = slot_df["seed"].apply(normalize_seed_value)
    slot_df["trial_seed"] = slot_df["trial_seed"].apply(normalize_seed_value)
    slot_df["canonical_trial_seed"] = slot_df["trial_seed"].apply(normalize_seed_value)
    slot_df["slot_id"] = slot_df.apply(
        lambda r: build_slot_id(
            stage=r["stage"],
            matrix=r["matrix"],
            family=r["family"],
            instance_id=r["instance_id"],
            encoding=r["encoding"],
            decoder=r["decoder"],
            alpha_mode=r["alpha_mode"],
            execution_mode=r["execution_mode"],
            trial_seed=r["trial_seed"],
            ell=r.get("ell"),
            m_active=r.get("m_active"),
        ),
        axis=1,
    )
    dup = slot_df["slot_id"].duplicated(keep=False)
    if dup.any():
        bad = slot_df.loc[
            dup,
            [
                "slot_id",
                "matrix",
                "stage",
                "instance_id",
                "encoding",
                "decoder",
                "alpha_mode",
                "execution_mode",
                "trial_seed",
            ],
        ]
        raise ValueError(f"duplicate slot_id in manifest grid generation:\n{bad.to_string(index=False)}")

    return slot_df, source_paths


def _load_observed_sources(results_root: Path, matrices: list[str]) -> pd.DataFrame:
    row_payloads: list[dict[str, Any]] = []

    for matrix in matrices:
        summary_path = results_root / "tables" / f"{matrix}_summary.csv"
        report_path = results_root / "reports" / f"{matrix}_report.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing core input: {summary_path}")
        if not report_path.exists():
            raise FileNotFoundError(f"missing core input: {report_path}")

        report_meta: dict[str, Any] = {}
        try:
            report_payload = json.loads(report_path.read_text())
            if isinstance(report_payload, dict):
                report_meta = report_payload
        except Exception:
            report_meta = {}

        summary_meta: dict[str, Any] = {}
        manifest_path = results_root / matrix / "manifest.json"
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text())
                if isinstance(manifest_payload, dict):
                    summary_meta = manifest_payload
            except Exception:
                summary_meta = {}

        df = pd.read_csv(summary_path)
        for idx, raw in enumerate(df.to_dict(orient="records")):
            rec = dict(raw)
            rec["matrix"] = str(rec.get("matrix") or matrix)
            rec["stage"] = _infer_stage(
                row=rec,
                run_meta={},
                report_meta=report_meta,
                summary_meta=summary_meta,
                path_hint=str(summary_path),
            )
            # For stages backed by authoritative append-only raw runs, avoid
            # re-ingesting projected compatibility rows from summary CSVs.
            if matrix == "matrix_b" and rec["stage"] == "alpha_validation":
                continue
            if matrix == "matrix_c" and rec["stage"] == "decoder_study":
                continue
            rec["trial_seed"] = normalize_trial_seed_value(
                row=rec, run_meta={}, report_meta=report_meta, summary_meta=summary_meta
            )
            rec["seed"] = normalize_seed_value(rec.get("seed"))
            rec["_source_summary_path"] = str(summary_path)
            rec["_source_report_path"] = str(report_path)
            rec["_source_file_mtime"] = float(summary_path.stat().st_mtime)
            rec["_source_row_index"] = int(idx)
            row_payloads.append(rec)

    # Stage C raw-run ingest: append-only rows under results/matrix_c/runs.
    if "matrix_c" in matrices:
        runs_dir = results_root / "matrix_c" / "runs"
        if runs_dir.exists():
            for run_file in sorted(runs_dir.glob("*.json")):
                try:
                    payload = json.loads(run_file.read_text())
                    if not isinstance(payload, dict):
                        continue
                except Exception:
                    continue
                rec = dict(payload)
                rec["matrix"] = str(rec.get("matrix") or "matrix_c")
                rec["stage"] = _infer_stage(
                    row=rec,
                    run_meta=payload,
                    report_meta={},
                    summary_meta={},
                    path_hint=str(run_file),
                )
                rec["trial_seed"] = normalize_trial_seed_value(
                    row=rec,
                    run_meta=payload,
                    report_meta={},
                    summary_meta={},
                )
                rec["seed"] = normalize_seed_value(rec.get("seed"))
                rec["_source_summary_path"] = str(run_file)
                rec["_source_report_path"] = str(results_root / "reports" / "matrix_c_report.json")
                rec["_source_file_mtime"] = float(run_file.stat().st_mtime)
                rec["_source_row_index"] = 0
                row_payloads.append(rec)

    # Stage D raw-run ingest: append-only rows under results/matrix_b/runs.
    if "matrix_b" in matrices:
        runs_dir = results_root / "matrix_b" / "runs"
        if runs_dir.exists():
            for run_file in sorted(runs_dir.glob("*.json")):
                try:
                    payload = json.loads(run_file.read_text())
                    if not isinstance(payload, dict):
                        continue
                except Exception:
                    continue
                rec = dict(payload)
                rec["matrix"] = str(rec.get("matrix") or "matrix_b")
                rec["stage"] = _infer_stage(
                    row=rec,
                    run_meta=payload,
                    report_meta={},
                    summary_meta={},
                    path_hint=str(run_file),
                )
                if rec["stage"] != "alpha_validation":
                    continue
                rec["trial_seed"] = normalize_trial_seed_value(
                    row=rec,
                    run_meta=payload,
                    report_meta={},
                    summary_meta={},
                )
                rec["seed"] = normalize_seed_value(rec.get("seed"))
                rec["_source_summary_path"] = str(run_file)
                rec["_source_report_path"] = str(results_root / "reports" / "matrix_b_report.json")
                rec["_source_file_mtime"] = float(run_file.stat().st_mtime)
                rec["_source_row_index"] = 0
                row_payloads.append(rec)

    if not row_payloads:
        return pd.DataFrame()

    observed = pd.DataFrame(row_payloads)
    observed["_stage"] = observed["stage"].astype(str)
    observed["_slot_id"] = observed.apply(
        lambda r: build_slot_id(
            stage=str(r["_stage"]),
            matrix=str(r["matrix"]),
            family=str(r["family"]),
            instance_id=str(r["instance_id"]),
            encoding=str(r["encoding"]),
            decoder=str(r["decoder"]),
            alpha_mode=str(r["alpha_mode"]),
            execution_mode=str(r["execution_mode"]),
            trial_seed=r.get("trial_seed"),
            ell=r.get("ell"),
            m_active=r.get("m_active"),
            run_id=(
                str(r.get("run_id"))
                if str(r.get("_stage", "")).strip() == "decoder_study"
                else None
            ),
        ),
        axis=1,
    )

    def _status_with_overrides(r: pd.Series) -> tuple[str, str, str]:
        base = _normalize_matrix_status(r.get("run_status"), r.get("run_error"), True)
        raw_reason_code = r.get("status_reason_code")
        raw_reason = r.get("status_reason")
        if not _is_missing_value(raw_reason_code):
            return (
                base[0],
                str(raw_reason_code).strip(),
                base[2] if _is_missing_value(raw_reason) else str(raw_reason).strip(),
            )
        if not _is_missing_value(raw_reason):
            return (base[0], base[1], str(raw_reason).strip())
        return base

    statuses = observed.apply(_status_with_overrides, axis=1, result_type="expand")
    observed["_canonical_status"] = statuses[0]
    observed["_status_reason_code"] = statuses[1]
    observed["_status_reason"] = statuses[2]
    observed["canonical_trial_seed"] = observed.apply(
        lambda r: normalize_seed_value(
            _first_non_null(
                r.get("canonical_trial_seed"),
                r.get("trial_seed"),
                r.get("seed"),
            )
        ),
        axis=1,
    )

    return observed


def _parse_timestamp_value(ts: Any) -> float:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return float("-inf")
    try:
        return pd.to_datetime(ts, utc=True).timestamp()
    except Exception:
        return float("-inf")


def _resolve_duplicates(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], int]:
    if len(candidates) == 1:
        return candidates[0], [str(candidates[0].get("run_id"))], 1

    def key_fn(rec: dict[str, Any]):
        status = str(rec.get("_canonical_status", "failed"))
        status_rank = 0 if status == "completed" else (1 if status == "failed" else 2)
        ts = _parse_timestamp_value(rec.get("run_timestamp_utc"))
        mtime = _safe_float(rec.get("_source_file_mtime"))
        if mtime is None:
            mtime = float("-inf")
        run_id = str(rec.get("run_id", ""))
        return (status_rank, -ts, -mtime, run_id)

    selected = sorted(candidates, key=key_fn)[0]
    all_ids = sorted(str(c.get("run_id")) for c in candidates)
    return selected, all_ids, len(candidates)


def _observed_stage_d_counterpart_key(row: pd.Series) -> str:
    seed = normalize_seed_for_hash(row.get("canonical_trial_seed"))
    ell = normalize_seed_for_hash(row.get("ell"))
    m_active = normalize_seed_for_hash(row.get("m_active"))
    return "|".join(
        [
            _normalize_token(row.get("matrix")),
            _normalize_token(row.get("_stage")),
            _normalize_token(row.get("family")),
            str(row.get("instance_id", "")),
            _normalize_token(row.get("encoding")),
            _normalize_token(row.get("decoder")),
            _normalize_token(row.get("alpha_mode")),
            f"ell={ell}",
            f"m_active={m_active}",
            f"canonical_trial_seed={seed}",
        ]
    )


def _validate_observed_stage_d_completed_counterpart_uniqueness(observed_df: pd.DataFrame) -> None:
    if observed_df.empty:
        return
    required = {"matrix", "_stage", "family", "execution_mode", "_canonical_status"}
    if not required.issubset(observed_df.columns):
        return
    stage_rows = observed_df[
        (observed_df["matrix"] == "matrix_b")
        & (observed_df["_stage"] == "alpha_validation")
        & (observed_df["family"] == "C")
        & (observed_df["execution_mode"].astype(str).str.strip().str.lower().isin(["mixture", "coherent"]))
        & (observed_df["_canonical_status"] == "completed")
    ].copy()
    if stage_rows.empty:
        return
    stage_rows["counterpart_key"] = stage_rows.apply(_observed_stage_d_counterpart_key, axis=1)
    stage_rows["execution_mode_norm"] = stage_rows["execution_mode"].apply(_normalize_token)
    counts = (
        stage_rows.groupby(["counterpart_key", "execution_mode_norm"], dropna=False)
        .size()
        .rename("n")
    )
    bad = counts[counts > 1]
    if bad.empty:
        return
    preview = bad.reset_index().head(20).to_dict(orient="records")
    raise ValueError(
        "duplicate completed counterpart rows detected in observed Stage D data: "
        f"{preview}"
    )


def _normalize_master_metrics(
    master_df: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    out = master_df.copy()

    for metric in metric_columns:
        if metric in out.columns and pd.api.types.is_bool_dtype(out[metric]):
            out[metric] = out[metric].astype(object)
        out[f"availability_{metric}"] = AVAIL_NA

    normalized_maps: list[dict[str, str]] = []
    for idx, row in out.iterrows():
        canonical = str(row.get("run_status", "failed"))
        row_map = _parse_metric_availability_map(row.get("metric_availability"))
        final_map: dict[str, str] = {}
        for metric in metric_columns:
            value = row.get(metric)
            value_present = not (pd.isna(value) if not isinstance(value, (dict, list)) else False)

            if canonical == "completed":
                if value_present:
                    avail = AVAIL_AVAILABLE
                else:
                    avail = _normalize_availability_token(row_map.get(metric))
                    if avail == AVAIL_AVAILABLE:
                        avail = AVAIL_NA
            elif canonical == "failed":
                avail = AVAIL_FAILED
            else:
                avail = AVAIL_NA

            if avail in {AVAIL_FAILED, AVAIL_NA}:
                out.at[idx, metric] = None
            out.at[idx, f"availability_{metric}"] = avail
            final_map[metric] = avail
        normalized_maps.append(final_map)

    out["metric_availability"] = normalized_maps
    return out


def _derive_decoder_gains(master_df: pd.DataFrame) -> pd.DataFrame:
    out = master_df.copy()
    if "gain_over_bp1" not in out.columns:
        out["gain_over_bp1"] = None
    if "gain_over_bruteforce" not in out.columns:
        out["gain_over_bruteforce"] = None

    key_cols = [
        "matrix",
        "stage",
        "family",
        "instance_id",
        "encoding",
        "alpha_mode",
        "execution_mode",
        "canonical_trial_seed",
    ]
    if not set(key_cols).issubset(out.columns):
        return out

    metric_col = "best_sampled_F" if "best_sampled_F" in out.columns else ("best_F" if "best_F" in out.columns else None)
    if metric_col is None:
        return out

    for _, idx in out.groupby(key_cols, dropna=False).groups.items():
        sub = out.loc[idx]
        bp1_val = None
        bf_val = None
        for _, row in sub.iterrows():
            if str(row.get("run_status")) != "completed":
                continue
            value = _safe_float(row.get(metric_col))
            if value is None:
                continue
            if str(row.get("decoder")) == "bp1":
                bp1_val = value
            if str(row.get("decoder")) == "bruteforce":
                bf_val = value
        for row_idx in idx:
            row = out.loc[row_idx]
            if str(row.get("run_status")) != "completed":
                out.at[row_idx, "gain_over_bp1"] = None
                out.at[row_idx, "gain_over_bruteforce"] = None
                continue
            cur = _safe_float(row.get(metric_col))
            if cur is None:
                out.at[row_idx, "gain_over_bp1"] = None
                out.at[row_idx, "gain_over_bruteforce"] = None
                continue
            out.at[row_idx, "gain_over_bp1"] = None if bp1_val is None else float(cur - bp1_val)
            out.at[row_idx, "gain_over_bruteforce"] = None if bf_val is None else float(cur - bf_val)
    return out


def _safe_probability_vector(raw: Any) -> list[float] | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    if isinstance(raw, list):
        try:
            return [float(x) for x in raw]
        except Exception:
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if isinstance(parsed, list):
            try:
                return [float(x) for x in parsed]
            except Exception:
                return None
    return None


def _distribution_tvd(a: Any, b: Any) -> float | None:
    pa = _safe_probability_vector(a)
    pb = _safe_probability_vector(b)
    if pa is None or pb is None:
        return None
    if len(pa) != len(pb):
        return None
    arr_a = pd.Series(pa, dtype="float64")
    arr_b = pd.Series(pb, dtype="float64")
    return float(0.5 * (arr_a.sub(arr_b).abs().sum()))


def _stage_d_counterpart_key(row: pd.Series) -> str:
    seed = normalize_seed_for_hash(row.get("canonical_trial_seed"))
    ell = normalize_seed_for_hash(row.get("ell"))
    m_active = normalize_seed_for_hash(row.get("m_active"))
    return "|".join(
        [
            str(row.get("matrix_norm", "unavailable")),
            str(row.get("stage_norm", "unavailable")),
            str(row.get("family_norm", "unavailable")),
            str(row.get("instance_id", "")),
            str(row.get("encoding_norm", "unavailable")),
            str(row.get("decoder_norm", "unavailable")),
            str(row.get("alpha_mode_norm", "unavailable")),
            f"ell={ell}",
            f"m_active={m_active}",
            f"canonical_trial_seed={seed}",
        ]
    )


def _stage_d_completed_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[
        (df["run_status_norm"] == "completed")
        & (df["execution_mode_norm"].isin(["mixture", "coherent"]))
    ].copy()


def _validate_stage_d_counterpart_uniqueness(df: pd.DataFrame) -> None:
    completed = _stage_d_completed_rows(df)
    if completed.empty:
        return
    counts = (
        completed.groupby(["counterpart_key", "execution_mode_norm"], dropna=False)
        .size()
        .rename("n")
    )
    bad = counts[counts > 1]
    if bad.empty:
        return
    preview = bad.reset_index().head(20).to_dict(orient="records")
    raise ValueError(
        "duplicate completed counterpart rows detected for "
        "(counterpart_key, execution_mode_norm): "
        f"{preview}"
    )


def _count_valid_stage_d_pairs(df: pd.DataFrame) -> int:
    completed = _stage_d_completed_rows(df)
    if completed.empty:
        return 0
    counts = (
        completed.groupby(["counterpart_key", "execution_mode_norm"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    mixture = counts["mixture"] if "mixture" in counts.columns else pd.Series(0, index=counts.index, dtype="int64")
    coherent = counts["coherent"] if "coherent" in counts.columns else pd.Series(0, index=counts.index, dtype="int64")
    valid = (mixture == 1) & (coherent == 1)
    return int(valid.sum())


def _select_stage_d_baseline(
    df: pd.DataFrame,
    *,
    min_pairs: int = STAGE_D_BASELINE_MIN_PAIRS_DEFAULT,
) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    pair_counts: dict[tuple[str, str], int] = {}
    availability: dict[tuple[str, str], bool] = {}

    for encoding_norm, decoder_norm, _rank in STAGE_D_BASELINE_CANDIDATES:
        candidate = df[
            (df["encoding_norm"] == encoding_norm)
            & (df["decoder_norm"] == decoder_norm)
        ].copy()
        pair_count = _count_valid_stage_d_pairs(candidate)
        available = not candidate.empty
        rejection = None
        if pair_count < int(min_pairs):
            rejection = "candidate_missing" if not available else "below_min_pairs"
        diagnostics.append(
            {
                "encoding_norm": encoding_norm,
                "decoder_norm": decoder_norm,
                "candidate_pair_count": int(pair_count),
                "candidate_available": bool(available),
                "candidate_rejection_reason": rejection,
            }
        )
        pair_counts[(encoding_norm, decoder_norm)] = int(pair_count)
        availability[(encoding_norm, decoder_norm)] = bool(available)

    preferred = STAGE_D_BASELINE_CANDIDATES[0]
    fallback = STAGE_D_BASELINE_CANDIDATES[1]
    pref_key = (preferred[0], preferred[1])
    fallback_key = (fallback[0], fallback[1])
    pref_pairs = int(pair_counts[pref_key])
    fallback_pairs = int(pair_counts[fallback_key])
    best_available_pair_count = int(max(pref_pairs, fallback_pairs))

    selected_encoding = None
    selected_decoder = None
    selected_pair_count: int | None = None
    selected_rank: int | None = None
    fallback_used = False
    selected_reason = STAGE_D_SELECTED_REASON_NONE

    if pref_pairs >= int(min_pairs):
        selected_encoding = preferred[0]
        selected_decoder = preferred[1]
        selected_pair_count = pref_pairs
        selected_rank = preferred[2]
        selected_reason = STAGE_D_SELECTED_REASON_PREFERRED
    elif fallback_pairs >= int(min_pairs):
        selected_encoding = fallback[0]
        selected_decoder = fallback[1]
        selected_pair_count = fallback_pairs
        selected_rank = fallback[2]
        fallback_used = True
        selected_reason = (
            STAGE_D_SELECTED_REASON_FALLBACK_MISSING
            if not availability[pref_key]
            else STAGE_D_SELECTED_REASON_FALLBACK_BELOW
        )

    return {
        "baseline_selector_policy": "fixed_priority",
        "baseline_min_pairs": int(min_pairs),
        "baseline_encoding_norm": selected_encoding,
        "baseline_decoder_norm": selected_decoder,
        "baseline_pair_count": selected_pair_count,
        "baseline_priority_rank": selected_rank,
        "baseline_selected_reason": selected_reason,
        "baseline_fallback_used": bool(fallback_used),
        "best_available_pair_count": best_available_pair_count,
        "candidate_diagnostics": diagnostics,
    }


def _assert_stage_d_selected_baseline_invariants(df: pd.DataFrame) -> None:
    if df.empty:
        return
    selected_rows = df[df["is_stage_d_selected_baseline"] == True].copy()
    selected_meta = df[
        df["baseline_encoding_norm"].notna()
        & df["baseline_decoder_norm"].notna()
    ].copy()
    if selected_meta.empty and selected_rows.empty:
        return

    meta_pairs = set(
        zip(
            selected_meta["baseline_encoding_norm"].astype(str),
            selected_meta["baseline_decoder_norm"].astype(str),
        )
    )
    if len(meta_pairs) > 1:
        raise ValueError(f"multiple_selected_baselines: {sorted(meta_pairs)}")
    if selected_meta.empty and not selected_rows.empty:
        raise ValueError("selected_baseline_mismatch: selected rows exist without selected metadata")
    if not selected_meta.empty and selected_rows.empty:
        raise ValueError("selected_baseline_mismatch: selected metadata exists but no rows are selected")

    selected_pair = next(iter(meta_pairs))
    row_pairs = set(
        zip(
            selected_rows["encoding_norm"].astype(str),
            selected_rows["decoder_norm"].astype(str),
        )
    )
    if len(row_pairs) > 1:
        raise ValueError(f"multiple_selected_baselines: {sorted(row_pairs)}")
    if selected_pair not in row_pairs:
        raise ValueError(
            "selected_baseline_mismatch: selected rows do not match "
            f"metadata pair {selected_pair}"
        )


def _derive_alpha_validation_fields(master_df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_canonical_norm_columns(master_df)
    required = {
        "matrix",
        "stage",
        "family",
        "instance_id",
        "encoding",
        "alpha_mode",
        "execution_mode",
        "canonical_trial_seed",
        "best_F",
        "best_top_G_sampled_F",
        "postselection_success",
    }
    if not required.issubset(out.columns):
        return out

    for col in [
        "coherent_vs_mixture_delta_best_F",
        "coherent_vs_mixture_delta_best_top_G_sampled_F",
        "coherent_vs_mixture_delta_postselection_success",
        "coherent_vs_mixture_distribution_distance",
        "alpha_ranking_within_instance",
    ]:
        if col not in out.columns:
            out[col] = None

    mask_stage = (
        (out["matrix"] == "matrix_b")
        & (out["stage"] == "alpha_validation")
        & (out["family"] == "C")
        & (out["decoder"] == "oracle")
    )
    stage_rows = out[mask_stage].copy()
    if stage_rows.empty:
        # Keep explicit selector columns present even when Stage D rows are absent.
        for key, default in [
            ("counterpart_key", None),
            ("baseline_selector_policy", None),
            ("baseline_min_pairs", None),
            ("baseline_encoding_norm", None),
            ("baseline_decoder_norm", None),
            ("baseline_pair_count", None),
            ("baseline_priority_rank", None),
            ("baseline_selected_reason", None),
            ("baseline_fallback_used", None),
            ("best_available_pair_count", None),
            ("is_stage_d_selected_baseline", False),
        ]:
            if key not in out.columns:
                out[key] = default
        return out

    stage_rows["counterpart_key"] = stage_rows.apply(_stage_d_counterpart_key, axis=1)
    _validate_stage_d_counterpart_uniqueness(stage_rows)

    for row_idx, row in stage_rows.iterrows():
        out.at[row_idx, "counterpart_key"] = row["counterpart_key"]

    completed = _stage_d_completed_rows(stage_rows)
    if not completed.empty:
        counts = (
            completed.groupby(["counterpart_key", "execution_mode_norm"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        mixture = counts["mixture"] if "mixture" in counts.columns else pd.Series(0, index=counts.index, dtype="int64")
        coherent = counts["coherent"] if "coherent" in counts.columns else pd.Series(0, index=counts.index, dtype="int64")
        valid_keys = set(counts.index[(mixture == 1) & (coherent == 1)].tolist())
        for counterpart_key in valid_keys:
            subset = completed[completed["counterpart_key"] == counterpart_key]
            coh = subset[subset["execution_mode_norm"] == "coherent"].iloc[0]
            mix = subset[subset["execution_mode_norm"] == "mixture"].iloc[0]
            coh_best = _safe_float(coh.get("best_F"))
            mix_best = _safe_float(mix.get("best_F"))
            coh_topg = _safe_float(coh.get("best_top_G_sampled_F"))
            mix_topg = _safe_float(mix.get("best_top_G_sampled_F"))
            coh_post = _safe_float(coh.get("postselection_success"))
            mix_post = _safe_float(mix.get("postselection_success"))
            delta_best = None if coh_best is None or mix_best is None else float(coh_best - mix_best)
            delta_topg = None if coh_topg is None or mix_topg is None else float(coh_topg - mix_topg)
            delta_post = None if coh_post is None or mix_post is None else float(coh_post - mix_post)
            tvd = _distribution_tvd(coh.get("candidate_probabilities"), mix.get("candidate_probabilities"))
            key_rows = stage_rows[stage_rows["counterpart_key"] == counterpart_key].index
            for row_idx in key_rows:
                out.at[row_idx, "coherent_vs_mixture_delta_best_F"] = delta_best
                out.at[row_idx, "coherent_vs_mixture_delta_best_top_G_sampled_F"] = delta_topg
                out.at[row_idx, "coherent_vs_mixture_delta_postselection_success"] = delta_post
                out.at[row_idx, "coherent_vs_mixture_distribution_distance"] = tvd

    rank_keys = [
        "matrix",
        "stage",
        "family",
        "instance_id",
        "encoding",
        "execution_mode",
        "ell",
        "m_active",
        "canonical_trial_seed",
    ]
    if set(rank_keys).issubset(stage_rows.columns):
        for _, idx in out[mask_stage].groupby(rank_keys, dropna=False).groups.items():
            sub = out.loc[idx]
            completed = sub[sub["run_status"] == "completed"].copy()
            completed["best_F_numeric"] = completed["best_F"].apply(_safe_float)
            completed = completed[completed["best_F_numeric"].notna()].copy()
            if completed.empty:
                continue
            completed = completed.sort_values(["best_F_numeric", "alpha_mode"], ascending=[False, True])
            rank = 1
            prev_val = None
            for row_idx, row in completed.iterrows():
                cur = float(row["best_F_numeric"])
                if prev_val is not None and cur < prev_val:
                    rank += 1
                out.at[row_idx, "alpha_ranking_within_instance"] = int(rank)
                prev_val = cur

    selector = _select_stage_d_baseline(stage_rows, min_pairs=STAGE_D_BASELINE_MIN_PAIRS_DEFAULT)
    stage_selector_cols: dict[str, Any] = {
        "baseline_selector_policy": selector["baseline_selector_policy"],
        "baseline_min_pairs": selector["baseline_min_pairs"],
        "baseline_encoding_norm": selector["baseline_encoding_norm"],
        "baseline_decoder_norm": selector["baseline_decoder_norm"],
        "baseline_pair_count": selector["baseline_pair_count"],
        "baseline_priority_rank": selector["baseline_priority_rank"],
        "baseline_selected_reason": selector["baseline_selected_reason"],
        "baseline_fallback_used": selector["baseline_fallback_used"],
        "best_available_pair_count": selector["best_available_pair_count"],
    }
    for col in list(stage_selector_cols.keys()) + ["is_stage_d_selected_baseline"]:
        if col not in out.columns:
            out[col] = None

    selected_encoding = selector["baseline_encoding_norm"]
    selected_decoder = selector["baseline_decoder_norm"]

    for row_idx in stage_rows.index:
        for col, val in stage_selector_cols.items():
            out.at[row_idx, col] = val
        is_selected = (
            selected_encoding is not None
            and selected_decoder is not None
            and str(out.at[row_idx, "encoding_norm"]) == str(selected_encoding)
            and str(out.at[row_idx, "decoder_norm"]) == str(selected_decoder)
        )
        out.at[row_idx, "is_stage_d_selected_baseline"] = bool(is_selected)

    _assert_stage_d_selected_baseline_invariants(out.loc[stage_rows.index].copy())

    return out


def _load_paired_pricing_artifact(
    *,
    results_root: Path | None,
    instance_id: str,
    cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    if results_root is None:
        return None
    key = str(instance_id)
    if key in cache:
        return cache[key]
    paired_path = results_root / "paired_pricing" / f"{key}.paired.json"
    if not paired_path.exists():
        cache[key] = None
        return None
    try:
        payload = json.loads(paired_path.read_text())
        cache[key] = payload if isinstance(payload, dict) else None
    except Exception:
        cache[key] = None
    return cache[key]


def _stage_e_duplicate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("matrix"),
        row.get("stage"),
        row.get("instance_id"),
        row.get("encoding"),
        row.get("decoder"),
        row.get("alpha_mode"),
        row.get("execution_mode"),
        row.get("canonical_trial_seed"),
    )


def _stage_e_duplicate_winner_key(row: dict[str, Any]) -> tuple[Any, ...]:
    completed_rank = 0 if str(row.get("run_status")) == "completed" else 1
    verified_rank = 0 if bool(row.get("has_verified_provenance", False)) else 1
    manifest_slot = str(row.get("manifest_slot") or row.get("slot_id") or "~")
    run_id = str(row.get("run_id") or "~")
    return (completed_rank, verified_rank, manifest_slot, run_id)


def _stage_e_unavailable_row(
    row: dict[str, Any],
    *,
    unavailable_class: str,
    unavailable_reason_code: str,
    unavailable_reason: str,
) -> dict[str, Any]:
    out = dict(row)
    out["unavailable_class"] = unavailable_class
    out["unavailable_reason_code"] = unavailable_reason_code
    out["unavailable_reason"] = unavailable_reason
    return out


def _stage_e_classify_row(row: dict[str, Any]) -> tuple[bool, str | None, str | None, str | None]:
    required_identity = [
        "run_id",
        "matrix",
        "stage",
        "instance_id",
        "encoding",
        "decoder",
        "alpha_mode",
        "execution_mode",
        "canonical_trial_seed",
    ]
    missing_identity = [
        field
        for field in required_identity
        if _is_missing_value(row.get(field))
    ]
    if missing_identity:
        return (
            False,
            "missing_required_fields",
            "missing_required_fields",
            f"missing required fields: {', '.join(missing_identity)}",
        )

    if str(row.get("run_status")) != "completed":
        reason_code = str(row.get("status_reason_code") or "not_completed_status")
        reason = str(row.get("status_reason") or "row is not completed")
        return (False, "not_completed_status", reason_code, reason)

    if not bool(row.get("_encoding_ok", True)):
        return (
            False,
            "encoding_validation_failed",
            str(row.get("_encoding_reason_code") or "encoding_validation_failed"),
            str(row.get("_encoding_reason") or "paired encoding validation failed"),
        )

    if bool(row.get("_faithfulness_blocked", False)):
        return (
            False,
            "insufficient_provenance",
            str(row.get("_faithfulness_reason_code") or "insufficient_provenance"),
            str(row.get("_faithfulness_reason") or "faithfulness computation blocked by provenance"),
        )

    missing_quality = [
        field
        for field in QUALITY_REQUIRED_FIELDS
        if _safe_float(row.get(field)) is None
    ]
    if missing_quality:
        return (
            False,
            "missing_quality_fields",
            "missing_quality_fields",
            f"missing quality fields: {', '.join(missing_quality)}",
        )

    qubits = _safe_float(row.get("qubits"))
    gates = _safe_float(row.get("gates"))
    depth = _safe_float(row.get("depth"))
    if qubits is None or gates is None or depth is None:
        return (
            False,
            "missing_resource_fields",
            "missing_resource_fields",
            "missing qubits/gates/depth resource values",
        )

    resource_status = str(row.get("resource_status") or "")
    if resource_status not in QUALITY_COMPARABLE_RESOURCE_STATUSES:
        code = str(row.get("resource_not_comparable_reason") or "resource_not_comparable")
        return (
            False,
            "resource_not_comparable",
            code,
            f"resource status is {resource_status or 'unknown'}",
        )
    if resource_status == "estimated_comparable":
        basis = row.get("resource_comparability_basis")
        if basis not in ESTIMATED_COMPARABILITY_BASIS:
            return (
                False,
                "resource_not_comparable",
                "missing_comparability_basis",
                "estimated resources missing valid comparability basis",
            )

    return (True, None, None, None)


def _build_quality_cost_tables(
    master_metrics: pd.DataFrame,
    *,
    results_root: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if master_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()

    candidate_rows = _to_serializable_records(master_metrics)
    paired_artifacts = _paired_artifacts_by_instance_for_stage_e(candidate_rows, results_root=results_root)
    payload = build_stage_e_quality_cost_tables(
        candidate_rows,
        paired_artifacts_by_instance=paired_artifacts,
        strict_validate=True,
    )
    return (
        pd.DataFrame(payload["quality_cost_master"]),
        pd.DataFrame(payload["quality_cost_unavailable"]),
    )


def _paired_artifacts_by_instance_for_stage_e(
    candidate_rows: list[dict[str, Any]],
    *,
    results_root: Path | None,
) -> dict[str, dict[str, Any] | None]:
    cache: dict[str, dict[str, Any] | None] = {}
    if results_root is None:
        return cache
    instance_ids = {
        str(row.get("instance_id"))
        for row in candidate_rows
        if bool(row.get(PAIR_VALIDATION_FLAG, False)) and not _is_missing_value(row.get("instance_id"))
    }
    for instance_id in sorted(instance_ids):
        _load_paired_pricing_artifact(results_root=results_root, instance_id=instance_id, cache=cache)
    return cache


def _ranking_inputs_from_row(row: dict[str, Any]) -> dict[str, Any]:
    ranking = stage_e_duplicate_winner_key(row)
    return {
        "winner_ranking_completed_rank": ranking[0],
        "winner_ranking_verified_rank": ranking[1],
        "winner_ranking_manifest_slot": ranking[2],
        "winner_ranking_run_id": ranking[3],
    }


def _build_stage_e_audits(
    master_metrics: pd.DataFrame,
    *,
    results_root: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if master_metrics.empty:
        return (
            pd.DataFrame(columns=QUALITY_COST_ADMISSION_AUDIT_COLUMNS),
            pd.DataFrame(columns=QUALITY_COST_DUPLICATE_AUDIT_COLUMNS),
        )

    candidate_rows = _to_serializable_records(master_metrics)
    paired_artifacts = _paired_artifacts_by_instance_for_stage_e(candidate_rows, results_root=results_root)

    admission_rows: list[dict[str, Any]] = []
    admitted_for_dedup: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []

    for candidate in candidate_rows:
        paired_artifact = paired_artifacts.get(str(candidate.get("instance_id")))
        normalized = normalize_stage_e_candidate_row(candidate, paired_artifact=paired_artifact)
        normalized_row = dict(normalized["normalized_row_candidate"])
        enriched_row = {**candidate, **normalized_row}
        admission = dict(normalized["admission"])

        admission_rows.append(
            {
                "run_id": candidate.get("run_id"),
                "run_key": candidate.get("run_key"),
                "comparison_key": candidate.get("comparison_key"),
                "counterpart_key": candidate.get("counterpart_key"),
                "stage_e_admission_key": candidate.get("stage_e_admission_key"),
                "matrix_norm": candidate.get("matrix_norm"),
                "stage_norm": candidate.get("stage_norm"),
                "family_norm": candidate.get("family_norm"),
                "encoding_norm": candidate.get("encoding_norm"),
                "decoder_norm": candidate.get("decoder_norm"),
                "alpha_mode_norm": candidate.get("alpha_mode_norm"),
                "execution_mode_norm": candidate.get("execution_mode_norm"),
                "admitted_to_master": bool(admission.get("admitted_to_master", False)),
                "route_to_unavailable": bool(admission.get("route_to_unavailable", False)),
                "unavailable_class": admission.get("unavailable_class"),
                "unavailable_reason_code": admission.get("unavailable_reason_code"),
                "unavailable_reason": admission.get("unavailable_reason"),
                "missing_required_fields": list(admission.get("missing_identity_fields", [])),
                "missing_quality_fields": list(admission.get("missing_quality_fields", [])),
                "missing_resource_fields": list(
                    normalized.get("unavailable_row_candidate", {}).get("missing_resource_fields", [])
                    if normalized.get("unavailable_row_candidate")
                    else []
                ),
                "run_status": normalized_row.get("run_status"),
                "resource_status": normalized_row.get("resource_status"),
                "resource_confidence": normalized_row.get("resource_confidence"),
                "resource_comparability_basis": normalized_row.get("resource_comparability_basis"),
                "has_verified_provenance": normalized_row.get("has_verified_provenance"),
            }
        )

        if bool(admission.get("admitted_to_master", False)):
            admitted_for_dedup.append(enriched_row)

    duplicate_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in admitted_for_dedup:
        duplicate_groups[stage_e_duplicate_key(row)].append(row)

    for group_key, rows in sorted(duplicate_groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        if len(rows) < 2:
            continue
        ranked = sorted(rows, key=stage_e_duplicate_winner_key)
        winner = ranked[0]
        loser_ids = [loser.get("run_id") for loser in ranked[1:]]
        ranking_inputs = _ranking_inputs_from_row(winner)
        for loser in ranked[1:]:
            duplicate_rows.append(
                {
                    "duplicate_group_key": "|".join(_format_key_part(value) for value in group_key),
                    "winner_run_id": winner.get("run_id"),
                    "winner_run_key": winner.get("run_key"),
                    "loser_run_id": loser.get("run_id"),
                    "loser_run_ids": loser_ids,
                    "loser_disposition": "quality_cost_unavailable:duplicate_key",
                    **ranking_inputs,
                }
            )

    return (
        pd.DataFrame(admission_rows, columns=QUALITY_COST_ADMISSION_AUDIT_COLUMNS),
        pd.DataFrame(duplicate_rows, columns=QUALITY_COST_DUPLICATE_AUDIT_COLUMNS),
    )


def _build_matrix_a_pairing_audit(master_metrics: pd.DataFrame) -> pd.DataFrame:
    if master_metrics.empty or "comparison_key" not in master_metrics.columns:
        return pd.DataFrame()
    focus = master_metrics[
        (master_metrics["matrix_norm"] == "matrix_a")
        & (master_metrics["family_norm"] == "p")
        & (master_metrics["comparison_key"].notna())
    ].copy()
    if focus.empty:
        return pd.DataFrame()

    counts = (
        focus.groupby("comparison_key", dropna=False)["encoding_norm"]
        .agg(
            comparison_group_size="size",
            wht_count=lambda s: int((s == "wht_truncated").sum()),
            ilp_count=lambda s: int((s == "ilp_derived").sum()),
        )
        .reset_index()
    )
    focus = focus.merge(counts, on="comparison_key", how="left")
    focus["has_wht_peer"] = focus["wht_count"] > 0
    focus["has_ilp_peer"] = focus["ilp_count"] > 0
    focus["pairable_across_encodings"] = focus["has_wht_peer"] & focus["has_ilp_peer"]
    return focus[
        [
            "run_id",
            "run_key",
            "comparison_key",
            "matrix_norm",
            "stage_norm",
            "family_norm",
            "encoding_norm",
            "decoder_norm",
            "alpha_mode_norm",
            "execution_mode_norm",
            "run_status_norm",
            "comparison_group_size",
            "wht_count",
            "ilp_count",
            "has_wht_peer",
            "has_ilp_peer",
            "pairable_across_encodings",
        ]
    ].copy()


def _build_master_tables(results_root: Path, matrices: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    slot_df, source_manifest_paths = _build_manifest_slots(results_root, matrices)
    observed_df = _load_observed_sources(results_root, matrices)
    warnings: list[str] = []

    # Stage D guardrail: alpha_validation rows must be matrix_b + family C + decoder oracle.
    if not observed_df.empty and {"_stage", "matrix", "family", "decoder", "run_id"}.issubset(observed_df.columns):
        bad_mask = (
            (observed_df["_stage"] == "alpha_validation")
            & (
                (observed_df["matrix"] != "matrix_b")
                | (observed_df["family"] != "C")
                | (observed_df["decoder"] != "oracle")
            )
        )
        if bool(bad_mask.any()):
            bad_rows = observed_df.loc[bad_mask, ["run_id", "matrix", "family", "decoder"]]
            for _, bad in bad_rows.iterrows():
                warnings.append(
                    "quarantined invalid alpha_validation row: "
                    f"run_id={bad.get('run_id')} matrix={bad.get('matrix')} "
                    f"family={bad.get('family')} decoder={bad.get('decoder')}"
                )
            observed_df = observed_df.loc[~bad_mask].copy()

    _validate_observed_stage_d_completed_counterpart_uniqueness(observed_df)

    # Stage C decoder-study rows are authoritative raw runs and may not exist in manifest grids.
    if not observed_df.empty:
        observed_slots = observed_df[
            [
                "_slot_id",
                "_stage",
                "matrix",
                "family",
                "instance_id",
                "encoding",
                "decoder",
                "alpha_mode",
                "execution_mode",
                "trial_seed",
                "canonical_trial_seed",
                "seed",
            ]
        ].copy()
        observed_slots = observed_slots.rename(columns={"_slot_id": "slot_id", "_stage": "stage"})
        observed_slots["trial_seed"] = observed_slots["trial_seed"].apply(normalize_seed_value)
        observed_slots["canonical_trial_seed"] = observed_slots["canonical_trial_seed"].apply(normalize_seed_value)
        observed_slots["seed"] = observed_slots["seed"].apply(normalize_seed_value)
        observed_slots = observed_slots.drop_duplicates(subset=["slot_id"]).reset_index(drop=True)
        slot_df = pd.concat([slot_df, observed_slots], ignore_index=True, sort=False)
        slot_df = slot_df.drop_duplicates(subset=["slot_id"], keep="first").reset_index(drop=True)

    observed_by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in _to_serializable_records(observed_df):
        observed_by_slot[str(rec["_slot_id"])].append(rec)

    rows: list[dict[str, Any]] = []
    for slot in _to_serializable_records(slot_df):
        slot_id = str(slot["slot_id"])
        candidates = observed_by_slot.get(slot_id, [])

        if candidates:
            selected, dup_ids, dup_count = _resolve_duplicates(candidates)
            if dup_count > 1:
                warnings.append(
                    f"duplicate observed rows collapsed for slot_id={slot_id}; candidates={dup_ids}"
                )
            canonical_status = selected.get("_canonical_status", "failed")
            reason_code = selected.get("_status_reason_code", "upstream_failed")
            reason = selected.get("_status_reason", "observed run failed")
            run_id = selected.get("run_id")
            run_error = selected.get("run_error")
            observed_run = True
            base = dict(selected)
        else:
            selected = {}
            dup_ids = []
            dup_count = 0
            canonical_status = "skipped"
            reason_code = "planned_not_executed"
            reason = "planned slot had no observed run row"
            run_id = f"planned::{slot_id}"
            run_error = None
            observed_run = False
            base = {}

        row: dict[str, Any] = {
            "slot_id": slot_id,
            "manifest_slot": slot_id,
            "run_id": run_id,
            "stage": slot["stage"],
            "matrix": slot["matrix"],
            "family": slot["family"],
            "instance_id": slot["instance_id"],
            "encoding": slot["encoding"],
            "decoder": slot["decoder"],
            "alpha_mode": slot["alpha_mode"],
            "execution_mode": slot["execution_mode"],
            "trial_seed": slot.get("trial_seed"),
            "canonical_trial_seed": normalize_seed_value(
                _first_non_null(
                    slot.get("canonical_trial_seed"),
                    slot.get("trial_seed"),
                    slot.get("seed"),
                )
            ),
            "seed": slot["seed"],
            "run_status": canonical_status,
            "status_reason_code": reason_code,
            "status_reason": reason,
            "in_experiment_matrix": True,
            "observed_run": bool(observed_run),
            "run_error": run_error,
            "duplicate_run_ids": dup_ids,
            "duplicate_observation_count": int(dup_count),
            "observed_metrics_path": base.get("_source_summary_path"),
            "observed_report_path": base.get("_source_report_path"),
            "producer": PRODUCER,
            "source_matrix_manifest": str(results_root / slot["matrix"] / "manifest.json"),
            "source_row_index": base.get("_source_row_index"),
            "ingested_at_utc": _now_utc_iso(),
            "observed_at_utc": base.get("run_timestamp_utc"),
            "normalization_version": NORMALIZATION_VERSION,
        }

        # Carry forward observed columns for compatibility projection.
        for key, value in base.items():
            if key.startswith("_"):
                continue
            if key.startswith("availability_"):
                continue
            if key in row:
                continue
            row[key] = value

        rows.append(row)

    master_metrics = pd.DataFrame(rows)

    metric_columns = _infer_metric_columns(observed_df)
    for metric in metric_columns:
        if metric not in master_metrics.columns:
            master_metrics[metric] = None
    master_metrics = _normalize_master_metrics(master_metrics, metric_columns)
    master_metrics = _derive_decoder_gains(master_metrics)
    master_metrics = _derive_alpha_validation_fields(master_metrics)
    master_metrics = _persist_reporting_identity_columns(master_metrics)

    # Resources table: one row per slot with explicit resource provenance/scope.
    resource_cols = [c for c in master_metrics.columns if c.startswith("resource_")]
    qiskit_resource_cols = [
        c
        for c in [
            "qiskit_subset_resource_report",
            "qiskit_transpiled_structural_count",
            "qiskit_subset_transpiled_depth",
            "qiskit_subset_cx_count",
            "qiskit_subset_qubit_count",
            "qiskit_subset_status",
        ]
        if c in master_metrics.columns
    ]
    resources = master_metrics[
        [
            "slot_id",
            "run_id",
            "stage",
            "matrix",
            "family",
            "instance_id",
            "encoding",
            "decoder",
            "alpha_mode",
            "execution_mode",
            "trial_seed",
            "canonical_trial_seed",
            "seed",
            "run_status",
            "status_reason_code",
            "status_reason",
            "in_experiment_matrix",
            "observed_run",
            "run_error",
            "observed_metrics_path",
            "observed_report_path",
            "producer",
            "source_matrix_manifest",
            "source_row_index",
            "ingested_at_utc",
            "observed_at_utc",
            "normalization_version",
        ]
        + resource_cols
        + qiskit_resource_cols
    ].copy()
    resources["resource_scope"] = resources.apply(
        lambda r: "encoding_level" if r.get("observed_run") else None,
        axis=1,
    )
    resources["resource_key"] = resources.apply(
        lambda r: f"{r['instance_id']}|{r['encoding']}|{r['execution_mode']}" if r.get("observed_run") else None,
        axis=1,
    )
    resources["resource_confidence"] = resources.get("resource_decode_uncompute_confidence")
    resources["resource_status"] = resources.get("resource_decode_uncompute_status")
    resources["resource_reason_code"] = resources.get("status_reason_code")
    resources["resource_model_version"] = REPORT_SCHEMA_VERSION
    resources["observed_resources_path"] = resources.get("observed_metrics_path")

    meta = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "created_at_utc": _now_utc_iso(),
        "source_commit": os.environ.get("SOURCE_COMMIT"),
        "source_manifest_paths": source_manifest_paths,
        "metric_columns": metric_columns,
        "warnings": warnings,
    }
    return master_metrics, resources, meta


def _write_manifest_json(
    *,
    path: Path,
    slot_df: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    payload = {
        "report_schema_version": meta["report_schema_version"],
        "manifest_version": meta["manifest_version"],
        "created_at_utc": meta["created_at_utc"],
        "source_commit": meta.get("source_commit"),
        "source_manifest_paths": meta.get("source_manifest_paths", []),
        "warning_count": len(meta.get("warnings", [])),
        "warnings": meta.get("warnings", []),
        "slot_count": int(len(slot_df)),
        "slots": _to_serializable_records(slot_df),
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_metrics_master(results_root: Path, master_metrics: pd.DataFrame) -> tuple[str, Path]:
    parquet_path = results_root / "metrics_master.parquet"
    json_path = results_root / "metrics_master.json"

    export_df = _artifact_export_df(master_metrics)
    records = _to_serializable_records(export_df)
    json_path.write_text(json.dumps({"rows": records}, indent=2))

    try:
        export_df.to_parquet(parquet_path, index=False)
        return "parquet", parquet_path
    except Exception:
        if parquet_path.exists():
            parquet_path.unlink()
        return "json", json_path


def _write_table_with_parquet_fallback(
    *,
    results_root: Path,
    base_name: str,
    df: pd.DataFrame,
) -> tuple[str, Path]:
    parquet_path = results_root / f"{base_name}.parquet"
    json_path = results_root / f"{base_name}.json"
    export_df = _artifact_export_df(df)
    payload = {
        "columns": list(export_df.columns),
        "rows": _to_serializable_records(export_df),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    try:
        export_df.to_parquet(parquet_path, index=False)
        return "parquet", parquet_path
    except Exception:
        if parquet_path.exists():
            parquet_path.unlink()
        return "json", json_path


def _write_resources_master(path: Path, resources: pd.DataFrame, meta: dict[str, Any]) -> None:
    export_df = _artifact_export_df(resources)
    payload = {
        "report_schema_version": meta["report_schema_version"],
        "created_at_utc": meta["created_at_utc"],
        "row_count": int(len(export_df)),
        "rows": _to_serializable_records(export_df),
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_compatibility_views(results_root: Path, master_metrics: pd.DataFrame, matrices: list[str], meta: dict[str, Any]) -> None:
    tables_dir = results_root / "tables"
    reports_dir = results_root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    drop_master_only = {
        "slot_id",
        "status_reason_code",
        "status_reason",
        "in_experiment_matrix",
        "observed_run",
        "duplicate_run_ids",
        "duplicate_observation_count",
        "observed_metrics_path",
        "observed_report_path",
        "producer",
        "source_matrix_manifest",
        "source_row_index",
        "ingested_at_utc",
        "observed_at_utc",
        "normalization_version",
    }

    for matrix in matrices:
        sub = master_metrics[master_metrics["matrix"] == matrix].copy()
        export_sub = _artifact_export_df(sub)
        csv_df = export_sub[[c for c in export_sub.columns if c not in drop_master_only]]
        csv_path = tables_dir / f"{matrix}_summary.csv"
        csv_df.to_csv(csv_path, index=False)

        by_status = {k: int(v) for k, v in export_sub["run_status"].value_counts(dropna=False).to_dict().items()}
        by_stage = {k: int(v) for k, v in export_sub["stage"].value_counts(dropna=False).to_dict().items()}
        report_payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "matrix": matrix,
            "generated_at": _now_utc_iso(),
            "source": "metrics_master_projection",
            "record_count": int(len(export_sub)),
            "by_status": by_status,
            "by_stage": by_stage,
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "records": _to_serializable_records(export_sub),
        }
        report_path = reports_dir / f"{matrix}_report.json"
        report_path.write_text(json.dumps(report_payload, indent=2))


def run_make_report(*, results_root: str | Path = "results", matrices: list[str] | None = None) -> dict[str, Any]:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    matrices = matrices or ["matrix_a", "matrix_b", "matrix_c"]

    master_metrics, resources, meta = _build_master_tables(root, matrices)

    # Manifest should reflect planned slots only.
    slot_cols = [
        "slot_id",
        "manifest_slot",
        "stage",
        "matrix",
        "family",
        "instance_id",
        "encoding",
        "decoder",
        "alpha_mode",
        "execution_mode",
        "trial_seed",
        "canonical_trial_seed",
        "seed",
    ]
    slot_df = master_metrics.copy()
    slot_df["manifest_slot"] = slot_df["slot_id"]
    slot_df = slot_df[slot_cols].drop_duplicates().reset_index(drop=True)

    run_manifest_path = root / "run_manifest.json"
    _write_manifest_json(path=run_manifest_path, slot_df=slot_df, meta=meta)

    metrics_mode, metrics_path = _write_metrics_master(root, master_metrics)
    resources_path = root / "resources_master.json"
    _write_resources_master(resources_path, resources, meta)
    _write_compatibility_views(root, master_metrics, matrices, meta)

    quality_cost_master, quality_cost_unavailable = _build_quality_cost_tables(
        master_metrics=master_metrics,
        results_root=root,
    )
    quality_cost_master_records = _to_serializable_records(_artifact_export_df(quality_cost_master))
    quality_cost_unavailable_records = _to_serializable_records(_artifact_export_df(quality_cost_unavailable))
    validate_quality_cost_master(quality_cost_master_records)
    validate_quality_cost_unavailable(quality_cost_unavailable_records)

    matrix_a_pairing_audit = _build_matrix_a_pairing_audit(master_metrics)
    quality_cost_admission_audit, quality_cost_duplicate_audit = _build_stage_e_audits(
        master_metrics,
        results_root=root,
    )

    qc_master_mode, qc_master_path = _write_table_with_parquet_fallback(
        results_root=root,
        base_name="quality_cost_master",
        df=quality_cost_master,
    )
    qc_unavail_mode, qc_unavail_path = _write_table_with_parquet_fallback(
        results_root=root,
        base_name="quality_cost_unavailable",
        df=quality_cost_unavailable,
    )
    pairing_audit_mode, pairing_audit_path = _write_table_with_parquet_fallback(
        results_root=root,
        base_name="matrix_a_pairing_audit",
        df=matrix_a_pairing_audit,
    )
    admission_audit_mode, admission_audit_path = _write_table_with_parquet_fallback(
        results_root=root,
        base_name="quality_cost_admission_audit",
        df=quality_cost_admission_audit,
    )
    duplicate_audit_mode, duplicate_audit_path = _write_table_with_parquet_fallback(
        results_root=root,
        base_name="quality_cost_duplicate_audit",
        df=quality_cost_duplicate_audit,
    )
    for required_json in [
        root / "matrix_a_pairing_audit.json",
        root / "quality_cost_admission_audit.json",
        root / "quality_cost_duplicate_audit.json",
    ]:
        if not required_json.exists():
            raise ValueError(f"required audit artifact missing: {required_json}")

    return {
        "run_manifest": str(run_manifest_path),
        "metrics_master": str(metrics_path),
        "metrics_master_mode": metrics_mode,
        "resources_master": str(resources_path),
        "quality_cost_master": str(qc_master_path),
        "quality_cost_master_mode": qc_master_mode,
        "quality_cost_master_rows": int(len(quality_cost_master)),
        "quality_cost_unavailable": str(qc_unavail_path),
        "quality_cost_unavailable_mode": qc_unavail_mode,
        "quality_cost_unavailable_rows": int(len(quality_cost_unavailable)),
        "matrix_a_pairing_audit": str(pairing_audit_path),
        "matrix_a_pairing_audit_mode": pairing_audit_mode,
        "quality_cost_admission_audit": str(admission_audit_path),
        "quality_cost_admission_audit_mode": admission_audit_mode,
        "quality_cost_duplicate_audit": str(duplicate_audit_path),
        "quality_cost_duplicate_audit_mode": duplicate_audit_mode,
        "warnings": meta.get("warnings", []),
        "rows_metrics": int(len(master_metrics)),
        "rows_resources": int(len(resources)),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage B reporting authority outputs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--matrices", default="matrix_a,matrix_b,matrix_c")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    matrices = [m.strip() for m in str(args.matrices).split(",") if m.strip()]
    out = run_make_report(results_root=args.results_root, matrices=matrices)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
