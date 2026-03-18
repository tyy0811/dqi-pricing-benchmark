"""Paired comparison runner for WHT-truncated vs ILP-exact-reference encodings."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable

import numpy as np

from src.ilp_to_xorsat_exact import build_ilp_exact_reference_artifact
from src.metrics_decision import compute_decision_metrics
from src.pricing_ilp import build_toy_pricing_ilp_model
from src.reduction import build_surrogate_artifact, surrogate_objective

PRIMARY_EXACT_REFERENCE_BACKEND = "ilp_derived_exact"
EXPLORATORY_WHT_BACKEND = "wht_truncated"
DECISIVE_METRICS = (
    "top1_regret",
    "topk_regret",
    "optimum_recall_at_k",
    "decision_distortion",
)
MAXIMIZE_METRICS = {"optimum_recall_at_k"}


def _normalize_for_hash(value):
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def _canonical_hash(payload: dict) -> str:
    normalized = _normalize_for_hash(payload)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def artifacts_compatible_for_paired_run(
    artifact_a_meta: dict,
    artifact_b_meta: dict,
    *,
    alpha_mode: str,
    execution_mode: str,
    decoder_mode: str,
) -> tuple[bool, list[str]]:
    """Validate compatibility contract for paired backend comparison."""
    notes: list[str] = []
    for key in (
        "instance_hash",
        "codeword_mapping",
        "bit_order",
        "bitstring_convention",
        "objective_semantics",
    ):
        if artifact_a_meta.get(key) != artifact_b_meta.get(key):
            notes.append(f"mismatch:{key}")

    if not alpha_mode:
        notes.append("missing:alpha_mode")
    if not execution_mode:
        notes.append("missing:execution_mode")
    if not decoder_mode:
        notes.append("missing:decoder_mode")

    return (len(notes) == 0), notes


def _artifact_structure_summary(B: np.ndarray) -> dict:
    row_weights = np.sum(B, axis=1)
    return {
        "row_weight_mean": float(np.mean(row_weights)),
        "row_weight_max": int(np.max(row_weights)),
        "density": float(np.mean(B)),
    }


def _wht_artifact_hash(artifact: dict) -> str:
    return _canonical_hash(
        {
            "artifact_version": artifact["artifact_version"],
            "mode": artifact["mode"],
            "n": int(artifact["n"]),
            "m": int(artifact["m"]),
            "indices": np.asarray(artifact["indices"], dtype=np.int64).tolist(),
            "coefficients": np.asarray(artifact["coefficients"], dtype=np.float64).tolist(),
        }
    )


def _select_candidates_by_g(g_values: np.ndarray, n_samples: int) -> np.ndarray:
    n = len(g_values)
    take = int(min(max(1, n_samples), n))
    order = np.lexsort((np.arange(n, dtype=np.int64), -g_values))
    return order[:take]


def _evaluate_g_over_full_domain(B: np.ndarray, v: np.ndarray, w: np.ndarray, n_bits: int) -> np.ndarray:
    n_states = 2 ** n_bits
    out = np.zeros(n_states, dtype=np.float64)
    for x in range(n_states):
        out[x] = surrogate_objective(x, B, v, w, n_bits)
    return out


def _required_metric_delta_names() -> list[str]:
    return [
        "best_sampled_F",
        "best_sampled_G",
        "top1_regret",
        "topk_regret",
        "optimum_recall_at_k",
        "precision_at_k_high_F",
        "best_top_G_sampled_F",
        "surrogate_best_vs_true_best_sampled_gap",
        "spearman_rho_F_G",
        "retained_energy_eta",
        "decision_distortion",
        "decode_success",
        "postselection_success",
    ]


def resolve_encoding_reviewer_status(result: dict) -> dict:
    """Return a normalized reviewer-facing encoding verdict/status summary."""

    def _to_str(value) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    def _to_str_list(values) -> list[str]:
        if not isinstance(values, (list, tuple)):
            return []
        return [str(v) for v in values]

    def _first_non_empty(source: dict, *keys: str) -> str | None:
        for key in keys:
            value = _to_str(source.get(key))
            if value is not None:
                return value
        return None

    def _normalize_status(raw_status) -> str | None:
        status = _to_str(raw_status)
        if status is None:
            return None
        lowered = status.lower()
        if lowered in {"ok", "completed", "available", "yes", "true"}:
            return "available"
        if lowered in {"not_applicable", "not-applicable", "na", "unsupported", "skipped"}:
            return "not_applicable"
        if lowered in {
            "unavailable",
            "not_available",
            "disabled",
            "missing",
            "not_available_for_instance",
            "failed",
            "error",
        }:
            return "unavailable"
        return lowered

    output = {
        "default_backend": None,
        "formal_verdict": None,
        "status": "unavailable",
        "reason": "encoding_reviewer_status_unavailable",
        "metrics_used": [],
    }

    if not isinstance(result, dict):
        return output

    recommendation = result.get("encoding_backend_recommendation")
    if isinstance(recommendation, dict):
        status = _normalize_status(recommendation.get("status"))
        return {
            "default_backend": _to_str(recommendation.get("default_reviewer_backend")),
            "formal_verdict": recommendation.get("formal_verdict"),
            "status": status if status is not None else output["status"],
            "reason": _first_non_empty(
                recommendation,
                "reason",
                "status_reason",
                "status_reason_code",
            )
            or output["reason"],
            "metrics_used": _to_str_list(recommendation.get("metrics_used")),
        }

    summary = result.get("summary")
    if isinstance(summary, dict):
        formal_verdict = summary.get("winner")
        reviewer_verdict = _to_str(summary.get("reviewer_verdict"))
        if reviewer_verdict is not None and formal_verdict is not None:
            return {
                "default_backend": reviewer_verdict,
                "formal_verdict": formal_verdict,
                "status": "available",
                "reason": "paired_comparison_complete",
                "metrics_used": _to_str_list(summary.get("metrics_used")),
            }
        return {
            "default_backend": reviewer_verdict,
            "formal_verdict": formal_verdict,
            "status": "unavailable",
            "reason": "paired_comparison_partial",
            "metrics_used": _to_str_list(summary.get("metrics_used")),
        }

    reviewer_verdict = _to_str(result.get("reviewer_verdict"))
    if reviewer_verdict is not None:
        formal_verdict = result.get("formal_verdict")
        if formal_verdict is not None:
            return {
                "default_backend": reviewer_verdict,
                "formal_verdict": formal_verdict,
                "status": "available",
                "reason": "paired_comparison_complete",
                "metrics_used": _to_str_list(result.get("metrics_used")),
            }
        return {
            "default_backend": reviewer_verdict,
            "formal_verdict": None,
            "status": "unavailable",
            "reason": "paired_comparison_partial",
            "metrics_used": _to_str_list(result.get("metrics_used")),
        }

    paired_keys = {"backend_a_name", "backend_a_reviewer_name", "backend_b_reviewer_name", "backend_b_name"}
    if any(key in result for key in paired_keys) or "backend_a_metrics" in result:
        return {
            "default_backend": None,
            "formal_verdict": None,
            "status": "unavailable",
            "reason": "paired_comparison_partial",
            "metrics_used": [],
        }

    status_hint = _normalize_status(result.get("status"))
    if status_hint in {"available", "unavailable", "not_applicable"}:
        return {
            "default_backend": _to_str(result.get("default_backend")),
            "formal_verdict": result.get("formal_verdict"),
            "status": status_hint,
            "reason": _first_non_empty(result, "reason", "status_reason", "status_reason_code") or output["reason"],
            "metrics_used": _to_str_list(result.get("metrics_used")),
        }

    return output


def paired_encoding_verdict(delta_metrics: dict) -> str:
    """Return conservative paired-comparison verdict from WHT-minus-exact deltas."""
    exact_non_worse = True
    exact_strict = False
    wht_non_worse = True
    wht_strict = False

    for metric in DECISIVE_METRICS:
        value = delta_metrics.get(f"delta_{metric}_a_minus_b")
        if value is None:
            value = delta_metrics.get(metric)
        if value is None:
            return "tie"
        delta = float(value)
        if delta < 0:
            exact_strict = True
            wht_non_worse = False
        elif delta > 0:
            wht_strict = True
            exact_non_worse = False

    if exact_non_worse and exact_strict:
        return "mainline_exact_reference"
    if wht_non_worse and wht_strict:
        return "exploratory_only"
    return "tie"


def _metric_lookup(metrics: dict, name: str):
    for section in ("core_decision", "faithfulness", "pipeline", "timing", "metadata"):
        block = metrics.get(section, {})
        if name in block:
            return block[name]
    return None


def flatten_decision_metrics(metrics: dict | None) -> dict:
    """Flatten nested decision-metric payloads into scalar notebook-friendly keys."""

    def _finite_or_none(metric_name: str):
        value = _metric_lookup(metrics or {}, metric_name)
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric

    return {
        "spearman_rho_F_G": _finite_or_none("spearman_rho_F_G"),
        "retained_energy_eta": _finite_or_none("retained_energy_eta"),
        "decision_distortion": _finite_or_none("decision_distortion"),
        "top1_regret": _finite_or_none("top1_regret"),
        "topk_regret": _finite_or_none("topk_regret"),
    }


def flatten_paired_metrics_for_encoding(report: dict, encoding_name: str) -> dict:
    """Flatten one backend's nested paired-comparison metrics for notebook/table use."""

    backend_specs = (
        ("backend_a_metrics", "backend_a_name", "backend_a_reviewer_name", "wht_truncated"),
        ("backend_b_metrics", "backend_b_name", "backend_b_reviewer_name", "ilp_derived"),
    )

    selected_metrics = None
    selected_label = None
    source_backend = None
    target = str(encoding_name)
    for metrics_key, backend_name_key, reviewer_name_key, fallback_name in backend_specs:
        backend_names = {
            str(report.get(backend_name_key, "")).strip(),
            str(report.get(reviewer_name_key, "")).strip(),
        }
        if any(report.get(key) is not None for key in (metrics_key, backend_name_key, reviewer_name_key)):
            backend_names.add(fallback_name)
        backend_names.discard("")
        if target in backend_names:
            candidate = report.get(metrics_key)
            if isinstance(candidate, dict):
                selected_metrics = candidate
                selected_label = report.get(backend_name_key) or report.get(reviewer_name_key) or fallback_name
                source_backend = metrics_key
                break
            return {
                "encoding_name": target,
                "source_backend": metrics_key,
                "source_label": report.get(backend_name_key) or report.get(reviewer_name_key) or fallback_name,
                "availability": "unavailable",
                "reason": "encoding_backend_metrics_missing_from_paired_report",
                "spearman_rho_F_G": None,
                "retained_energy_eta": None,
                "decision_distortion": None,
                "top1_regret": None,
                "topk_regret": None,
            }

    if selected_metrics is None:
        return {
            "encoding_name": target,
            "source_backend": None,
            "source_label": None,
            "availability": "unavailable",
            "reason": "encoding_backend_missing_from_paired_report",
            "spearman_rho_F_G": None,
            "retained_energy_eta": None,
            "decision_distortion": None,
            "top1_regret": None,
            "topk_regret": None,
        }

    metrics = {
        "encoding_name": target,
        "source_backend": source_backend,
        "source_label": selected_label,
        "availability": "available",
        "reason": None,
        **flatten_decision_metrics(selected_metrics),
    }
    required = (
        "spearman_rho_F_G",
        "retained_energy_eta",
        "decision_distortion",
        "top1_regret",
        "topk_regret",
    )
    missing = [name for name in required if metrics[name] is None]
    if missing:
        metrics["availability"] = "partial"
        metrics["reason"] = "missing_metrics:" + ",".join(missing)
    return metrics


def run_paired_encoding_comparison(
    *,
    prob,
    k_wht: int,
    alpha_mode: str,
    execution_mode: str,
    decoder_mode: str,
    n_samples: int = 100,
    seed: int | None = None,
    top_k_eval: int = 10,
) -> dict:
    """Run paired comparison on one instance with fixed non-encoding knobs."""
    del seed  # deterministic top-G selection for Stage 1 reference comparison
    ilp_model = build_toy_pricing_ilp_model(prob)
    exact_artifact = build_ilp_exact_reference_artifact(ilp_model)
    wht_artifact = build_surrogate_artifact(prob, k=k_wht)

    n_bits = int(prob.n_bits)
    f_table = np.asarray(prob.build_truth_table(), dtype=np.float64)
    f_opt = float(np.max(f_table))
    optimum_x = np.flatnonzero(np.isclose(f_table, f_opt)).astype(np.int64)

    B_wht = np.asarray(wht_artifact["B"], dtype=np.int8)
    v_wht = np.asarray(wht_artifact["v"], dtype=np.int8)
    w_wht = np.asarray(wht_artifact["term_weights"], dtype=np.float64)
    g_wht_full = _evaluate_g_over_full_domain(B_wht, v_wht, w_wht, n_bits)
    idx_wht = _select_candidates_by_g(g_wht_full, n_samples=n_samples)

    B_exact = np.asarray(exact_artifact["B"], dtype=np.int8)
    v_exact = np.asarray(exact_artifact["v"], dtype=np.int8)
    w_exact = np.asarray(exact_artifact["term_weights"], dtype=np.float64)
    g_exact_full = _evaluate_g_over_full_domain(B_exact, v_exact, w_exact, n_bits)
    idx_exact = _select_candidates_by_g(g_exact_full, n_samples=n_samples)

    metrics_wht = compute_decision_metrics(
        F_values=f_table[idx_wht],
        G_values=g_wht_full[idx_wht],
        f_opt=f_opt,
        k_eval=top_k_eval,
        sampled_x=idx_wht.tolist(),
        optimum_x=optimum_x.tolist(),
        decode_success=None,
        postselection_success=None,
        retained_energy_eta=float(wht_artifact["diagnostics"]["energy_fraction"]),
        candidate_source_label="wht_truncated",
    )
    metrics_exact = compute_decision_metrics(
        F_values=f_table[idx_exact],
        G_values=g_exact_full[idx_exact],
        f_opt=f_opt,
        k_eval=top_k_eval,
        sampled_x=idx_exact.tolist(),
        optimum_x=optimum_x.tolist(),
        decode_success=None,
        postselection_success=None,
        retained_energy_eta=1.0,
        candidate_source_label=PRIMARY_EXACT_REFERENCE_BACKEND,
    )

    shared_meta = {
        "instance_hash": ilp_model["instance_hash"],
        "codeword_mapping": ilp_model["codeword_mapping"],
        "bit_order": ilp_model["bit_order"],
        "bitstring_convention": ilp_model["bitstring_convention"],
        "objective_semantics": ilp_model["objective_semantics"],
    }
    compatible, compatibility_notes = artifacts_compatible_for_paired_run(
        shared_meta,
        shared_meta,
        alpha_mode=alpha_mode,
        execution_mode=execution_mode,
        decoder_mode=decoder_mode,
    )

    wht_summary = {
        "backend_name": EXPLORATORY_WHT_BACKEND,
        "encoding_type": EXPLORATORY_WHT_BACKEND,
        "reviewer_backend_name": EXPLORATORY_WHT_BACKEND,
        "n_bits": int(wht_artifact["n"]),
        "m_terms": int(wht_artifact["m"]),
        "retained_energy_eta": float(wht_artifact["diagnostics"]["energy_fraction"]),
        "exact_full_spectrum": False,
        "constant_offset": float(np.asarray(wht_artifact["f_hat"], dtype=np.float64)[0]),
        "artifact_hash": _wht_artifact_hash(wht_artifact),
        "story_role": wht_artifact.get("story_role", "exploratory"),
        "benchmark_status": wht_artifact.get("benchmark_status", "non_mainline_surrogate"),
        **_artifact_structure_summary(B_wht),
    }
    exact_summary = {
        "backend_name": "ilp_exact_reference",
        "encoding_type": exact_artifact["encoding_type"],
        "reviewer_backend_name": PRIMARY_EXACT_REFERENCE_BACKEND,
        "n_bits": int(exact_artifact["n_bits"]),
        "m_terms": int(exact_artifact["m_terms"]),
        "retained_energy_eta": 1.0,
        "exact_full_spectrum": bool(exact_artifact["exact_full_spectrum"]),
        "constant_offset": float(exact_artifact["constant_offset"]),
        "artifact_hash": exact_artifact["artifact_hash"],
        **_artifact_structure_summary(B_exact),
    }

    metric_deltas = {}
    for name in _required_metric_delta_names():
        a_val = _metric_lookup(metrics_wht, name)
        b_val = _metric_lookup(metrics_exact, name)
        delta_key = f"delta_{name}_a_minus_b"
        if a_val is None or b_val is None:
            metric_deltas[delta_key] = None
        else:
            metric_deltas[delta_key] = float(a_val) - float(b_val)

    formal_verdict = paired_encoding_verdict(metric_deltas)
    reviewer_verdict = (
        EXPLORATORY_WHT_BACKEND
        if formal_verdict == "exploratory_only"
        else PRIMARY_EXACT_REFERENCE_BACKEND
    )

    comparison_payload = {
        "schema_version": "1.0",
        "instance_hash": ilp_model["instance_hash"],
        "backend_a_hash": wht_summary["artifact_hash"],
        "backend_b_hash": exact_summary["artifact_hash"],
        "alpha_mode": alpha_mode,
        "execution_mode": execution_mode,
        "decoder_mode": decoder_mode,
        "n_samples": int(n_samples),
        "top_k_eval": int(top_k_eval),
    }
    comparison_id = _canonical_hash(comparison_payload)

    return {
        "comparison_id": comparison_id,
        "comparison_metadata": comparison_payload,
        "instance_metadata": {
            "instance_id": ilp_model["instance_id"],
            "instance_hash": ilp_model["instance_hash"],
            "n_features": ilp_model["n_features"],
            "n_bits": ilp_model["n_bits"],
        },
        "backend_a_name": "wht_truncated",
        "backend_b_name": "ilp_exact_reference",
        "backend_a_reviewer_name": EXPLORATORY_WHT_BACKEND,
        "backend_b_reviewer_name": PRIMARY_EXACT_REFERENCE_BACKEND,
        "backend_a_artifact_summary": wht_summary,
        "backend_b_artifact_summary": exact_summary,
        "artifacts_compatible_for_paired_run": bool(compatible),
        "compatibility_notes": compatibility_notes,
        "backend_a_metrics": metrics_wht,
        "backend_b_metrics": metrics_exact,
        "metric_deltas": metric_deltas,
        "summary": {
            "winner": formal_verdict,
            "metrics_used": list(DECISIVE_METRICS),
            "reviewer_verdict": reviewer_verdict,
        },
    }
