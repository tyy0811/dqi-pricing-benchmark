"""Matrix runners with standardized artifact outputs for P/C/S benchmark families.

This module provides command-line entrypoints for:
- Matrix A: pricing-family runs (P3/P4/P5) via ``run_benchmark``.
- Matrix B: coherent tiny-family runs (C1/C2/C3) via canonical coherent artifacts.
- Matrix C: synthetic structure family runs (Family S) via ``run_structure_sweep_benchmark``.

All matrix runs write:
- per-run JSON records under ``results/matrix_*/runs/``
- aggregated CSV under ``results/tables/``
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.benchmark import (
    load_scaling_instance,
    run_benchmark,
    run_structure_sweep_benchmark,
)
from src.boundary import build_boundary_record, capture_execution_metrics, classify_boundary_exception
from src.coherent_reference import load_coherent_family
from src.dqi_pipeline import DQIPipeline
from src.dqi_pipeline import compare_candidate_distributions_tvd
from src.dqi_state import decoder_diagnostics, run_dqi_statevector
from src.encoding_compare import run_paired_encoding_comparison
from src.metrics_decision import compute_decision_metrics
from src.resources import dqi_resource_estimate
from src.structure_metrics import compute_structure_metrics
from src.weights import (
    heuristic_alpha_weights,
    max_safe_ell,
    paper_alpha_weights,
    uniform_weights,
    validate_dqi_parameters,
)


SCHEMA_VERSION = "1.0"

MATRIX_NAME_TO_DIR = {
    "matrix_a": "matrix_a",
    "matrix_b": "matrix_b",
    "matrix_c": "matrix_c",
}

REQUIRED_RECORD_FIELDS = [
    "instance_id",
    "family",
    "encoding",
    "decoder",
    "alpha_mode",
    "execution_mode",
    "seed",
    "best_sampled_F",
    "best_sampled_G",
    "top1_regret",
    "topk_regret",
    "optimum_recall_at_k",
    "best_top_G_sampled_F",
    "spearman_rho_F_G",
    "retained_energy_eta",
    "decision_distortion",
    "decode_success",
    "postselection_success",
    "runtime_sec",
]

RESOURCE_FIELDS = [
    "resource_decode_uncompute_model",
    "resource_decode_uncompute_status",
    "resource_decode_uncompute_confidence",
    "resource_total_gate_est_with_decode",
    "resource_total_depth_est_with_decode",
    "resource_total_qubits_est_with_decode",
    "resource_prefix_gate_est",
    "resource_prefix_depth_est",
    "resource_decode_model_gate_est",
    "resource_decode_model_depth_est",
    "resource_weight_register_qubits",
    "resource_coherent_weight_register_gate_est",
    "resource_coherent_weight_register_depth_est",
    "qiskit_subset_resource_report",
    "qiskit_transpiled_structural_count",
    "qiskit_subset_transpiled_depth",
    "qiskit_subset_cx_count",
    "qiskit_subset_qubit_count",
    "qiskit_subset_status",
    "resource_qiskit_subset_resource_report",
    "resource_qiskit_transpiled_structural_count",
    "resource_qiskit_subset_transpiled_depth",
    "resource_qiskit_subset_cx_count",
    "resource_qiskit_subset_qubit_count",
    "resource_qiskit_subset_status",
]

METRIC_FIELDS = [
    "best_sampled_F",
    "best_sampled_G",
    "top1_regret",
    "topk_regret",
    "optimum_recall_at_k",
    "best_top_G_sampled_F",
    "spearman_rho_F_G",
    "retained_energy_eta",
    "decision_distortion",
    "decode_success",
    "postselection_success",
    "runtime_sec",
]

STRUCTURE_FIELDS = [
    "structure_row_weight_mean",
    "structure_row_weight_max",
    "structure_col_weight_mean",
    "structure_col_weight_max",
    "structure_rank",
    "structure_rank_deficiency",
    "structure_short_cycle_proxy",
    "structure_decodable_syndrome_fraction",
    "structure_ambiguous_syndrome_count",
    "structure_collision_rate",
    "structure_distance_proxy",
]

STATUS_ALIASES = {
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


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_slug(raw: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in raw)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _canonical_run_status(status: Any, *, default: str = "unavailable") -> str:
    if status is None:
        return default
    raw = str(status).strip().lower()
    if not raw:
        return default
    return STATUS_ALIASES.get(raw, raw)


def _decode_model_for_decoder(decoder_mode: str) -> str:
    if decoder_mode == "bruteforce":
        return "lookup_table_reversible"
    if decoder_mode == "bp1":
        return "bp1_classical_oracle_estimate"
    raise ValueError(f"Unsupported decoder_mode={decoder_mode!r}")


def _alpha_vector_for_mode(
    *,
    alpha_mode: str,
    B: np.ndarray,
    v: np.ndarray,
    ell: int,
) -> np.ndarray:
    if alpha_mode == "uniform":
        return uniform_weights(ell)
    if alpha_mode == "paper":
        return paper_alpha_weights(ell)
    if alpha_mode == "heuristic":
        return heuristic_alpha_weights(B, v, ell)
    raise ValueError(f"Unsupported alpha_mode={alpha_mode!r}")


def _matrix_b_decoder_modes(decoder_label: str) -> tuple[str, str]:
    label = str(decoder_label).strip().lower()
    if label in {"oracle", "bruteforce"}:
        return "bruteforce", "oracle"
    if label == "bp1":
        return "bp1", "bp1"
    raise ValueError(
        f"Unsupported matrix-b decoder={decoder_label!r}. Use one of ['oracle', 'bp1']."
    )


def _normalize_matrix_c_decoder_label(decoder_label: str) -> str:
    raw = str(decoder_label).strip().lower()
    if raw in {"bruteforce", "brute-force", "brute_force"}:
        return "bruteforce"
    if raw in {"oracle"}:
        return "oracle"
    if raw in {"bp1"}:
        return "bp1"
    if raw in {"bp+osd-lite", "bp_osd_lite", "bp-osd-lite", "bposdlite"}:
        return "bp+osd-lite"
    raise ValueError(
        "Unsupported matrix-c decoder={!r}. Use one of "
        "['bruteforce', 'bp1', 'bp+osd-lite', 'oracle'].".format(decoder_label)
    )


def _matrix_c_internal_decoder(decoder_label: str) -> str | None:
    norm = _normalize_matrix_c_decoder_label(decoder_label)
    if norm == "bruteforce":
        return "bruteforce"
    if norm == "oracle":
        return "bruteforce"
    if norm == "bp1":
        return "bp1"
    return None


def _structure_field_defaults() -> dict[str, Any]:
    return {field: None for field in STRUCTURE_FIELDS}


def _structure_fields(structure_summary: dict[str, Any] | None) -> dict[str, Any]:
    out = _structure_field_defaults()
    if not isinstance(structure_summary, dict):
        return out
    out.update(
        {
            "structure_row_weight_mean": _to_float(structure_summary.get("row_weight_mean")),
            "structure_row_weight_max": _to_float(structure_summary.get("row_weight_max")),
            "structure_col_weight_mean": _to_float(structure_summary.get("col_weight_mean")),
            "structure_col_weight_max": _to_float(structure_summary.get("col_weight_max")),
            "structure_rank": _to_float(
                structure_summary.get("rank", structure_summary.get("rank_B_gf2"))
            ),
            "structure_rank_deficiency": _to_float(structure_summary.get("rank_deficiency")),
            "structure_short_cycle_proxy": _to_float(structure_summary.get("short_cycle_proxy")),
            "structure_decodable_syndrome_fraction": _to_float(
                structure_summary.get("decodable_syndrome_fraction")
            ),
            "structure_ambiguous_syndrome_count": _to_float(
                structure_summary.get("ambiguous_syndrome_count")
            ),
            "structure_collision_rate": _to_float(
                structure_summary.get("decoder_collision_rate")
            ),
            "structure_distance_proxy": _to_float(
                structure_summary.get(
                    "distance_proxy",
                    structure_summary.get("approximate_distance_proxy"),
                )
            ),
        }
    )
    return out


def ensure_output_layout(output_root: str | Path = "results") -> dict[str, Path]:
    root = Path(output_root)
    layout = {
        "root": root,
        "matrix_a": root / "matrix_a",
        "matrix_b": root / "matrix_b",
        "matrix_c": root / "matrix_c",
        "tables": root / "tables",
        "plots": root / "plots",
        "reports": root / "reports",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    for matrix_key in ("matrix_a", "matrix_b", "matrix_c"):
        (layout[matrix_key] / "runs").mkdir(parents=True, exist_ok=True)
    return layout


def _resource_field_defaults() -> dict[str, Any]:
    return {field: None for field in RESOURCE_FIELDS}


def _resource_fields(resource_summary: dict[str, Any] | None) -> dict[str, Any]:
    out = _resource_field_defaults()
    if not isinstance(resource_summary, dict):
        return out

    qiskit_subset_resource_report = resource_summary.get("qiskit_subset_resource_report")
    if not isinstance(qiskit_subset_resource_report, dict):
        qiskit_subset_resource_report = None

    qiskit_transpiled_structural_count = resource_summary.get("qiskit_transpiled_structural_count")
    if not isinstance(qiskit_transpiled_structural_count, dict):
        qiskit_transpiled_structural_count = None

    qiskit_subset_transpiled_depth = (
        qiskit_subset_resource_report.get("transpiled_depth") if isinstance(qiskit_subset_resource_report, dict) else None
    )
    qiskit_subset_cx_count = (
        qiskit_subset_resource_report.get("cx_count") if isinstance(qiskit_subset_resource_report, dict) else None
    )
    qiskit_subset_qubit_count = (
        qiskit_subset_resource_report.get("qubit_count") if isinstance(qiskit_subset_resource_report, dict) else None
    )
    qiskit_subset_status = (
        qiskit_subset_resource_report.get("status")
        if isinstance(qiskit_subset_resource_report, dict)
        else resource_summary.get("qiskit_transpiled_structural_count", {}).get("status")
        if isinstance(resource_summary.get("qiskit_transpiled_structural_count"), dict)
        else None
    )

    out.update(
        {
            "resource_decode_uncompute_model": resource_summary.get("decode_uncompute_model"),
            "resource_decode_uncompute_status": resource_summary.get("decode_uncompute_status"),
            "resource_decode_uncompute_confidence": resource_summary.get("decode_uncompute_confidence"),
            "resource_total_gate_est_with_decode": _to_float(resource_summary.get("total_gate_est_with_decode")),
            "resource_total_depth_est_with_decode": _to_float(resource_summary.get("total_depth_est_with_decode")),
            "resource_total_qubits_est_with_decode": _to_float(resource_summary.get("total_qubits_est_with_decode")),
            "resource_prefix_gate_est": _to_float(resource_summary.get("prefix_gate_est")),
            "resource_prefix_depth_est": _to_float(resource_summary.get("prefix_depth_est")),
            "resource_decode_model_gate_est": _to_float(resource_summary.get("decode_model_gate_est")),
            "resource_decode_model_depth_est": _to_float(resource_summary.get("decode_model_depth_est")),
            "resource_weight_register_qubits": _to_float(resource_summary.get("weight_register_qubits")),
            "resource_coherent_weight_register_gate_est": _to_float(resource_summary.get("coherent_weight_register_gate_est")),
            "resource_coherent_weight_register_depth_est": _to_float(resource_summary.get("coherent_weight_register_depth_est")),
            "resource_qiskit_subset_resource_report": qiskit_subset_resource_report,
            "resource_qiskit_transpiled_structural_count": qiskit_transpiled_structural_count,
            "resource_qiskit_subset_transpiled_depth": _to_float(qiskit_subset_transpiled_depth),
            "resource_qiskit_subset_cx_count": _to_float(qiskit_subset_cx_count),
            "resource_qiskit_subset_qubit_count": _to_float(qiskit_subset_qubit_count),
            "resource_qiskit_subset_status": qiskit_subset_status,
            "qiskit_subset_resource_report": qiskit_subset_resource_report,
            "qiskit_transpiled_structural_count": qiskit_transpiled_structural_count,
            "qiskit_subset_transpiled_depth": qiskit_subset_transpiled_depth,
            "qiskit_subset_cx_count": qiskit_subset_cx_count,
            "qiskit_subset_qubit_count": qiskit_subset_qubit_count,
            "qiskit_subset_status": qiskit_subset_status,
        }
    )
    return out


def _metric_fields(decision_metrics: dict[str, Any] | None) -> dict[str, Any]:
    core = {}
    faith = {}
    pipeline = {}
    timing = {}
    if isinstance(decision_metrics, dict):
        core = decision_metrics.get("core_decision", {}) or {}
        faith = decision_metrics.get("faithfulness", {}) or {}
        pipeline = decision_metrics.get("pipeline", {}) or {}
        timing = decision_metrics.get("timing", {}) or {}
    return {
        "top1_regret": _to_float(core.get("top1_regret")),
        "topk_regret": _to_float(core.get("topk_regret")),
        "optimum_recall_at_k": _to_float(core.get("optimum_recall_at_k")),
        "best_top_G_sampled_F": _to_float(core.get("best_top_G_sampled_F")),
        "spearman_rho_F_G": _to_float(faith.get("spearman_rho_F_G")),
        "retained_energy_eta": _to_float(faith.get("retained_energy_eta")),
        "decision_distortion": _to_float(core.get("decision_distortion")),
        "decode_success": _to_float(pipeline.get("decode_success")),
        "postselection_success": _to_float(pipeline.get("postselection_success")),
        "runtime_sec": _to_float(timing.get("runtime_s")),
    }


def _base_record(
    *,
    matrix: str,
    run_id: str,
    run_status: str,
    run_error: str | None,
    instance_id: str,
    family: str,
    encoding: str,
    decoder: str,
    alpha_mode: str,
    execution_mode: str,
    seed: int | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "matrix": matrix,
        "run_id": run_id,
        "run_timestamp_utc": _now_iso_utc(),
        "run_status": _canonical_run_status(run_status),
        "run_error": run_error,
        "instance_id": instance_id,
        "family": family,
        "encoding": encoding,
        "decoder": decoder,
        "alpha_mode": alpha_mode,
        "execution_mode": execution_mode,
        "seed": seed,
        "best_sampled_F": None,
        "best_sampled_G": None,
        "top1_regret": None,
        "topk_regret": None,
        "optimum_recall_at_k": None,
        "best_top_G_sampled_F": None,
        "spearman_rho_F_G": None,
        "retained_energy_eta": None,
        "energy_eta_applicable": True,  # False for non-WHT encodings where energy_eta is N/A
        "decision_distortion": None,
        "decode_success": None,
        "postselection_success": None,
        "runtime_sec": None,
        "attempted_execution": None,
        "boundary_reason": None,
        "error_type": None,
        "error_message": None,
        "estimated_qubits": None,
        "estimated_memory_gb": None,
        "wall_clock_s": None,
        "peak_memory_mb": None,
    }
    record.update(_resource_field_defaults())
    return record


def _validate_record_shape(record: dict[str, Any]) -> None:
    for key in REQUIRED_RECORD_FIELDS:
        if key not in record:
            raise KeyError(f"Missing required record key: {key}")


def _pricing_estimated_qubits(*, n_bits: int, m_terms: int, ell: int) -> int:
    return int(n_bits + m_terms + max(int(ell), 0) + 1)


def _estimated_statevector_memory_gb(total_qubits: int) -> float:
    return float((2 ** int(total_qubits)) * 16 / 1e9)


def summarize_status_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        inst = str(row.get("instance_id"))
        status = _canonical_run_status(row.get("run_status"), default="unavailable")
        inst_bucket = out.setdefault(inst, {"completed": 0, "unavailable": 0, "failed": 0})
        if status not in inst_bucket:
            inst_bucket[status] = 0
        inst_bucket[status] += 1
    return out


def _attach_metric_availability(record: dict[str, Any]) -> None:
    record["metric_availability"] = {
        key: ("available" if record.get(key) is not None else "unavailable")
        for key in METRIC_FIELDS
    }


def _write_per_run_json(matrix_dir: Path, record: dict[str, Any]) -> Path:
    run_path = matrix_dir / "runs" / f"{record['run_id']}.json"
    with open(run_path, "w") as f:
        json.dump(record, f, indent=2)
    return run_path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_aggregated_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    preferred = [
        "schema_version",
        "matrix",
        "run_id",
        "run_timestamp_utc",
        "run_status",
        "run_error",
        *REQUIRED_RECORD_FIELDS,
        *RESOURCE_FIELDS,
    ]
    all_keys = set()
    for rec in records:
        all_keys.update(rec.keys())
    ordered = preferred + sorted(k for k in all_keys if k not in set(preferred))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: _csv_value(rec.get(k)) for k in ordered})


def _write_matrix_manifest(
    *,
    matrix_name: str,
    matrix_dir: Path,
    csv_path: Path,
    run_files: list[Path],
    config: dict[str, Any],
) -> Path:
    manifest_path = matrix_dir / "manifest.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "matrix": matrix_name,
        "generated_at": _now_iso_utc(),
        "config": config,
        "aggregated_csv": str(csv_path),
        "run_count": len(run_files),
        "run_files": [str(path) for path in run_files],
    }
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


def _persist_matrix_artifacts(
    *,
    matrix_name: str,
    records: list[dict[str, Any]],
    layout: dict[str, Path],
    config: dict[str, Any],
    csv_filename: str | None = None,
) -> dict[str, Any]:
    matrix_dir = layout[matrix_name]
    run_files = [_write_per_run_json(matrix_dir, rec) for rec in records]
    csv_name = csv_filename or f"{matrix_name}_runs.csv"
    csv_path = layout["tables"] / csv_name
    _write_aggregated_csv(records, csv_path)
    manifest_path = _write_matrix_manifest(
        matrix_name=matrix_name,
        matrix_dir=matrix_dir,
        csv_path=csv_path,
        run_files=run_files,
        config=config,
    )
    return {
        "matrix_dir": matrix_dir,
        "runs_dir": matrix_dir / "runs",
        "aggregated_csv": csv_path,
        "manifest": manifest_path,
        "run_count": len(records),
    }


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def run_matrix_a(
    *,
    output_root: str | Path = "results",
    features: Iterable[int] = (3, 4, 5, 6, 7),
    n_dqi_samples: int = 1000,
    sa_iter: int = 10000,
    top_k_by_surrogate: int = 10,
    seed: int | None = 42,
    include_stage1_paired_encoding: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run Matrix A (P-family scaling) and persist standardized artifacts."""
    layout = ensure_output_layout(output_root)
    feature_list = [int(x) for x in features]
    if smoke:
        feature_list = [min(feature_list)] if feature_list else [3]
        n_dqi_samples = min(int(n_dqi_samples), 24)
        top_k_by_surrogate = min(int(top_k_by_surrogate), 5)

    records: list[dict[str, Any]] = []
    alphas = ["uniform", "paper", "heuristic"]
    decoders = ["bruteforce", "bp1"]
    for n_features in feature_list:
        prob, _from_frozen, _path = load_scaling_instance(n_features)
        instance_id = f"P{n_features}"
        if n_features >= 5:
            k = 15
        else:
            k = min(12, 2 ** (2 * n_features) - 1)
        ell = 1 if n_features >= 6 else min(3, max_safe_ell(k, prob.n_bits))
        is_valid, message = validate_dqi_parameters(k, prob.n_bits, ell)
        if not is_valid:
            raise ValueError(f"Invalid Matrix A parameters for P{n_features}: {message}")

        estimated_qubits = _pricing_estimated_qubits(n_bits=2 * n_features, m_terms=k, ell=ell)
        estimated_memory_gb = _estimated_statevector_memory_gb(estimated_qubits)

        if n_features >= 7 and estimated_qubits > 30:
            for encoding in ("wht_truncated", "ilp_derived"):
                for decoder in decoders:
                    for alpha_mode in alphas:
                        execution_mode = "mixture"
                        run_seed = _to_int(seed)
                        label = "wht" if encoding == "wht_truncated" else "ilp"
                        run_id = _safe_slug(
                            f"matrix_a_{instance_id}_{label}_{decoder}_{alpha_mode}_{execution_mode}_seed{run_seed}"
                        )
                        record = _base_record(
                            matrix="matrix_a",
                            run_id=run_id,
                            run_status="unavailable",
                            run_error="qubit threshold exceeded",
                            instance_id=instance_id,
                            family="P",
                            encoding=encoding,
                            decoder=decoder,
                            alpha_mode=alpha_mode,
                            execution_mode=execution_mode,
                            seed=run_seed,
                        )
                        record["ell"] = int(ell)
                        record.update(
                            build_boundary_record(
                                base_record=record,
                                run_status="unavailable",
                                boundary_reason="qubit_count_exceeded",
                                attempted_execution=False,
                                estimated_qubits=estimated_qubits,
                                estimated_memory_gb=estimated_memory_gb,
                                error_type=None,
                                error_message=None,
                                wall_clock_s=0.0,
                                peak_memory_mb=0.0,
                            )
                        )
                        _attach_metric_availability(record)
                        _validate_record_shape(record)
                        records.append(record)
            continue

        out = run_benchmark(
            prob,
            k=k,
            ell=ell,
            n_dqi_samples=int(n_dqi_samples),
            sa_iter=int(sa_iter),
            top_k_by_surrogate=int(top_k_by_surrogate),
            seed=seed,
            include_stage1_paired_encoding=True,
        )
        x_opt = int(out["ground_truth"]["x_opt"])
        f_opt = float(out["ground_truth"]["f_opt"])
        resources_by_decoder = out.get("resources_by_decoder_model", {})
        existing_wht_runs = {
            ("bruteforce", "uniform", "mixture"): out.get("dqi_uniform_mixture"),
            ("bruteforce", "paper", "mixture"): out.get("dqi_paper_mixture"),
            ("bruteforce", "heuristic", "mixture"): out.get("dqi_heuristic_mixture"),
            ("bp1", "paper", "mixture"): out.get("dqi_paper_mixture_bp1"),
        }

        for decoder in decoders:
            for alpha_mode in alphas:
                execution_mode = "mixture"
                run_seed = _to_int(seed)
                run_id = _safe_slug(
                    f"matrix_a_{instance_id}_wht_{decoder}_{alpha_mode}_{execution_mode}_seed{run_seed}"
                )

                resource_summary = None
                if isinstance(resources_by_decoder, dict):
                    resource_summary = resources_by_decoder.get(decoder)
                if resource_summary is None and decoder == "bruteforce":
                    resource_summary = out.get("resources")

                run = existing_wht_runs.get((decoder, alpha_mode, execution_mode))
                run_status = "completed"
                run_error = None
                decision_metrics = None
                best_f = None
                best_g = None
                runtime = None

                if isinstance(run, dict):
                    run_status = _canonical_run_status(run.get("status", "completed"), default="completed")
                    if run_status != "completed":
                        reason = run.get("reason")
                        err = run.get("error")
                        run_error = (
                            str(reason)
                            if reason is not None
                            else (str(err) if err is not None else None)
                        )
                    decision_metrics = (
                        run.get("decision_metrics")
                        if isinstance(run.get("decision_metrics"), dict)
                        else None
                    )
                    best_f = _to_float(run.get("best_F"))
                    best_g = _to_float(run.get("best_G_weighted", run.get("best_G")))
                    runtime = _to_float(run.get("time"))
                else:
                    t0 = time.perf_counter()
                    try:
                        pipeline = DQIPipeline(
                            prob,
                            k=k,
                            ell=ell,
                            alpha_mode=alpha_mode,
                            execution_mode=execution_mode,
                            decoder_mode=decoder,
                        )
                        dqi_run = pipeline.run_with_metrics(
                            n_samples=int(n_dqi_samples),
                            seed=seed,
                        )
                        elapsed = float(time.perf_counter() - t0)
                        decision_metrics = compute_decision_metrics(
                            F_values=np.asarray(dqi_run["F_values"], dtype=np.float64).tolist(),
                            G_values=np.asarray(dqi_run["G_weighted_values"], dtype=np.float64).tolist(),
                            f_opt=f_opt,
                            k_eval=int(top_k_by_surrogate),
                            sampled_x=np.asarray(dqi_run["samples"], dtype=np.int64).tolist(),
                            optimum_x=[x_opt],
                            decode_success=_to_float(dqi_run.get("decoder_success_rate")),
                            postselection_success=_to_float(dqi_run.get("success_prob")),
                            retained_energy_eta=_to_float(dqi_run.get("energy_fraction")),
                            candidate_source_label=(
                                f"matrix_a|{instance_id}|wht_truncated|{decoder}|{alpha_mode}|mixture"
                            ),
                            runtime_s=elapsed,
                        )
                        best_f = _to_float(dqi_run.get("best_F"))
                        best_g = _to_float(dqi_run.get("best_G_weighted", dqi_run.get("best_G")))
                        runtime = elapsed
                    except Exception as exc:  # pragma: no cover - defensive path
                        run_status = "failed"
                        run_error = str(exc)

                metric_vals = _metric_fields(decision_metrics)
                record = _base_record(
                    matrix="matrix_a",
                    run_id=run_id,
                    run_status=run_status,
                    run_error=run_error,
                    instance_id=instance_id,
                    family="P",
                    encoding="wht_truncated",
                    decoder=decoder,
                    alpha_mode=alpha_mode,
                    execution_mode=execution_mode,
                    seed=run_seed,
                )
                record["ell"] = int(ell)
                record["attempted_execution"] = True
                record["estimated_qubits"] = estimated_qubits
                record["estimated_memory_gb"] = estimated_memory_gb
                record["best_sampled_F"] = best_f
                record["best_sampled_G"] = best_g
                record["runtime_sec"] = _to_float(runtime if runtime is not None else metric_vals["runtime_sec"])
                record["wall_clock_s"] = record["runtime_sec"]
                for key, value in metric_vals.items():
                    if key == "runtime_sec":
                        continue
                    record[key] = value
                record.update(_resource_fields(resource_summary))
                _attach_metric_availability(record)
                _validate_record_shape(record)
                records.append(record)

        coherent_run = out.get("dqi_paper_coherent")
        if isinstance(coherent_run, dict):
            decoder = str(coherent_run.get("decoder_mode", "bruteforce"))
            run_seed = _to_int(seed)
            run_id = _safe_slug(
                f"matrix_a_{instance_id}_wht_{decoder}_paper_coherent_seed{run_seed}"
            )
            resource_summary = None
            if isinstance(resources_by_decoder, dict):
                resource_summary = resources_by_decoder.get(decoder)
            if resource_summary is None and decoder == "bruteforce":
                resource_summary = out.get("resources")

            decision_metrics = (
                coherent_run.get("decision_metrics")
                if isinstance(coherent_run.get("decision_metrics"), dict)
                else None
            )
            metric_vals = _metric_fields(decision_metrics)
            run_status = _canonical_run_status(coherent_run.get("status", "completed"), default="completed")
            run_error = (
                str(coherent_run.get("reason"))
                if coherent_run.get("reason") is not None
                else None
            )
            record = _base_record(
                matrix="matrix_a",
                run_id=run_id,
                run_status=run_status,
                run_error=run_error,
                instance_id=instance_id,
                family="P",
                encoding="wht_truncated",
                decoder=decoder,
                alpha_mode=str(coherent_run.get("alpha_mode", "paper")),
                execution_mode="coherent",
                seed=run_seed,
            )
            record["ell"] = int(ell)
            record["attempted_execution"] = True
            record["estimated_qubits"] = estimated_qubits
            record["estimated_memory_gb"] = estimated_memory_gb
            record["best_sampled_F"] = _to_float(coherent_run.get("best_F"))
            record["best_sampled_G"] = _to_float(
                coherent_run.get("best_G_weighted", coherent_run.get("best_G"))
            )
            runtime = coherent_run.get("time")
            record["runtime_sec"] = _to_float(runtime if runtime is not None else metric_vals["runtime_sec"])
            record["wall_clock_s"] = record["runtime_sec"]
            for key, value in metric_vals.items():
                if key == "runtime_sec":
                    continue
                record[key] = value
            record.update(_resource_fields(resource_summary))
            _attach_metric_availability(record)
            _validate_record_shape(record)
            records.append(record)

        for decoder in decoders:
            for alpha_mode in alphas:
                run_seed = _to_int(seed)
                run_id = _safe_slug(
                    f"matrix_a_{instance_id}_ilp_{decoder}_{alpha_mode}_mixture_seed{run_seed}"
                )
                run_status = "completed"
                run_error = None
                metric_vals = _metric_fields(None)
                best_f = None
                best_g = None
                comparison_id = None
                try:
                    capture = capture_execution_metrics(
                        lambda: run_paired_encoding_comparison(
                            prob=prob,
                            k_wht=int(k),
                            alpha_mode=alpha_mode,
                            execution_mode="mixture",
                            decoder_mode=decoder,
                            n_samples=int(n_dqi_samples),
                            seed=seed,
                            top_k_eval=int(top_k_by_surrogate),
                        )
                    )
                    paired = capture.result
                    comparison_id = paired.get("comparison_id")
                    metric_vals = _metric_fields(paired.get("backend_b_metrics"))
                    backend_b_core = (
                        (paired.get("backend_b_metrics") or {}).get("core_decision") or {}
                    )
                    best_f = _to_float(backend_b_core.get("best_sampled_F"))
                    best_g = _to_float(backend_b_core.get("best_sampled_G"))
                    wall_clock_s = capture.wall_clock_s
                    peak_memory_mb = capture.peak_memory_mb
                except Exception as exc:  # pragma: no cover - defensive path
                    run_status = "failed"
                    run_error = str(exc)
                    wall_clock_s = None
                    peak_memory_mb = None

                record = _base_record(
                    matrix="matrix_a",
                    run_id=run_id,
                    run_status=run_status,
                    run_error=run_error,
                    instance_id=instance_id,
                    family="P",
                    encoding="ilp_derived",
                    decoder=decoder,
                    alpha_mode=alpha_mode,
                    execution_mode="mixture",
                    seed=run_seed,
                )
                record["ell"] = int(ell)
                record["estimated_qubits"] = estimated_qubits
                record["estimated_memory_gb"] = estimated_memory_gb
                record["best_sampled_F"] = best_f
                record["best_sampled_G"] = best_g
                for key, value in metric_vals.items():
                    if key == "runtime_sec":
                        record[key] = value
                        continue
                    record[key] = value
                if run_status == "completed":
                    record["attempted_execution"] = True
                    record["wall_clock_s"] = _to_float(wall_clock_s)
                    record["peak_memory_mb"] = _to_float(peak_memory_mb)
                    record["runtime_sec"] = _to_float(record.get("runtime_sec"))
                else:
                    boundary_reason, error_type, error_message = classify_boundary_exception(
                        RuntimeError(run_error or "runtime failure")
                    )
                    record = build_boundary_record(
                        base_record=record,
                        run_status="failed",
                        boundary_reason=boundary_reason,
                        attempted_execution=True,
                        estimated_qubits=estimated_qubits,
                        estimated_memory_gb=estimated_memory_gb,
                        error_type=error_type,
                        error_message=error_message,
                        wall_clock_s=wall_clock_s,
                        peak_memory_mb=peak_memory_mb,
                    )
                record["paired_comparison_id"] = comparison_id
                _attach_metric_availability(record)
                _validate_record_shape(record)
                records.append(record)

    config = {
        "features": feature_list,
        "n_dqi_samples": int(n_dqi_samples),
        "sa_iter": int(sa_iter),
        "top_k_by_surrogate": int(top_k_by_surrogate),
        "seed": _to_int(seed),
        "include_stage1_paired_encoding": True,
        "encodings": ["wht_truncated", "ilp_derived"],
        "decoders": decoders,
        "alpha_modes": alphas,
        "execution_mode": "mixture",
        "smoke": bool(smoke),
    }
    out_paths = _persist_matrix_artifacts(
        matrix_name="matrix_a",
        records=records,
        layout=layout,
        config=config,
        csv_filename="matrix_a_summary.csv",
    )
    report_path = layout["reports"] / "matrix_a_report.json"
    by_status: dict[str, int] = {}
    by_encoding: dict[str, int] = {}
    unavailable_by_metric = {metric: 0 for metric in METRIC_FIELDS}
    for rec in records:
        status = _canonical_run_status(rec.get("run_status"), default="unavailable")
        by_status[status] = by_status.get(status, 0) + 1
        by_encoding[rec["encoding"]] = by_encoding.get(rec["encoding"], 0) + 1
        for metric in METRIC_FIELDS:
            if rec.get(metric) is None:
                unavailable_by_metric[metric] += 1
    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "matrix": "matrix_a",
        "generated_at": _now_iso_utc(),
        "config": config,
        "record_count": len(records),
        "by_status": by_status,
        "by_encoding": by_encoding,
        "unavailable_by_metric": unavailable_by_metric,
        "required_record_fields": REQUIRED_RECORD_FIELDS,
        "records": records,
    }
    with open(report_path, "w") as f:
        json.dump(report_payload, f, indent=2)
    out_paths["report_json"] = report_path
    out_paths["records"] = records
    return out_paths


def run_matrix_b(
    *,
    output_root: str | Path = "results",
    instance_ids: Iterable[str] | None = None,
    alpha_modes: Iterable[str] = ("paper",),
    execution_modes: Iterable[str] = ("mixture", "coherent"),
    decoders: Iterable[str] = ("bruteforce", "bp1"),
    n_dqi_samples: int = 256,
    top_k_by_surrogate: int = 10,
    seed: int | None = 42,
    ell: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run Matrix B (C-family coherent tiny instances) and persist artifacts."""
    layout = ensure_output_layout(output_root)
    family = load_coherent_family()
    all_ids = [rec["meta"]["instance_id"] for rec in family]

    if instance_ids is None:
        target_ids = all_ids
    else:
        requested = [str(x) for x in instance_ids]
        unknown = sorted(set(requested).difference(all_ids))
        if unknown:
            raise KeyError(f"Unknown Matrix B instance_id values: {unknown}")
        target_ids = [x for x in all_ids if x in set(requested)]

    alpha_list = [str(x) for x in alpha_modes]
    exec_list = [str(x) for x in execution_modes]
    decoder_list = [str(x) for x in decoders]

    if smoke:
        target_ids = target_ids[:1]
        alpha_list = ["paper"]
        exec_list = ["mixture"]
        decoder_list = ["oracle"]
        n_dqi_samples = min(int(n_dqi_samples), 24)
        top_k_by_surrogate = min(int(top_k_by_surrogate), 5)

    by_id = {rec["meta"]["instance_id"]: rec for rec in family}
    records: list[dict[str, Any]] = []
    probability_lookup: dict[str, list[float]] = {}
    run_index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for instance_id in target_ids:
        rec = by_id[instance_id]
        meta = rec["meta"]
        arrays = rec["arrays"]

        B = np.asarray(arrays["B"], dtype=np.int8)
        v = np.asarray(arrays["v"], dtype=np.int8)
        truth_f = np.asarray(arrays["truth_table_F"], dtype=np.float64)
        truth_g = np.asarray(arrays["truth_table_G"], dtype=np.float64)
        n_bits = int(meta["n_bits"])
        ell_eff = int(meta["ell_default"] if ell is None else ell)
        f_star = float(np.max(truth_f))
        optimum_set = np.flatnonzero(np.isclose(truth_f, f_star)).astype(np.int64).tolist()
        encoding = str(meta.get("encoding_backend", "wht_exact"))
        ilp_feasible = (
            str(meta.get("source_type", "")) == "pricing_ilp_exact"
            or meta.get("exact_encoding_source_path") not in (None, "")
        )
        encoding_modes = [("wht_exact", True), ("ilp_derived_exact", ilp_feasible)]

        for encoding_mode, feasible in encoding_modes:
            if not feasible:
                continue
            for alpha_mode in alpha_list:
                alpha_vec = _alpha_vector_for_mode(alpha_mode=alpha_mode, B=B, v=v, ell=ell_eff)
                for execution_mode in exec_list:
                    for idx, decoder in enumerate(decoder_list):
                        internal_decoder, decoder_label = _matrix_b_decoder_modes(decoder)
                        run_seed = None if seed is None else int(seed + idx)
                        decode_model = _decode_model_for_decoder(internal_decoder)
                        # Qiskit structural counts are decoder-agnostic (they measure
                        # the implemented subset without decode). Enable for all decoders.
                        resource_summary = dqi_resource_estimate(
                            B=B,
                            v=v,
                            ell=ell_eff,
                            decode_uncompute_model=decode_model,
                            include_qiskit_structural_count=True,
                        )
                        diag = decoder_diagnostics(B=B, ell=ell_eff, decoder_mode=internal_decoder)
                        run_id = _safe_slug(
                            f"matrix_b_{instance_id}_{encoding_mode}_{decoder_label}_{alpha_mode}_{execution_mode}_seed{run_seed}"
                        )

                        if execution_mode == "coherent" and not bool(meta.get("coherent_supported", False)):
                            record = _base_record(
                                matrix="matrix_b",
                                run_id=run_id,
                                run_status="not_applicable",
                                run_error="coherent_not_supported_by_instance_metadata",
                                instance_id=instance_id,
                                family="C",
                                encoding=encoding_mode,
                                decoder=decoder_label,
                                alpha_mode=alpha_mode,
                                execution_mode=execution_mode,
                                seed=run_seed,
                            )
                            record["coherent_mode_record"] = True
                            record["decoder_internal_mode"] = internal_decoder
                            record["coherent_vs_mixture_delta_best_F"] = None
                            record["coherent_vs_mixture_delta_best_top_G_sampled_F"] = None
                            record["coherent_vs_mixture_delta_postselection_success"] = None
                            record["coherent_vs_mixture_distribution_distance"] = None
                            record["coherent_vs_mixture_distribution_metric"] = "tvd"
                            record.update(_resource_fields(resource_summary))
                            _attach_metric_availability(record)
                            _validate_record_shape(record)
                            records.append(record)
                            run_index[
                                (
                                    instance_id,
                                    encoding_mode,
                                    decoder_label,
                                    alpha_mode,
                                    execution_mode,
                                )
                            ] = record
                            continue

                        t0 = time.perf_counter()
                        try:
                            samples, probabilities, success_prob = run_dqi_statevector(
                                B=B,
                                v=v,
                                alpha=alpha_vec,
                                ell=ell_eff,
                                n_samples=int(n_dqi_samples),
                                seed=run_seed,
                                decoder_mode=internal_decoder,
                                execution_mode=execution_mode,
                            )
                            elapsed = float(time.perf_counter() - t0)
                            samples_arr = np.asarray(samples, dtype=np.int64)
                            f_values = truth_f[samples_arr]
                            g_values = truth_g[samples_arr]
                            best_idx = int(np.argmax(f_values))

                            decision_metrics = compute_decision_metrics(
                                F_values=f_values.tolist(),
                                G_values=g_values.tolist(),
                                f_opt=f_star,
                                k_eval=int(top_k_by_surrogate),
                                sampled_x=samples_arr.tolist(),
                                optimum_x=optimum_set,
                                decode_success=_to_float(diag.get("decoder_success_rate")),
                                postselection_success=float(success_prob),
                                retained_energy_eta=1.0,
                                candidate_source_label=(
                                    f"family_c|{instance_id}|{encoding_mode}|{decoder_label}|{alpha_mode}|{execution_mode}"
                                ),
                                runtime_s=elapsed,
                            )
                            metric_vals = _metric_fields(decision_metrics)

                            record = _base_record(
                                matrix="matrix_b",
                                run_id=run_id,
                                run_status="completed",
                                run_error=None,
                                instance_id=instance_id,
                                family="C",
                                encoding=encoding_mode,
                                decoder=decoder_label,
                                alpha_mode=alpha_mode,
                                execution_mode=execution_mode,
                                seed=run_seed,
                            )
                            record["best_sampled_F"] = _to_float(f_values[best_idx])
                            record["best_sampled_G"] = _to_float(np.max(g_values))
                            record["runtime_sec"] = elapsed
                            for key, value in metric_vals.items():
                                if key == "runtime_sec":
                                    continue
                                record[key] = value
                            if record["decode_success"] is None:
                                record["decode_success"] = _to_float(diag.get("decoder_success_rate"))
                            if record["postselection_success"] is None:
                                record["postselection_success"] = _to_float(success_prob)
                            record["coherent_mode_record"] = execution_mode == "coherent"
                            record["decoder_internal_mode"] = internal_decoder
                            record["coherent_vs_mixture_delta_best_F"] = None
                            record["coherent_vs_mixture_delta_best_top_G_sampled_F"] = None
                            record["coherent_vs_mixture_delta_postselection_success"] = None
                            record["coherent_vs_mixture_distribution_distance"] = None
                            record["coherent_vs_mixture_distribution_metric"] = "tvd"
                            record.update(_resource_fields(resource_summary))
                            probability_lookup[run_id] = [
                                float(p) for p in np.asarray(probabilities, dtype=np.float64)
                            ]
                        except Exception as exc:  # pragma: no cover - defensive path
                            record = _base_record(
                                matrix="matrix_b",
                                run_id=run_id,
                                run_status="failed",
                                run_error=str(exc),
                                instance_id=instance_id,
                                family="C",
                                encoding=encoding_mode,
                                decoder=decoder_label,
                                alpha_mode=alpha_mode,
                                execution_mode=execution_mode,
                                seed=run_seed,
                            )
                            record["coherent_mode_record"] = execution_mode == "coherent"
                            record["decoder_internal_mode"] = internal_decoder
                            record["coherent_vs_mixture_delta_best_F"] = None
                            record["coherent_vs_mixture_delta_best_top_G_sampled_F"] = None
                            record["coherent_vs_mixture_delta_postselection_success"] = None
                            record["coherent_vs_mixture_distribution_distance"] = None
                            record["coherent_vs_mixture_distribution_metric"] = "tvd"
                            record.update(_resource_fields(resource_summary))
                        _attach_metric_availability(record)
                        _validate_record_shape(record)
                        records.append(record)
                        run_index[
                            (
                                instance_id,
                                encoding_mode,
                                decoder_label,
                                alpha_mode,
                                execution_mode,
                            )
                        ] = record

    coherent_vs_mixture: list[dict[str, Any]] = []
    for instance_id in target_ids:
        for encoding_mode, feasible in [("wht_exact", True), ("ilp_derived_exact", True)]:
            if encoding_mode == "ilp_derived_exact":
                meta = by_id[instance_id]["meta"]
                feasible = (
                    str(meta.get("source_type", "")) == "pricing_ilp_exact"
                    or meta.get("exact_encoding_source_path") not in (None, "")
                )
            if not feasible:
                continue
            for decoder_label in ["oracle", "bp1"]:
                for alpha_mode in alpha_list:
                    key_mix = (instance_id, encoding_mode, decoder_label, alpha_mode, "mixture")
                    key_coh = (instance_id, encoding_mode, decoder_label, alpha_mode, "coherent")
                    mix = run_index.get(key_mix)
                    coh = run_index.get(key_coh)
                    if not isinstance(mix, dict) or not isinstance(coh, dict):
                        continue
                    if mix.get("run_status") != "completed" or coh.get("run_status") != "completed":
                        continue
                    delta_best_f = _to_float(coh.get("best_sampled_F")) - _to_float(mix.get("best_sampled_F"))
                    delta_best_top_g = _to_float(coh.get("best_top_G_sampled_F")) - _to_float(
                        mix.get("best_top_G_sampled_F")
                    )
                    delta_postselection = _to_float(coh.get("postselection_success")) - _to_float(
                        mix.get("postselection_success")
                    )
                    tvd_out = compare_candidate_distributions_tvd(
                        {"probabilities": probability_lookup.get(mix["run_id"])},
                        {"probabilities": probability_lookup.get(coh["run_id"])},
                    )
                    tvd_value = tvd_out.get("tvd") if tvd_out.get("ok") else None
                    comparison = {
                        "instance_id": instance_id,
                        "encoding": encoding_mode,
                        "decoder": decoder_label,
                        "alpha_mode": alpha_mode,
                        "delta_direction": "coherent_minus_mixture",
                        "delta_best_F": delta_best_f,
                        "delta_best_top_G_sampled_F": delta_best_top_g,
                        "delta_postselection_success": delta_postselection,
                        "distribution_metric": "tvd",
                        "distribution_distance": _to_float(tvd_value),
                    }
                    coherent_vs_mixture.append(comparison)
                    for row in (mix, coh):
                        row["coherent_vs_mixture_delta_best_F"] = comparison["delta_best_F"]
                        row["coherent_vs_mixture_delta_best_top_G_sampled_F"] = comparison[
                            "delta_best_top_G_sampled_F"
                        ]
                        row["coherent_vs_mixture_delta_postselection_success"] = comparison[
                            "delta_postselection_success"
                        ]
                        row["coherent_vs_mixture_distribution_distance"] = comparison[
                            "distribution_distance"
                        ]
                        row["coherent_vs_mixture_distribution_metric"] = comparison[
                            "distribution_metric"
                        ]

    config = {
        "instance_ids": target_ids,
        "alpha_modes": alpha_list,
        "execution_modes": exec_list,
        "decoders": decoder_list,
        "decoder_labels": ["oracle", "bp1"],
        "encodings": ["wht_exact", "ilp_derived_exact_where_feasible"],
        "n_dqi_samples": int(n_dqi_samples),
        "top_k_by_surrogate": int(top_k_by_surrogate),
        "seed": _to_int(seed),
        "ell": _to_int(ell),
        "smoke": bool(smoke),
    }
    out_paths = _persist_matrix_artifacts(
        matrix_name="matrix_b",
        records=records,
        layout=layout,
        config=config,
        csv_filename="matrix_b_summary.csv",
    )
    report_path = layout["reports"] / "matrix_b_report.json"
    by_status: dict[str, int] = {}
    by_instance: dict[str, int] = {}
    coherent_status_by_instance: dict[str, dict[str, int]] = {}
    for rec in records:
        status = _canonical_run_status(rec.get("run_status"), default="unavailable")
        by_status[status] = by_status.get(status, 0) + 1
        by_instance[rec["instance_id"]] = by_instance.get(rec["instance_id"], 0) + 1
        inst = rec["instance_id"]
        coherent_status_by_instance.setdefault(inst, {})
        if rec.get("execution_mode") == "coherent":
            coherent_status_by_instance[inst][status] = (
                coherent_status_by_instance[inst].get(status, 0) + 1
            )

    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "matrix": "matrix_b",
        "generated_at": _now_iso_utc(),
        "config": config,
        "record_count": len(records),
        "by_status": by_status,
        "by_instance": by_instance,
        "coherent_status_by_instance": coherent_status_by_instance,
        "coherent_vs_mixture_comparisons": coherent_vs_mixture,
        "required_record_fields": REQUIRED_RECORD_FIELDS,
        "records": records,
    }
    with open(report_path, "w") as f:
        json.dump(report_payload, f, indent=2)
    out_paths["report_json"] = report_path
    out_paths["records"] = records
    return out_paths


def run_matrix_c(
    *,
    output_root: str | Path = "results",
    instance_ids: Iterable[str] | None = None,
    alpha_modes: Iterable[str] = ("paper", "uniform"),
    execution_mode: str = "mixture",
    decoders: Iterable[str] = ("bruteforce", "bp1", "bp+osd-lite", "oracle"),
    n_dqi_samples: int = 256,
    top_k_by_surrogate: int = 10,
    seed: int | None = 42,
    ell: int | None = None,
    p_features: Iterable[int] = (3, 4, 5, 6, 7),
    include_p_family: bool = True,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run Matrix C (S-family + applicable P-family) and persist artifacts."""
    if execution_mode != "mixture":
        raise ValueError("Matrix C supports execution_mode='mixture' only.")

    layout = ensure_output_layout(output_root)
    alpha_list = [str(a).strip() for a in alpha_modes if str(a).strip()]
    if not alpha_list:
        alpha_list = ["paper", "uniform"]
    decoder_list = [_normalize_matrix_c_decoder_label(d) for d in decoders]
    # Preserve order while de-duplicating.
    seen_decoders: set[str] = set()
    decoder_list = [d for d in decoder_list if not (d in seen_decoders or seen_decoders.add(d))]
    instance_list = None if instance_ids is None else [str(x) for x in instance_ids]
    p_feature_list = [int(x) for x in p_features]

    internal_needed: list[str] = []
    if any(_matrix_c_internal_decoder(d) == "bruteforce" for d in decoder_list):
        internal_needed.append("bruteforce")
    if any(_matrix_c_internal_decoder(d) == "bp1" for d in decoder_list):
        internal_needed.append("bp1")

    if smoke:
        instance_list = ["S_dense_low_collision"]
        alpha_list = ["paper"]
        decoder_list = ["bruteforce", "bp1", "bp+osd-lite", "oracle"]
        internal_needed = ["bruteforce", "bp1"]
        p_feature_list = [3]
        n_dqi_samples = min(int(n_dqi_samples), 24)
        top_k_by_surrogate = min(int(top_k_by_surrogate), 5)

    records: list[dict[str, Any]] = []
    # S-family panel
    for alpha_mode in alpha_list:
        runs = run_structure_sweep_benchmark(
            instance_ids=instance_list,
            ell=ell,
            alpha_mode=alpha_mode,
            execution_mode=execution_mode,
            decoders=tuple(internal_needed or ("bruteforce",)),
            n_dqi_samples=int(n_dqi_samples),
            top_k_by_surrogate=int(top_k_by_surrogate),
            seed=seed,
            output_dir=layout["matrix_c"],
            write_json=False,
        )
        decoder_seed_lookup = {
            decoder: (None if seed is None else int(seed + idx))
            for idx, decoder in enumerate(internal_needed or ["bruteforce"])
        }
        for sweep_rec in runs:
            instance_id = str(sweep_rec["instance_id"])
            encoding = str(sweep_rec.get("source_type", "synthetic_structured_bv"))
            structure = sweep_rec.get("structure_metrics")
            dqi_by_decoder = sweep_rec.get("dqi_by_decoder", {})

            for decoder_label in decoder_list:
                internal_decoder = _matrix_c_internal_decoder(decoder_label)
                run_seed = decoder_seed_lookup.get(internal_decoder, _to_int(seed))
                run_id = _safe_slug(
                    f"matrix_c_S_{instance_id}_{encoding}_{decoder_label}_{alpha_mode}_mixture_seed{run_seed}"
                )
                if internal_decoder is None:
                    record = _base_record(
                        matrix="matrix_c",
                        run_id=run_id,
                        run_status="unavailable",
                        run_error="decoder_not_implemented_in_repo",
                        instance_id=instance_id,
                        family="S",
                        encoding=encoding,
                        decoder=decoder_label,
                        alpha_mode=alpha_mode,
                        execution_mode="mixture",
                        seed=run_seed,
                    )
                    # S-family uses non-WHT encodings; energy_eta is not applicable
                    record["energy_eta_applicable"] = False
                    record.update(_structure_fields(structure))
                    record["runtime"] = None
                    record["gain_over_bp1"] = None
                    record["gain_over_bruteforce"] = None
                    _attach_metric_availability(record)
                    _validate_record_shape(record)
                    records.append(record)
                    continue

                source_run = dqi_by_decoder.get(internal_decoder)
                if not isinstance(source_run, dict):
                    record = _base_record(
                        matrix="matrix_c",
                        run_id=run_id,
                        run_status="unavailable",
                        run_error=f"decoder_run_missing:{internal_decoder}",
                        instance_id=instance_id,
                        family="S",
                        encoding=encoding,
                        decoder=decoder_label,
                        alpha_mode=alpha_mode,
                        execution_mode="mixture",
                        seed=run_seed,
                    )
                    # S-family uses non-WHT encodings; energy_eta is not applicable
                    record["energy_eta_applicable"] = False
                    record.update(_structure_fields(structure))
                    record["runtime"] = None
                    record["gain_over_bp1"] = None
                    record["gain_over_bruteforce"] = None
                    _attach_metric_availability(record)
                    _validate_record_shape(record)
                    records.append(record)
                    continue

                run_status = _canonical_run_status(source_run.get("status", "completed"), default="completed")
                record = _base_record(
                    matrix="matrix_c",
                    run_id=run_id,
                    run_status=run_status,
                    run_error=source_run.get("error"),
                    instance_id=instance_id,
                    family="S",
                    encoding=encoding,
                    decoder=decoder_label,
                    alpha_mode=alpha_mode,
                    execution_mode="mixture",
                    seed=run_seed,
                )
                # S-family uses non-WHT encodings; energy_eta is not applicable
                record["energy_eta_applicable"] = False
                # Propagate status_reason fields for not_applicable cases
                if "status_reason_code" in source_run:
                    record["status_reason_code"] = source_run["status_reason_code"]
                if "status_reason" in source_run:
                    record["status_reason"] = source_run["status_reason"]
                if run_status == "completed":
                    record["best_sampled_F"] = _to_float(source_run.get("best_F"))
                    record["best_sampled_G"] = _to_float(source_run.get("best_G"))
                    record["runtime_sec"] = _to_float(source_run.get("elapsed_s"))
                    metric_vals = _metric_fields(source_run.get("decision_metrics"))
                    for key, value in metric_vals.items():
                        if key == "runtime_sec":
                            if record["runtime_sec"] is None:
                                record["runtime_sec"] = value
                            continue
                        record[key] = value
                    if record["decode_success"] is None:
                        record["decode_success"] = _to_float(
                            (source_run.get("decoder_diagnostics") or {}).get("decoder_success_rate")
                        )
                    if record["postselection_success"] is None:
                        record["postselection_success"] = _to_float(source_run.get("success_prob"))
                record.update(_resource_fields(source_run.get("resources")))
                record.update(_structure_fields(structure))
                record["runtime"] = record.get("runtime_sec")
                record["decoder_internal_mode"] = internal_decoder
                if decoder_label == "oracle":
                    record["decoder_alias_of"] = "bruteforce"
                record["gain_over_bp1"] = None
                record["gain_over_bruteforce"] = None
                _attach_metric_availability(record)
                _validate_record_shape(record)
                records.append(record)

    # P-family panel (where applicable)
    if include_p_family:
        for n_features in p_feature_list:
            prob, _from_frozen, _path = load_scaling_instance(n_features)
            instance_id = f"P{n_features}"
            if n_features >= 5:
                k = 15
            else:
                k = min(12, 2 ** (2 * n_features) - 1)
            ell_eff = int((1 if n_features >= 6 else min(3, max_safe_ell(k, prob.n_bits))) if ell is None else ell)
            is_valid, message = validate_dqi_parameters(k, prob.n_bits, ell_eff)
            if not is_valid:
                raise ValueError(f"Invalid Matrix C P-family parameters for {instance_id}: {message}")

            estimated_qubits = _pricing_estimated_qubits(n_bits=2 * n_features, m_terms=k, ell=ell_eff)
            estimated_memory_gb = _estimated_statevector_memory_gb(estimated_qubits)

            f_table = np.array([prob.evaluate(x) for x in range(2 ** prob.n_bits)], dtype=np.float64)
            f_opt = float(np.max(f_table))
            optimum_set = np.flatnonzero(np.isclose(f_table, f_opt)).astype(np.int64).tolist()

            for alpha_mode in alpha_list:
                # Build once to capture shared surrogate/structure payload.
                base_pipe = DQIPipeline(
                    prob,
                    k=k,
                    ell=ell_eff,
                    alpha_mode=alpha_mode,
                    execution_mode="mixture",
                    decoder_mode="bruteforce",
                )
                B = np.asarray(base_pipe.B, dtype=np.int8)
                v = np.asarray(base_pipe.v, dtype=np.int8)
                structure = compute_structure_metrics(B, ell_eff)

                if n_features >= 7 and estimated_qubits > 30:
                    for idx, decoder_label in enumerate(decoder_list):
                        run_seed = None if seed is None else int(seed + idx)
                        run_id = _safe_slug(
                            f"matrix_c_P_{instance_id}_wht_truncated_pricing_{decoder_label}_{alpha_mode}_mixture_seed{run_seed}"
                        )
                        record = _base_record(
                            matrix="matrix_c",
                            run_id=run_id,
                            run_status="unavailable",
                            run_error="qubit threshold exceeded",
                            instance_id=instance_id,
                            family="P",
                            encoding="wht_truncated_pricing",
                            decoder=decoder_label,
                            alpha_mode=alpha_mode,
                            execution_mode="mixture",
                            seed=run_seed,
                        )
                        record["ell"] = int(ell_eff)
                        record.update(
                            build_boundary_record(
                                base_record=record,
                                run_status="unavailable",
                                boundary_reason="qubit_count_exceeded",
                                attempted_execution=False,
                                estimated_qubits=estimated_qubits,
                                estimated_memory_gb=estimated_memory_gb,
                                error_type=None,
                                error_message=None,
                                wall_clock_s=0.0,
                                peak_memory_mb=0.0,
                            )
                        )
                        record.update(_structure_fields(structure))
                        record["runtime"] = record.get("runtime_sec")
                        _attach_metric_availability(record)
                        _validate_record_shape(record)
                        records.append(record)
                    continue

                internal_results: dict[str, dict[str, Any]] = {}
                for idx, internal_decoder in enumerate(internal_needed):
                    run_seed = None if seed is None else int(seed + idx)
                    try:
                        pipe = DQIPipeline(
                            prob,
                            k=k,
                            ell=ell_eff,
                            alpha_mode=alpha_mode,
                            execution_mode="mixture",
                            decoder_mode=internal_decoder,
                        )
                        capture = capture_execution_metrics(
                            lambda: pipe.run_with_metrics(n_samples=int(n_dqi_samples), seed=run_seed)
                        )
                        out = capture.result
                        elapsed = float(capture.wall_clock_s)
                        decision_metrics = compute_decision_metrics(
                            F_values=np.asarray(out["F_values"], dtype=np.float64).tolist(),
                            G_values=np.asarray(out["G_weighted_values"], dtype=np.float64).tolist(),
                            f_opt=f_opt,
                            k_eval=int(top_k_by_surrogate),
                            sampled_x=np.asarray(out["samples"], dtype=np.int64).tolist(),
                            optimum_x=optimum_set,
                            decode_success=_to_float(out.get("decoder_success_rate")),
                            postselection_success=_to_float(out.get("success_prob")),
                            retained_energy_eta=_to_float(out.get("energy_fraction")),
                            candidate_source_label=(
                                f"matrix_c|P|{instance_id}|{internal_decoder}|{alpha_mode}|mixture"
                            ),
                            runtime_s=elapsed,
                        )
                        decode_model = _decode_model_for_decoder(internal_decoder)
                        # Qiskit structural counts are decoder-agnostic (they measure
                        # the implemented subset without decode). Enable for all decoders.
                        resources = dqi_resource_estimate(
                            B=B,
                            v=v,
                            ell=ell_eff,
                            decode_uncompute_model=decode_model,
                            include_qiskit_structural_count=True,
                        )
                        internal_results[internal_decoder] = {
                            "status": "completed",
                            "error": None,
                            "error_type": None,
                            "boundary_reason": None,
                            "best_F": _to_float(out.get("best_F")),
                            "best_G": _to_float(out.get("best_G_weighted", out.get("best_G"))),
                            "elapsed_s": elapsed,
                            "peak_memory_mb": _to_float(capture.peak_memory_mb),
                            "decision_metrics": decision_metrics,
                            "decode_success": _to_float(out.get("decoder_success_rate")),
                            "postselection_success": _to_float(out.get("success_prob")),
                            "resources": resources,
                        }
                    except Exception as exc:  # pragma: no cover - defensive payload path
                        boundary_reason, error_type, error_message = classify_boundary_exception(exc)
                        internal_results[internal_decoder] = {
                            "status": "failed",
                            "error": error_message,
                            "error_type": error_type,
                            "boundary_reason": boundary_reason,
                            "resources": None,
                            "elapsed_s": None,
                            "peak_memory_mb": None,
                        }

                for idx, decoder_label in enumerate(decoder_list):
                    run_seed = None if seed is None else int(seed + idx)
                    run_id = _safe_slug(
                        f"matrix_c_P_{instance_id}_wht_truncated_pricing_{decoder_label}_{alpha_mode}_mixture_seed{run_seed}"
                    )
                    internal_decoder = _matrix_c_internal_decoder(decoder_label)
                    if internal_decoder is None:
                        record = _base_record(
                            matrix="matrix_c",
                            run_id=run_id,
                            run_status="unavailable",
                            run_error="decoder_not_implemented_in_repo",
                            instance_id=instance_id,
                            family="P",
                            encoding="wht_truncated_pricing",
                            decoder=decoder_label,
                            alpha_mode=alpha_mode,
                            execution_mode="mixture",
                            seed=run_seed,
                        )
                        record.update(_structure_fields(structure))
                        record["runtime"] = None
                        record["gain_over_bp1"] = None
                        record["gain_over_bruteforce"] = None
                        _attach_metric_availability(record)
                        _validate_record_shape(record)
                        records.append(record)
                        continue

                    src = internal_results.get(internal_decoder, {})
                    record = _base_record(
                        matrix="matrix_c",
                        run_id=run_id,
                        run_status=_canonical_run_status(src.get("status", "unavailable")),
                        run_error=src.get("error"),
                        instance_id=instance_id,
                        family="P",
                        encoding="wht_truncated_pricing",
                        decoder=decoder_label,
                        alpha_mode=alpha_mode,
                        execution_mode="mixture",
                        seed=run_seed,
                    )
                    record["ell"] = int(ell_eff)
                    record["estimated_qubits"] = estimated_qubits
                    record["estimated_memory_gb"] = estimated_memory_gb
                    if record["run_status"] == "completed":
                        record["best_sampled_F"] = _to_float(src.get("best_F"))
                        record["best_sampled_G"] = _to_float(src.get("best_G"))
                        record["runtime_sec"] = _to_float(src.get("elapsed_s"))
                        record["wall_clock_s"] = record["runtime_sec"]
                        record["peak_memory_mb"] = _to_float(src.get("peak_memory_mb"))
                        record["attempted_execution"] = True
                        metric_vals = _metric_fields(src.get("decision_metrics"))
                        for key, value in metric_vals.items():
                            if key == "runtime_sec":
                                if record["runtime_sec"] is None:
                                    record["runtime_sec"] = value
                                continue
                            record[key] = value
                        if record["decode_success"] is None:
                            record["decode_success"] = _to_float(src.get("decode_success"))
                        if record["postselection_success"] is None:
                            record["postselection_success"] = _to_float(src.get("postselection_success"))
                    else:
                        record = build_boundary_record(
                            base_record=record,
                            run_status=str(record["run_status"]),
                            boundary_reason=str(src.get("boundary_reason") or "runtime_exception"),
                            attempted_execution=True,
                            estimated_qubits=estimated_qubits,
                            estimated_memory_gb=estimated_memory_gb,
                            error_type=src.get("error_type"),
                            error_message=src.get("error"),
                            wall_clock_s=src.get("elapsed_s"),
                            peak_memory_mb=src.get("peak_memory_mb"),
                        )
                    record.update(_resource_fields(src.get("resources")))
                    record.update(_structure_fields(structure))
                    record["runtime"] = record.get("runtime_sec")
                    record["decoder_internal_mode"] = internal_decoder
                    if decoder_label == "oracle":
                        record["decoder_alias_of"] = "bruteforce"
                    record["gain_over_bp1"] = None
                    record["gain_over_bruteforce"] = None
                    _attach_metric_availability(record)
                    _validate_record_shape(record)
                    records.append(record)

    # Decoder gain overlays
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (
            str(rec.get("family")),
            str(rec.get("instance_id")),
            str(rec.get("encoding")),
            str(rec.get("alpha_mode")),
            str(rec.get("execution_mode")),
        )
        grouped.setdefault(key, []).append(rec)

    for group_rows in grouped.values():
        f_bp1 = None
        f_bf = None
        for row in group_rows:
            if row.get("run_status") != "completed":
                continue
            if row.get("decoder") == "bp1" and row.get("best_sampled_F") is not None:
                f_bp1 = float(row["best_sampled_F"])
            if row.get("decoder") == "bruteforce" and row.get("best_sampled_F") is not None:
                f_bf = float(row["best_sampled_F"])
        for row in group_rows:
            row["runtime"] = row.get("runtime_sec")
            if row.get("run_status") == "completed" and row.get("best_sampled_F") is not None:
                cur = float(row["best_sampled_F"])
                row["gain_over_bp1"] = None if f_bp1 is None else float(cur - f_bp1)
                row["gain_over_bruteforce"] = None if f_bf is None else float(cur - f_bf)
            else:
                row["gain_over_bp1"] = None
                row["gain_over_bruteforce"] = None

    config = {
        "instance_ids": instance_list if instance_list is not None else "all",
        "alpha_modes": alpha_list,
        "execution_mode": execution_mode,
        "decoders": list(decoder_list),
        "include_p_family": bool(include_p_family),
        "p_features": p_feature_list,
        "n_dqi_samples": int(n_dqi_samples),
        "top_k_by_surrogate": int(top_k_by_surrogate),
        "seed": _to_int(seed),
        "ell": _to_int(ell),
        "smoke": bool(smoke),
    }
    out_paths = _persist_matrix_artifacts(
        matrix_name="matrix_c",
        records=records,
        layout=layout,
        config=config,
        csv_filename="matrix_c_summary.csv",
    )
    report_path = layout["reports"] / "matrix_c_report.json"
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    by_decoder: dict[str, int] = {}
    unavailable_by_metric = {metric: 0 for metric in METRIC_FIELDS}
    for rec in records:
        status = _canonical_run_status(rec.get("run_status"), default="unavailable")
        by_status[status] = by_status.get(status, 0) + 1
        by_family[rec["family"]] = by_family.get(rec["family"], 0) + 1
        by_decoder[rec["decoder"]] = by_decoder.get(rec["decoder"], 0) + 1
        for metric in METRIC_FIELDS:
            if rec.get(metric) is None:
                unavailable_by_metric[metric] += 1

    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "matrix": "matrix_c",
        "generated_at": _now_iso_utc(),
        "config": config,
        "record_count": len(records),
        "by_status": by_status,
        "by_family": by_family,
        "by_decoder": by_decoder,
        "unavailable_by_metric": unavailable_by_metric,
        "required_record_fields": REQUIRED_RECORD_FIELDS,
        "records": records,
    }
    with open(report_path, "w") as f:
        json.dump(report_payload, f, indent=2)
    out_paths["report_json"] = report_path
    out_paths["records"] = records
    return out_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Matrix A/B/C benchmarks with standardized artifact outputs."
    )
    subparsers = parser.add_subparsers(dest="matrix", required=True)

    p_a = subparsers.add_parser("matrix-a", help="Run Matrix A (P family).")
    p_a.add_argument("--output-root", default="results")
    p_a.add_argument("--features", nargs="+", type=int, default=[3, 4, 5])
    p_a.add_argument("--n-dqi-samples", type=int, default=1000)
    p_a.add_argument("--sa-iter", type=int, default=10000)
    p_a.add_argument("--top-k-by-surrogate", type=int, default=10)
    p_a.add_argument("--seed", type=int, default=42)
    p_a.add_argument("--include-stage1-paired-encoding", action="store_true")
    p_a.add_argument("--smoke", action="store_true")

    p_b = subparsers.add_parser("matrix-b", help="Run Matrix B (C family).")
    p_b.add_argument("--output-root", default="results")
    p_b.add_argument("--instance-ids", nargs="*")
    p_b.add_argument("--alpha-modes", default="uniform,paper,heuristic")
    p_b.add_argument("--execution-modes", default="mixture,coherent")
    p_b.add_argument("--decoders", default="oracle,bp1")
    p_b.add_argument("--n-dqi-samples", type=int, default=256)
    p_b.add_argument("--top-k-by-surrogate", type=int, default=10)
    p_b.add_argument("--seed", type=int, default=42)
    p_b.add_argument("--ell", type=int)
    p_b.add_argument("--smoke", action="store_true")

    p_c = subparsers.add_parser("matrix-c", help="Run Matrix C (S family).")
    p_c.add_argument("--output-root", default="results")
    p_c.add_argument("--instance-ids", nargs="*")
    p_c.add_argument("--alpha-modes", default="paper,uniform")
    p_c.add_argument("--execution-mode", default="mixture")
    p_c.add_argument("--decoders", default="bruteforce,bp1,bp+osd-lite,oracle")
    p_c.add_argument("--p-features", default="3,4,5")
    p_c.add_argument("--exclude-p-family", action="store_true")
    p_c.add_argument("--n-dqi-samples", type=int, default=256)
    p_c.add_argument("--top-k-by-surrogate", type=int, default=10)
    p_c.add_argument("--seed", type=int, default=42)
    p_c.add_argument("--ell", type=int)
    p_c.add_argument("--smoke", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.matrix == "matrix-a":
        out = run_matrix_a(
            output_root=args.output_root,
            features=args.features,
            n_dqi_samples=args.n_dqi_samples,
            sa_iter=args.sa_iter,
            top_k_by_surrogate=args.top_k_by_surrogate,
            seed=args.seed,
            include_stage1_paired_encoding=args.include_stage1_paired_encoding,
            smoke=args.smoke,
        )
    elif args.matrix == "matrix-b":
        out = run_matrix_b(
            output_root=args.output_root,
            instance_ids=args.instance_ids,
            alpha_modes=_split_csv(args.alpha_modes),
            execution_modes=_split_csv(args.execution_modes),
            decoders=_split_csv(args.decoders),
            n_dqi_samples=args.n_dqi_samples,
            top_k_by_surrogate=args.top_k_by_surrogate,
            seed=args.seed,
            ell=args.ell,
            smoke=args.smoke,
        )
    elif args.matrix == "matrix-c":
        out = run_matrix_c(
            output_root=args.output_root,
            instance_ids=args.instance_ids,
            alpha_modes=_split_csv(args.alpha_modes),
            execution_mode=args.execution_mode,
            decoders=_split_csv(args.decoders),
            p_features=[int(x) for x in _split_csv(args.p_features)],
            include_p_family=not args.exclude_p_family,
            n_dqi_samples=args.n_dqi_samples,
            top_k_by_surrogate=args.top_k_by_surrogate,
            seed=args.seed,
            ell=args.ell,
            smoke=args.smoke,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown matrix subcommand: {args.matrix}")

    terminal_payload = {
        "matrix": args.matrix,
        "run_count": out["run_count"],
        "matrix_dir": str(out["matrix_dir"]),
        "runs_dir": str(out["runs_dir"]),
        "aggregated_csv": str(out["aggregated_csv"]),
        "manifest": str(out["manifest"]),
    }
    print(json.dumps(terminal_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
