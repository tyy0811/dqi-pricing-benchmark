"""Stage C decoder-study raw runner (append-only)."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.benchmark import load_scaling_instance
from src.dqi_pipeline import DQIPipeline
from src.dqi_state import decoder_diagnostics, run_dqi_statevector
from src.family_s_manifest import load_family_s_family
from src.metrics_decision import compute_decision_metrics
from src.reduction import surrogate_objective
from src.structure_metrics import compute_structure_metrics
from src.weights import (
    heuristic_alpha_weights,
    max_safe_ell,
    paper_alpha_weights,
    uniform_weights,
    validate_dqi_parameters,
)


SCHEMA_VERSION = "1.0"
STAGE_NAME = "decoder_study"
MATRIX_NAME = "matrix_c"


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_slug(raw: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in raw)


def _alpha_vector_for_mode(*, alpha_mode: str, B: np.ndarray, v: np.ndarray, ell: int) -> np.ndarray:
    if alpha_mode == "uniform":
        return uniform_weights(ell)
    if alpha_mode == "paper":
        return paper_alpha_weights(ell)
    if alpha_mode == "heuristic":
        return heuristic_alpha_weights(B, v, ell)
    raise ValueError(f"Unsupported alpha_mode={alpha_mode!r}")


def _metric_availability_for_status(run_status: str, base: dict[str, str] | None) -> dict[str, str]:
    if run_status == "completed":
        return dict(base or {})
    if run_status == "failed":
        return {
            "best_sampled_F": "failed_upstream",
            "best_sampled_G": "failed_upstream",
            "top1_regret": "failed_upstream",
            "topk_regret": "failed_upstream",
            "optimum_recall_at_k": "failed_upstream",
            "best_top_G_sampled_F": "failed_upstream",
            "spearman_rho_F_G": "failed_upstream",
            "retained_energy_eta": "failed_upstream",
            "decision_distortion": "failed_upstream",
            "decode_success": "failed_upstream",
            "postselection_success": "failed_upstream",
            "runtime_sec": "failed_upstream",
        }
    if run_status == "not_applicable":
        return {
            "best_sampled_F": "not_applicable",
            "best_sampled_G": "not_applicable",
            "top1_regret": "not_applicable",
            "topk_regret": "not_applicable",
            "optimum_recall_at_k": "not_applicable",
            "best_top_G_sampled_F": "not_applicable",
            "spearman_rho_F_G": "not_applicable",
            "retained_energy_eta": "not_applicable",
            "decision_distortion": "not_applicable",
            "decode_success": "not_applicable",
            "postselection_success": "not_applicable",
            "runtime_sec": "not_applicable",
        }
    return {
        "best_sampled_F": "not_applicable",
        "best_sampled_G": "not_applicable",
        "top1_regret": "not_applicable",
        "topk_regret": "not_applicable",
        "optimum_recall_at_k": "not_applicable",
        "best_top_G_sampled_F": "not_applicable",
        "spearman_rho_F_G": "not_applicable",
        "retained_energy_eta": "not_applicable",
        "decision_distortion": "not_applicable",
        "decode_success": "not_applicable",
        "postselection_success": "not_applicable",
        "runtime_sec": "not_applicable",
    }


def _row_template(
    *,
    family: str,
    instance_id: str,
    encoding: str,
    decoder: str,
    alpha_mode: str,
    trial_seed: int | None,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix": MATRIX_NAME,
        "stage": STAGE_NAME,
        "run_id": run_id,
        "run_timestamp_utc": _now_iso_utc(),
        "family": family,
        "instance_id": instance_id,
        "encoding": encoding,
        "decoder": decoder,
        "alpha_mode": alpha_mode,
        "execution_mode": "mixture",
        "trial_seed": trial_seed,
        "seed": trial_seed,
        "run_status": "completed",
        "status_reason_code": "completed",
        "status_reason": "completed",
        "decoder_status": "completed",
        "decoder_error": None,
        "in_experiment_matrix": True,
        "best_F": None,
        "approximation_ratio": None,
        "decode_success": None,
        "postselection_success": None,
        "topk_regret": None,
        "runtime_sec": None,
        "best_sampled_F": None,
        "best_sampled_G": None,
        "top1_regret": None,
        "optimum_recall_at_k": None,
        "best_top_G_sampled_F": None,
        "spearman_rho_F_G": None,
        "retained_energy_eta": None,
        "decision_distortion": None,
        "metric_availability": {},
    }


def _apply_structure_fields(row: dict[str, Any], structure: dict[str, Any]) -> None:
    for key in [
        "structure_collision_rate",
        "structure_ambiguous_syndrome_count",
        "structure_short_cycle_proxy",
        "structure_distance_proxy",
        "structure_row_weight_min",
        "structure_row_weight_mean",
        "structure_row_weight_max",
        "structure_col_weight_min",
        "structure_col_weight_mean",
        "structure_col_weight_max",
        "structure_rank",
        "structure_rank_deficiency",
    ]:
        row[key] = structure.get(key)


def _mark_not_applicable(
    row: dict[str, Any],
    *,
    reason_code: str,
    reason: str,
    decoder_status: str,
) -> None:
    row["run_status"] = "not_applicable"
    row["status_reason_code"] = reason_code
    row["status_reason"] = reason
    row["decoder_status"] = decoder_status
    row["metric_availability"] = _metric_availability_for_status("not_applicable", None)


def _mark_failed(row: dict[str, Any], exc: Exception) -> None:
    row["run_status"] = "failed"
    row["status_reason_code"] = "decoder_runtime_error"
    row["status_reason"] = "decoder runtime failure"
    row["decoder_status"] = "failed"
    row["decoder_error"] = str(exc)
    row["metric_availability"] = _metric_availability_for_status("failed", None)


def _attach_decision_metrics(
    row: dict[str, Any],
    *,
    decision_metrics: dict[str, Any],
    best_f: float,
    approximation_ratio: float | None,
    decode_success: float | None,
    postselection_success: float | None,
    runtime_sec: float,
) -> None:
    core = (decision_metrics.get("core_decision") or {})
    faith = (decision_metrics.get("faithfulness") or {})
    row["best_F"] = float(best_f)
    row["approximation_ratio"] = None if approximation_ratio is None else float(approximation_ratio)
    row["decode_success"] = None if decode_success is None else float(decode_success)
    row["postselection_success"] = None if postselection_success is None else float(postselection_success)
    row["topk_regret"] = core.get("topk_regret")
    row["runtime_sec"] = float(runtime_sec)
    row["best_sampled_F"] = core.get("best_sampled_F")
    row["best_sampled_G"] = core.get("best_sampled_G")
    row["top1_regret"] = core.get("top1_regret")
    row["optimum_recall_at_k"] = core.get("optimum_recall_at_k")
    row["best_top_G_sampled_F"] = core.get("best_top_G_sampled_F")
    row["spearman_rho_F_G"] = faith.get("spearman_rho_F_G")
    row["retained_energy_eta"] = faith.get("retained_energy_eta")
    row["decision_distortion"] = core.get("decision_distortion")
    row["metric_availability"] = _metric_availability_for_status(
        "completed", decision_metrics.get("metric_availability") or {}
    )


def _run_single_p_decoder(
    *,
    instance_id: str,
    n_features: int,
    alpha_mode: str,
    decoder: str,
    n_dqi_samples: int,
    top_k_by_surrogate: int,
    trial_seed: int | None,
    oracle_max_m: int,
) -> dict[str, Any]:
    prob, _from_frozen, _path = load_scaling_instance(n_features)
    k = 15 if n_features == 5 else min(12, 2 ** (2 * n_features) - 1)
    ell = min(3, max_safe_ell(k, prob.n_bits))
    ok, message = validate_dqi_parameters(k, prob.n_bits, ell)
    if not ok:
        raise ValueError(f"invalid decoder-study parameters for {instance_id}: {message}")

    run_id = _safe_slug(
        f"{MATRIX_NAME}_{STAGE_NAME}_{instance_id}_wht_truncated_pricing_{decoder}_{alpha_mode}_seed{trial_seed}_{uuid.uuid4().hex[:8]}"
    )
    row = _row_template(
        family="P",
        instance_id=instance_id,
        encoding="wht_truncated_pricing",
        decoder=decoder,
        alpha_mode=alpha_mode,
        trial_seed=trial_seed,
        run_id=run_id,
    )

    base_pipe = DQIPipeline(
        prob,
        k=k,
        ell=ell,
        alpha_mode=alpha_mode,
        execution_mode="mixture",
        decoder_mode="bruteforce",
    )
    structure = compute_structure_metrics(np.asarray(base_pipe.B, dtype=np.int8), ell)
    _apply_structure_fields(row, structure)

    if decoder == "oracle" and int(base_pipe.B.shape[0]) > int(oracle_max_m):
        _mark_not_applicable(
            row,
            reason_code="oracle_tiny_cap_exceeded",
            reason="oracle tiny-cap exceeded for this instance",
            decoder_status="skipped_tiny_only",
        )
        return row

    t0 = time.perf_counter()
    try:
        out = DQIPipeline(
            prob,
            k=k,
            ell=ell,
            alpha_mode=alpha_mode,
            execution_mode="mixture",
            decoder_mode=decoder,
        ).run_with_metrics(n_samples=int(n_dqi_samples), seed=trial_seed)
        elapsed = float(time.perf_counter() - t0)
        x_opt, f_opt = prob.brute_force_solve()
        decision = compute_decision_metrics(
            F_values=np.asarray(out["F_values"], dtype=np.float64).tolist(),
            G_values=np.asarray(out["G_weighted_values"], dtype=np.float64).tolist(),
            f_opt=float(f_opt),
            k_eval=int(top_k_by_surrogate),
            sampled_x=np.asarray(out["samples"], dtype=np.int64).tolist(),
            optimum_x=[int(x_opt)],
            decode_success=out.get("decoder_success_rate"),
            postselection_success=out.get("success_prob"),
            retained_energy_eta=out.get("energy_fraction"),
            candidate_source_label=f"{STAGE_NAME}|P|{instance_id}|{decoder}|{alpha_mode}",
            runtime_s=elapsed,
        )
        _attach_decision_metrics(
            row,
            decision_metrics=decision,
            best_f=float(out["best_F"]),
            approximation_ratio=out.get("approximation_ratio"),
            decode_success=out.get("decoder_success_rate"),
            postselection_success=out.get("success_prob"),
            runtime_sec=elapsed,
        )
    except Exception as exc:  # pragma: no cover - defensive runner behavior
        _mark_failed(row, exc)
    return row


def _run_single_s_decoder(
    *,
    inst: dict[str, Any],
    alpha_mode: str,
    decoder: str,
    n_dqi_samples: int,
    top_k_by_surrogate: int,
    trial_seed: int | None,
    oracle_max_m: int,
) -> dict[str, Any]:
    instance_id = str(inst["instance_id"])
    B = np.asarray(inst["B"], dtype=np.int8)
    v = np.asarray(inst["v"], dtype=np.int8)
    n_bits = int(inst["n_bits"])
    m_rows = int(inst["m_rows"])
    ell = int(inst["ell_default"])
    alpha = _alpha_vector_for_mode(alpha_mode=alpha_mode, B=B, v=v, ell=ell)
    run_id = _safe_slug(
        f"{MATRIX_NAME}_{STAGE_NAME}_{instance_id}_synthetic_structured_bv_{decoder}_{alpha_mode}_seed{trial_seed}_{uuid.uuid4().hex[:8]}"
    )
    row = _row_template(
        family="S",
        instance_id=instance_id,
        encoding="synthetic_structured_bv",
        decoder=decoder,
        alpha_mode=alpha_mode,
        trial_seed=trial_seed,
        run_id=run_id,
    )
    structure = compute_structure_metrics(B, ell)
    _apply_structure_fields(row, structure)

    if decoder == "oracle" and m_rows > int(oracle_max_m):
        _mark_not_applicable(
            row,
            reason_code="oracle_tiny_cap_exceeded",
            reason="oracle tiny-cap exceeded for this instance",
            decoder_status="skipped_tiny_only",
        )
        return row

    unit_weights = np.ones(m_rows, dtype=np.float64)
    truth_f = np.array([surrogate_objective(x, B, v, unit_weights, n_bits) for x in range(2 ** n_bits)], dtype=np.float64)
    f_opt = float(np.max(truth_f))
    optimum_set = np.flatnonzero(np.isclose(truth_f, f_opt)).astype(np.int64).tolist()
    diagnostics = decoder_diagnostics(B=B, ell=ell, decoder_mode=decoder)

    # Pre-check: Skip DQI execution if structurally impossible (zero decodable syndromes)
    decoder_success_rate = diagnostics.get("decoder_success_rate", 0.0)
    if decoder_success_rate == 0.0:
        # Mark as not_applicable instead of failed - this is a structural boundary
        row["run_status"] = "not_applicable"
        row["status_reason_code"] = "dqi_impossible_zero_decodable_syndromes"
        row["run_error"] = (
            f"DQI execution skipped: instance has zero decodable syndromes at ell={ell}. "
            f"This is a structural property (decoder_success_rate=0.0), not a decoder failure."
        )
        row["metric_availability"] = _metric_availability_for_status("not_applicable", None)
        return row

    t0 = time.perf_counter()
    try:
        samples, probabilities, success_prob = run_dqi_statevector(
            B=B,
            v=v,
            alpha=alpha,
            ell=ell,
            n_samples=int(n_dqi_samples),
            seed=trial_seed,
            decoder_mode=decoder,
            execution_mode="mixture",
        )
        elapsed = float(time.perf_counter() - t0)
        sample_arr = np.asarray(samples, dtype=np.int64)
        f_vals = truth_f[sample_arr]
        g_vals = f_vals.copy()
        decision = compute_decision_metrics(
            F_values=f_vals.tolist(),
            G_values=g_vals.tolist(),
            f_opt=f_opt,
            k_eval=int(top_k_by_surrogate),
            sampled_x=sample_arr.tolist(),
            optimum_x=optimum_set,
            decode_success=diagnostics.get("decoder_success_rate"),
            postselection_success=float(success_prob),
            retained_energy_eta=None,
            candidate_source_label=f"{STAGE_NAME}|S|{instance_id}|{decoder}|{alpha_mode}",
            runtime_s=elapsed,
        )
        best_f = float(np.max(f_vals))
        approx = float(best_f / f_opt) if abs(f_opt) > 1e-15 else None
        _attach_decision_metrics(
            row,
            decision_metrics=decision,
            best_f=best_f,
            approximation_ratio=approx,
            decode_success=diagnostics.get("decoder_success_rate"),
            postselection_success=float(success_prob),
            runtime_sec=elapsed,
        )
        row["best_sampled_G"] = float(np.max(np.asarray(probabilities, dtype=np.float64))) if len(probabilities) else row["best_sampled_G"]
    except Exception as exc:  # pragma: no cover - defensive runner behavior
        _mark_failed(row, exc)
    return row


def run_decoder_study(
    *,
    output_root: str | Path = "results",
    p_instances: Iterable[str] = ("P3", "P4", "P5", "P6", "P7"),
    s_instances: Iterable[str] | None = None,
    alpha_modes: Iterable[str] = ("paper", "uniform"),
    decoders: Iterable[str] = ("bruteforce", "bp1", "bp_osd_lite", "oracle"),
    n_dqi_samples: int = 128,
    top_k_by_surrogate: int = 10,
    trial_seed: int | None = 42,
    oracle_max_m: int = 10,
) -> dict[str, Any]:
    """Append Stage C raw decoder-study rows to ``results/matrix_c/runs``."""
    output_root = Path(output_root)
    runs_dir = output_root / "matrix_c" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    p_ids = [str(x) for x in p_instances]
    s_ids = None if s_instances is None else [str(x) for x in s_instances]
    alpha_list = [str(x) for x in alpha_modes]
    decoder_list = [str(x) for x in decoders]

    for instance_id in p_ids:
        if not instance_id.startswith("P") or not instance_id[1:].isdigit():
            raise ValueError(f"invalid P-family instance id: {instance_id!r}")
        n_features = int(instance_id[1:])
        for alpha_mode in alpha_list:
            for decoder in decoder_list:
                rows.append(
                    _run_single_p_decoder(
                        instance_id=instance_id,
                        n_features=n_features,
                        alpha_mode=alpha_mode,
                        decoder=decoder,
                        n_dqi_samples=int(n_dqi_samples),
                        top_k_by_surrogate=int(top_k_by_surrogate),
                        trial_seed=trial_seed,
                        oracle_max_m=int(oracle_max_m),
                    )
                )

    for inst in load_family_s_family(instance_ids=s_ids):
        for alpha_mode in alpha_list:
            for decoder in decoder_list:
                rows.append(
                    _run_single_s_decoder(
                        inst=inst,
                        alpha_mode=alpha_mode,
                        decoder=decoder,
                        n_dqi_samples=int(n_dqi_samples),
                        top_k_by_surrogate=int(top_k_by_surrogate),
                        trial_seed=trial_seed,
                        oracle_max_m=int(oracle_max_m),
                    )
                )

    out_files: list[str] = []
    for row in rows:
        path = runs_dir / f"{row['run_id']}.json"
        path.write_text(json.dumps(row, indent=2))
        out_files.append(str(path))

    return {
        "output_root": str(output_root),
        "runs_dir": str(runs_dir),
        "row_count": len(rows),
        "run_files": out_files,
    }
