"""Stage D alpha validation runner (Family C exact tiny reference)."""

from __future__ import annotations

import json
from math import log2
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.coherent_reference import (
    load_coherent_family,
    supports_tiny_coherent,
    evaluate_coherent_exact,
    evaluate_mixture_exact,
)
from src.report_schema import validate_alpha_validation_row


SCHEMA_VERSION = "1.0"
MATRIX_NAME = "matrix_b"
STAGE_NAME = "alpha_validation"
FAMILY_NAME = "C"
_REVISED_FAMILY_C_GRID: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "C1": ((1, 2), (4,)),
    "C2": ((1, 2), (6,)),
    "C3": ((2,), (4,)),
}


def _default_stage_d_slot_grid(*, instance_id: str, meta: dict[str, Any]) -> tuple[list[int], list[int]]:
    if instance_id in _REVISED_FAMILY_C_GRID:
        ell_list, m_list = _REVISED_FAMILY_C_GRID[instance_id]
        return list(map(int, ell_list)), list(map(int, m_list))
    return [int(meta["ell_default"])], [int(meta["m_rows"])]


def _select_slot_grid(
    *,
    instance_id: str,
    meta: dict[str, Any],
    rec: dict[str, Any],
    ell_values: Iterable[int] | None,
    m_active_values: Iterable[int] | None,
) -> list[tuple[int, int]]:
    ell_list = [int(x) for x in (ell_values if ell_values is not None else [])]
    m_list = [int(x) for x in (m_active_values if m_active_values is not None else [])]
    if not ell_list and not m_list:
        ell_list, m_list = _default_stage_d_slot_grid(instance_id=instance_id, meta=meta)
        slots: list[tuple[int, int]] = []
        for ell in ell_list:
            for m_active in m_list:
                supported, _reason = supports_tiny_coherent(rec, ell=ell, m_active=m_active)
                if supported:
                    slots.append((ell, m_active))
        return slots

    if not ell_list:
        ell_list = [int(meta["ell_default"])]
    if not m_list:
        m_list = [int(meta["m_rows"])]
    return [(int(ell), int(m_active)) for ell in ell_list for m_active in m_list]


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_slug(raw: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in raw)


def _candidate_distribution_summary(
    candidate_probabilities: list[float] | None,
) -> dict[str, float | None]:
    if not isinstance(candidate_probabilities, list) or not candidate_probabilities:
        return {
            "candidate_entropy": None,
            "candidate_top1_probability": None,
            "candidate_top2_probability": None,
        }
    probs = [float(x) for x in candidate_probabilities]
    total = float(sum(probs))
    if total <= 0.0 or not all(isinstance(x, (int, float)) for x in candidate_probabilities):
        return {
            "candidate_entropy": None,
            "candidate_top1_probability": None,
            "candidate_top2_probability": None,
        }
    normalized = [x / total for x in probs]
    nz = [x for x in normalized if x > 0.0]
    if nz:
        entropy = -sum(x * log2(x) for x in nz)
    else:
        entropy = 0.0
    sorted_probs = sorted(normalized, reverse=True)
    return {
        "candidate_entropy": float(entropy),
        "candidate_top1_probability": float(sorted_probs[0]),
        "candidate_top2_probability": float(sum(sorted_probs[:2])),
    }


def _normalize_seed(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _encoding_label(meta: dict[str, Any]) -> str:
    source_type = str(meta.get("source_type", ""))
    if source_type == "pricing_ilp_exact" or meta.get("exact_encoding_source_path"):
        return "ilp_derived_exact"
    if bool(meta.get("wht_exact_available", False)):
        return "wht_exact"
    return str(meta.get("encoding_backend", "coherent_exact"))


def _metric_availability_for_status(run_status: str) -> dict[str, str]:
    if run_status == "completed":
        token = "available"
    elif run_status == "not_applicable":
        token = "not_applicable"
    else:
        token = "failed_upstream"
    return {
        "success_probability": token,
        "best_F": token,
        "best_sampled_F": token,
        "best_top_G_sampled_F": token,
        "postselection_success": token,
        "F_star": token,
        "top1_regret": token,
    }


def _build_row(
    *,
    run_id: str,
    instance_id: str,
    encoding: str,
    alpha_mode: str,
    execution_mode: str,
    ell: int,
    m_active: int,
    trial_seed: int | None,
    source_artifact: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_timestamp_utc": _now_iso_utc(),
        "matrix": MATRIX_NAME,
        "stage": STAGE_NAME,
        "family": FAMILY_NAME,
        "instance_id": instance_id,
        "encoding": encoding,
        "decoder": "oracle",
        "alpha_mode": alpha_mode,
        "execution_mode": execution_mode,
        "ell": int(ell),
        "m_active": int(m_active),
        "trial_seed": trial_seed,
        "canonical_trial_seed": trial_seed,
        "seed": trial_seed,
        "run_status": "failed",
        "status_reason_code": "execution_failed",
        "status_reason": "execution_failed",
        "in_experiment_matrix": True,
        "run_error": None,
        "coherent_supported": False,
        "oracle_exact": True,
        "success_probability": None,
        "best_F": None,
        "best_sampled_F": None,
        "best_top_G_sampled_F": None,
        "F_star": None,
        "top1_regret": None,
        "postselection_success": None,
        "candidate_probabilities": None,
        "candidate_entropy": None,
        "candidate_top1_probability": None,
        "candidate_top2_probability": None,
        "source_artifact": source_artifact,
        "coherent_mode_record": execution_mode == "coherent",
        "metric_availability": _metric_availability_for_status("failed"),
    }


def _write_run_row(*, runs_dir: Path, row: dict[str, Any]) -> Path:
    path = runs_dir / f"{row['run_id']}.json"
    path.write_text(json.dumps(row, indent=2))
    return path


def run_alpha_validation(
    *,
    output_root: str | Path = "results",
    instance_ids: Iterable[str] | None = None,
    alpha_modes: Iterable[str] = ("uniform", "paper", "heuristic"),
    execution_modes: Iterable[str] = ("coherent", "mixture"),
    ell_values: Iterable[int] | None = None,
    m_active_values: Iterable[int] | None = None,
    trial_seed: int | None = 42,
    n_dqi_samples: int = 256,
) -> dict[str, Any]:
    """Run Stage D strict planned-grid alpha validation and emit append-only JSON rows."""
    out_root = Path(output_root)
    runs_dir = out_root / MATRIX_NAME / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    family = load_coherent_family()
    by_id = {rec["meta"]["instance_id"]: rec for rec in family}
    if instance_ids is None:
        target_ids = [rec["meta"]["instance_id"] for rec in family]
    else:
        target_ids = [str(x) for x in instance_ids]
        unknown = sorted(set(target_ids).difference(by_id.keys()))
        if unknown:
            raise KeyError(f"unknown coherent instance ids: {unknown}")

    alpha_list = [str(x) for x in alpha_modes]
    exec_list = [str(x) for x in execution_modes]
    if not alpha_list:
        raise ValueError("alpha_modes must not be empty")
    if not exec_list:
        raise ValueError("execution_modes must not be empty")

    seed_norm = _normalize_seed(trial_seed)
    run_files: list[str] = []
    planned_count = 0

    for instance_id in target_ids:
        rec = by_id[instance_id]
        meta = rec["meta"]
        m_rows = int(meta["m_rows"])
        encoding = _encoding_label(meta)
        source_artifact = str(rec["json_path"])

        slot_specs = _select_slot_grid(
            instance_id=instance_id,
            meta=meta,
            rec=rec,
            ell_values=ell_values,
            m_active_values=m_active_values,
        )
        if not slot_specs:
            raise ValueError(f"no supported Stage D slots for instance {instance_id!r}")
        for alpha_mode in alpha_list:
            for execution_mode in exec_list:
                for ell, m_active in slot_specs:
                    planned_count += 1
                    run_id = _safe_slug(
                        f"{MATRIX_NAME}_{STAGE_NAME}_{instance_id}_{encoding}_oracle_"
                        f"{alpha_mode}_{execution_mode}_ell{ell}_m{m_active}_seed{seed_norm}_{uuid.uuid4().hex[:8]}"
                    )
                    row = _build_row(
                        run_id=run_id,
                        instance_id=instance_id,
                        encoding=encoding,
                        alpha_mode=alpha_mode,
                        execution_mode=execution_mode,
                        ell=ell,
                        m_active=m_active,
                        trial_seed=seed_norm,
                        source_artifact=source_artifact,
                    )

                    # Grid rows are emitted even when outside tiny-evaluable bounds.
                    if int(m_active) < 1 or int(m_active) > int(m_rows):
                        row["run_status"] = "not_applicable"
                        row["status_reason_code"] = "m_active_out_of_bounds"
                        row["status_reason"] = "m_active_out_of_bounds"
                        row["metric_availability"] = _metric_availability_for_status("not_applicable")
                        validate_alpha_validation_row(row)
                        run_files.append(str(_write_run_row(runs_dir=runs_dir, row=row)))
                        continue
                    if int(ell) > int(m_active):
                        row["run_status"] = "not_applicable"
                        row["status_reason_code"] = "ell_exceeds_m_active"
                        row["status_reason"] = "ell_exceeds_m_active"
                        row["metric_availability"] = _metric_availability_for_status("not_applicable")
                        validate_alpha_validation_row(row)
                        run_files.append(str(_write_run_row(runs_dir=runs_dir, row=row)))
                        continue

                    if execution_mode == "coherent":
                        supported, reason = supports_tiny_coherent(rec, ell=ell, m_active=m_active)
                        row["coherent_supported"] = bool(supported)
                        if not supported:
                            row["run_status"] = "not_applicable"
                            row["status_reason_code"] = str(reason)
                            row["status_reason"] = str(reason)
                            row["metric_availability"] = _metric_availability_for_status("not_applicable")
                            validate_alpha_validation_row(row)
                            run_files.append(str(_write_run_row(runs_dir=runs_dir, row=row)))
                            continue
                    else:
                        supported, _reason = supports_tiny_coherent(rec, ell=ell, m_active=m_active)
                        row["coherent_supported"] = bool(supported)

                    try:
                        if execution_mode == "coherent":
                            out = evaluate_coherent_exact(
                                rec,
                                alpha_mode=alpha_mode,
                                ell=ell,
                                m_active=m_active,
                                trial_seed=seed_norm,
                                n_dqi_samples=n_dqi_samples,
                            )
                        elif execution_mode == "mixture":
                            out = evaluate_mixture_exact(
                                rec,
                                alpha_mode=alpha_mode,
                                ell=ell,
                                m_active=m_active,
                                trial_seed=seed_norm,
                                n_dqi_samples=n_dqi_samples,
                            )
                        else:
                            raise ValueError(f"unsupported execution_mode={execution_mode!r}")
                        row["run_status"] = "completed"
                        row["status_reason_code"] = "completed"
                        row["status_reason"] = "completed"
                        row["run_error"] = None
                        row["success_probability"] = float(out["success_probability"])
                        row["best_F"] = float(out["best_F"]) if out["best_F"] is not None else None
                        row["best_sampled_F"] = float(out["best_sampled_F"]) if out.get("best_sampled_F") is not None else None
                        row["best_top_G_sampled_F"] = (
                            float(out["best_top_G_sampled_F"])
                            if out["best_top_G_sampled_F"] is not None
                            else None
                        )
                        row["F_star"] = float(out["F_star"]) if out.get("F_star") is not None else None
                        row["top1_regret"] = float(out["top1_regret"]) if out.get("top1_regret") is not None else None
                        row["postselection_success"] = float(out["postselection_success"])
                        row["candidate_probabilities"] = out.get("candidate_probabilities")
                        row.update(
                            _candidate_distribution_summary(
                                candidate_probabilities=row["candidate_probabilities"]
                            )
                        )
                        row["metric_availability"] = _metric_availability_for_status("completed")
                    except Exception as exc:  # pragma: no cover - defensive path
                        row["run_status"] = "failed"
                        row["status_reason_code"] = "execution_failed"
                        row["status_reason"] = "execution_failed"
                        row["run_error"] = str(exc)
                        row["metric_availability"] = _metric_availability_for_status("failed")

                    validate_alpha_validation_row(row)
                    run_files.append(str(_write_run_row(runs_dir=runs_dir, row=row)))

    return {
        "output_root": str(out_root),
        "runs_dir": str(runs_dir),
        "planned_count": int(planned_count),
        "written_count": int(len(run_files)),
        "run_files": run_files,
    }
