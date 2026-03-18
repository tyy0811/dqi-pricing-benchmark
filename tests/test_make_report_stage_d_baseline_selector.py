"""Stage D fixed-priority baseline selector contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, ".")

from scripts.make_report import run_make_report


def _write_matrix_b_inputs(
    results_root: Path,
    *,
    wht_pairs: int,
    ilp_pairs: int,
    duplicate_completed_counterpart: bool = False,
    include_distributional_fields: bool = False,
) -> None:
    (results_root / "matrix_b").mkdir(parents=True, exist_ok=True)
    (results_root / "matrix_b" / "runs").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "schema_version": "1.0",
        "matrix": "matrix_b",
        "generated_at": "2026-03-13T00:00:00+00:00",
        "config": {
            "instance_ids": ["C1"],
            "alpha_modes": ["paper"],
            "execution_modes": ["mixture", "coherent"],
            "decoders": ["oracle"],
            "seed": 42,
        },
        "aggregated_csv": str(results_root / "tables" / "matrix_b_summary.csv"),
        "run_count": 0,
        "run_files": [],
    }
    (results_root / "matrix_b" / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    # Legacy compatibility artifacts are required by report build, but Stage D
    # tests operate on authoritative matrix_b/runs rows.
    pd.DataFrame(columns=["run_id"]).to_csv(results_root / "tables" / "matrix_b_summary.csv", index=False)
    (results_root / "reports" / "matrix_b_report.json").write_text(
        json.dumps({"matrix": "matrix_b", "record_count": 0}, indent=2)
    )

    def _rows_for_encoding(encoding: str, n_pairs: int) -> list[dict]:
        out: list[dict] = []
        for idx in range(n_pairs):
            alpha_mode = f"a{idx % 2}"
            seed = 100 + idx
            for execution_mode, best_f in [("mixture", 1.0 + idx), ("coherent", 2.0 + idx)]:
                out.append(
                    {
                        "matrix": "matrix_b",
                        "stage": "alpha_validation",
                        "run_id": f"{encoding}_{execution_mode}_{idx}",
                        "run_timestamp_utc": "2026-03-13T10:00:00+00:00",
                        "run_status": "completed",
                        "status_reason_code": "completed",
                        "status_reason": "completed",
                        "in_experiment_matrix": True,
                        "run_error": None,
                        "family": "C",
                        "instance_id": "C1",
                        "encoding": encoding,
                        "decoder": "oracle",
                        "alpha_mode": alpha_mode,
                        "execution_mode": execution_mode,
                        "ell": 1,
                        "m_active": 4,
                        "trial_seed": seed,
                        "canonical_trial_seed": seed,
                        "seed": seed,
                        "oracle_exact": True,
                        "coherent_supported": True,
                        "best_F": float(best_f),
                        "best_top_G_sampled_F": float(best_f),
                        "postselection_success": 0.75,
                        "success_probability": 0.75,
                        "candidate_probabilities": [0.5, 0.5],
                        "candidate_entropy": None,
                        "candidate_top1_probability": None,
                        "candidate_top2_probability": None,
                        "metric_availability": {"best_F": "available", "success_probability": "available"},
                    }
                )
                if include_distributional_fields:
                    out[-1]["candidate_entropy"] = 0.5 + (idx % 3) * 0.1
                    out[-1]["candidate_top1_probability"] = 0.5 + (idx % 3) * 0.05
                    out[-1]["candidate_top2_probability"] = 0.5 - (idx % 3) * 0.02
        return out

    stage_rows = []
    stage_rows.extend(_rows_for_encoding("wht_exact", wht_pairs))
    stage_rows.extend(_rows_for_encoding("ilp_derived_exact", ilp_pairs))

    if duplicate_completed_counterpart:
        stage_rows.append(
            {
                "matrix": "matrix_b",
                "stage": "alpha_validation",
                "run_id": "dup_wht_coherent",
                "run_timestamp_utc": "2026-03-13T10:00:01+00:00",
                "run_status": "completed",
                "status_reason_code": "completed",
                "status_reason": "completed",
                "in_experiment_matrix": True,
                "run_error": None,
                "family": "C",
                "instance_id": "C1",
                "encoding": "wht_exact",
                "decoder": "oracle",
                "alpha_mode": "a0",
                "execution_mode": "coherent",
                "ell": 1,
                "m_active": 4,
                "trial_seed": 100,
                "canonical_trial_seed": 100,
                "seed": 100,
                "oracle_exact": True,
                "coherent_supported": True,
                "best_F": 9.0,
                "best_top_G_sampled_F": 9.0,
                "postselection_success": 0.99,
                "success_probability": 0.99,
                "candidate_probabilities": [0.8, 0.2],
                "metric_availability": {"best_F": "available", "success_probability": "available"},
            }
        )

    for row in stage_rows:
        (results_root / "matrix_b" / "runs" / f"{row['run_id']}.json").write_text(json.dumps(row, indent=2))


def _load_stage_d_metrics(results_root: Path) -> pd.DataFrame:
    path = results_root / "metrics_master.parquet"
    if path.exists():
        metrics = pd.read_parquet(path)
    else:
        metrics = pd.DataFrame(json.loads((results_root / "metrics_master.json").read_text())["rows"])
    return metrics[
        (metrics["matrix"] == "matrix_b")
        & (metrics["stage"] == "alpha_validation")
        & (metrics["family"] == "C")
    ].copy()


def test_selector_prefers_wht_exact_oracle_when_threshold_met(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_b_inputs(results_root, wht_pairs=8, ilp_pairs=8)

    run_make_report(results_root=results_root, matrices=["matrix_b"])
    stage_d = _load_stage_d_metrics(results_root)
    assert not stage_d.empty

    required_cols = [
        "baseline_selector_policy",
        "baseline_min_pairs",
        "baseline_encoding_norm",
        "baseline_decoder_norm",
        "baseline_pair_count",
        "baseline_priority_rank",
        "baseline_selected_reason",
        "baseline_fallback_used",
        "best_available_pair_count",
        "is_stage_d_selected_baseline",
    ]
    for col in required_cols:
        assert col in stage_d.columns

    assert stage_d["baseline_selector_policy"].eq("fixed_priority").all()
    assert stage_d["baseline_encoding_norm"].eq("wht_exact").all()
    assert stage_d["baseline_decoder_norm"].eq("oracle").all()
    assert stage_d["baseline_pair_count"].eq(8).all()
    assert stage_d["baseline_priority_rank"].eq(1).all()
    assert stage_d["baseline_selected_reason"].eq("preferred_available_meets_threshold").all()
    assert stage_d["baseline_fallback_used"].eq(False).all()
    assert stage_d["best_available_pair_count"].eq(8).all()
    selected = stage_d[stage_d["is_stage_d_selected_baseline"] == True]
    assert selected["encoding"].eq("wht_exact").all()
    assert selected["decoder"].eq("oracle").all()


def test_selector_uses_fallback_when_preferred_below_threshold(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_b_inputs(results_root, wht_pairs=7, ilp_pairs=8)

    run_make_report(results_root=results_root, matrices=["matrix_b"])
    stage_d = _load_stage_d_metrics(results_root)
    assert not stage_d.empty
    assert stage_d["baseline_encoding_norm"].eq("ilp_derived_exact").all()
    assert stage_d["baseline_decoder_norm"].eq("oracle").all()
    assert stage_d["baseline_pair_count"].eq(8).all()
    assert stage_d["baseline_priority_rank"].eq(2).all()
    assert stage_d["baseline_selected_reason"].eq("preferred_below_threshold_used_fallback").all()
    assert stage_d["baseline_fallback_used"].eq(True).all()
    selected = stage_d[stage_d["is_stage_d_selected_baseline"] == True]
    assert selected["encoding"].eq("ilp_derived_exact").all()
    assert selected["decoder"].eq("oracle").all()


def test_selector_emits_explicit_no_baseline_state_when_threshold_not_met(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_b_inputs(results_root, wht_pairs=7, ilp_pairs=7)

    run_make_report(results_root=results_root, matrices=["matrix_b"])
    stage_d = _load_stage_d_metrics(results_root)
    assert not stage_d.empty

    assert stage_d["baseline_selector_policy"].eq("fixed_priority").all()
    assert stage_d["baseline_min_pairs"].eq(8).all()
    assert stage_d["baseline_selected_reason"].eq("no_baseline_meets_threshold").all()
    assert stage_d["baseline_fallback_used"].eq(False).all()
    assert stage_d["best_available_pair_count"].eq(7).all()
    assert stage_d["is_stage_d_selected_baseline"].eq(False).all()
    assert stage_d["baseline_encoding_norm"].isna().all()
    assert stage_d["baseline_decoder_norm"].isna().all()
    assert stage_d["baseline_priority_rank"].isna().all()
    assert stage_d["baseline_pair_count"].isna().all()


def test_duplicate_completed_counterpart_rows_fail_before_selection(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_b_inputs(
        results_root,
        wht_pairs=8,
        ilp_pairs=0,
        duplicate_completed_counterpart=True,
    )

    with pytest.raises(ValueError, match="duplicate completed counterpart"):
        run_make_report(results_root=results_root, matrices=["matrix_b"])


def test_stage_d_distributional_metric_fields_propagate_to_metrics_master_and_keys_are_unique(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_b_inputs(results_root, wht_pairs=15, ilp_pairs=0, include_distributional_fields=True)

    run_make_report(results_root=results_root, matrices=["matrix_b"])
    stage_d = _load_stage_d_metrics(results_root)
    assert len(stage_d) == 30

    for col in [
        "counterpart_key",
        "execution_mode_norm",
        "candidate_entropy",
        "candidate_top1_probability",
        "candidate_top2_probability",
    ]:
        assert col in stage_d.columns

    assert stage_d["candidate_entropy"].notna().all()
    assert stage_d["candidate_top1_probability"].notna().all()
    assert stage_d["candidate_top2_probability"].notna().all()

    dup_counts = stage_d.groupby(["counterpart_key", "execution_mode_norm"], dropna=False).size()
    assert (dup_counts <= 1).all()
