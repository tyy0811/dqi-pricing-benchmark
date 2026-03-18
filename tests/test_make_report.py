"""Tests for Stage B report compiler."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, ".")

from scripts.make_report import (
    _resolve_duplicates,
    build_slot_id,
    normalize_seed_for_hash,
    run_make_report,
)


def _write_minimal_matrix_a_inputs(results_root: Path) -> None:
    (results_root / "matrix_a").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "schema_version": "1.0",
        "matrix": "matrix_a",
        "generated_at": "2026-03-11T00:00:00+00:00",
        "config": {
            "features": [3],
            "n_dqi_samples": 64,
            "sa_iter": 100,
            "top_k_by_surrogate": 5,
            "seed": 42,
            "include_stage1_paired_encoding": True,
            "encodings": ["wht_truncated", "ilp_derived"],
            "decoders": ["bruteforce"],
            "alpha_modes": ["uniform"],
            "execution_mode": "mixture",
            "smoke": True,
        },
        "aggregated_csv": str(results_root / "tables" / "matrix_a_summary.csv"),
        "run_count": 1,
        "run_files": [],
    }
    (results_root / "matrix_a" / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    df = pd.DataFrame(
        [
            {
                "schema_version": 1.0,
                "matrix": "matrix_a",
                "run_id": "observed_a_1",
                "run_timestamp_utc": "2026-03-11T10:00:00+00:00",
                "run_status": "ok",
                "run_error": None,
                "instance_id": "P3",
                "family": "P",
                "encoding": "wht_truncated",
                "decoder": "bruteforce",
                "alpha_mode": "uniform",
                "execution_mode": "mixture",
                "seed": 42,
                "best_sampled_F": 300.0,
                "top1_regret": 0.0,
                "metric_availability": json.dumps(
                    {
                        "best_sampled_F": "available",
                        "top1_regret": "available",
                    }
                ),
            }
        ]
    )
    df.to_csv(results_root / "tables" / "matrix_a_summary.csv", index=False)
    (results_root / "reports" / "matrix_a_report.json").write_text(
        json.dumps({"matrix": "matrix_a", "record_count": 1}, indent=2)
    )


def _write_minimal_matrix_a_boundary_inputs(results_root: Path) -> None:
    (results_root / "matrix_a").mkdir(parents=True, exist_ok=True)
    (results_root / "matrix_a" / "runs").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)

    run_filename = "matrix_a_P7_ilp_derived_bp1_uniform_mixture_seed42.json"

    manifest_payload = {
        "schema_version": "1.0",
        "matrix": "matrix_a",
        "generated_at": "2026-03-15T00:00:00+00:00",
        "config": {
            "features": [7],
            "n_dqi_samples": 64,
            "sa_iter": 100,
            "top_k_by_surrogate": 5,
            "seed": 42,
            "include_stage1_paired_encoding": True,
            "encodings": ["wht_truncated", "ilp_derived"],
            "decoders": ["bp1"],
            "alpha_modes": ["uniform"],
            "execution_mode": "mixture",
            "smoke": False,
        },
        "aggregated_csv": str(results_root / "tables" / "matrix_a_summary.csv"),
        "run_count": 1,
        "run_files": [str(results_root / "matrix_a" / "runs" / run_filename)],
    }
    (results_root / "matrix_a" / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    row = {
        "schema_version": 1.0,
        "matrix": "matrix_a",
        "run_id": "matrix_a_P7_ilp_derived_bp1_uniform_mixture_seed42",
        "run_timestamp_utc": "2026-03-15T10:00:00+00:00",
        "run_status": "unavailable",
        "run_error": "qubit threshold exceeded",
        "instance_id": "P7",
        "family": "P",
        "encoding": "ilp_derived",
        "decoder": "bp1",
        "alpha_mode": "uniform",
        "execution_mode": "mixture",
        "seed": 42,
        "attempted_execution": False,
        "boundary_reason": "qubit_count_exceeded",
        "estimated_qubits": 31,
        "estimated_memory_gb": 16.0,
        "wall_clock_s": 0.0,
        "peak_memory_mb": 0.0,
        "best_sampled_F": None,
        "top1_regret": None,
        "metric_availability": json.dumps(
            {
                "best_sampled_F": "unavailable",
                "top1_regret": "unavailable",
            }
        ),
    }

    df = pd.DataFrame([row])
    df.to_csv(results_root / "tables" / "matrix_a_summary.csv", index=False)
    (results_root / "matrix_a" / "runs" / run_filename).write_text(json.dumps(row, indent=2))
    (results_root / "reports" / "matrix_a_report.json").write_text(
        json.dumps({"matrix": "matrix_a", "record_count": 1}, indent=2)
    )


def _write_minimal_matrix_c_inputs_with_decoder_study(results_root: Path) -> None:
    (results_root / "matrix_c").mkdir(parents=True, exist_ok=True)
    (results_root / "matrix_c" / "runs").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "schema_version": "1.0",
        "matrix": "matrix_c",
        "generated_at": "2026-03-12T00:00:00+00:00",
        "config": {
            "instance_ids": ["S_dense_low_collision"],
            "alpha_modes": ["paper"],
            "decoders": ["bp1"],
            "include_p_family": False,
            "seed": 7,
        },
        "aggregated_csv": str(results_root / "tables" / "matrix_c_summary.csv"),
        "run_count": 1,
        "run_files": [],
    }
    (results_root / "matrix_c" / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    legacy = pd.DataFrame(
        [
            {
                "schema_version": 1.0,
                "matrix": "matrix_c",
                "run_id": "legacy_benchmark_1",
                "run_timestamp_utc": "2026-03-12T10:00:00+00:00",
                "run_status": "completed",
                "instance_id": "S_dense_low_collision",
                "family": "S",
                "encoding": "synthetic_structured_bv",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "seed": 7,
                "best_sampled_F": 1.0,
                "topk_regret": 0.0,
                "metric_availability": json.dumps({"best_sampled_F": "available", "topk_regret": "available"}),
            }
        ]
    )
    legacy.to_csv(results_root / "tables" / "matrix_c_summary.csv", index=False)
    (results_root / "reports" / "matrix_c_report.json").write_text(
        json.dumps({"matrix": "matrix_c", "record_count": 1}, indent=2)
    )

    raw_rows = [
        {
            "matrix": "matrix_c",
            "stage": "decoder_study",
            "run_id": "study_bp1",
            "run_timestamp_utc": "2026-03-12T11:00:00+00:00",
            "run_status": "completed",
            "family": "S",
            "instance_id": "S_dense_low_collision",
            "encoding": "synthetic_structured_bv",
            "decoder": "bp1",
            "alpha_mode": "paper",
            "execution_mode": "mixture",
            "seed": 9,
            "best_sampled_F": 3.0,
            "metric_availability": {"best_sampled_F": "available"},
        },
        {
            "matrix": "matrix_c",
            "stage": "decoder_study",
            "run_id": "study_bf",
            "run_timestamp_utc": "2026-03-12T11:00:01+00:00",
            "run_status": "completed",
            "family": "S",
            "instance_id": "S_dense_low_collision",
            "encoding": "synthetic_structured_bv",
            "decoder": "bruteforce",
            "alpha_mode": "paper",
            "execution_mode": "mixture",
            "seed": 9,
            "best_sampled_F": 4.0,
            "metric_availability": {"best_sampled_F": "available"},
        },
        {
            "matrix": "matrix_c",
            "stage": "decoder_study",
            "run_id": "study_osd",
            "run_timestamp_utc": "2026-03-12T11:00:02+00:00",
            "run_status": "completed",
            "family": "S",
            "instance_id": "S_dense_low_collision",
            "encoding": "synthetic_structured_bv",
            "decoder": "bp_osd_lite",
            "alpha_mode": "paper",
            "execution_mode": "mixture",
            "seed": 9,
            "best_sampled_F": 6.0,
            "metric_availability": {"best_sampled_F": "available"},
        },
    ]
    for row in raw_rows:
        (results_root / "matrix_c" / "runs" / f"{row['run_id']}.json").write_text(json.dumps(row, indent=2))


def _write_minimal_matrix_b_inputs_with_alpha_validation(results_root: Path) -> None:
    (results_root / "matrix_b").mkdir(parents=True, exist_ok=True)
    (results_root / "matrix_b" / "runs").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "schema_version": "1.0",
        "matrix": "matrix_b",
        "generated_at": "2026-03-12T00:00:00+00:00",
        "config": {
            "instance_ids": ["C1"],
            "alpha_modes": ["paper"],
            "execution_modes": ["mixture", "coherent"],
            "decoders": ["oracle"],
            "seed": 42,
        },
        "aggregated_csv": str(results_root / "tables" / "matrix_b_summary.csv"),
        "run_count": 1,
        "run_files": [],
    }
    (results_root / "matrix_b" / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    legacy = pd.DataFrame(
        [
            {
                "schema_version": 1.0,
                "matrix": "matrix_b",
                "run_id": "legacy_bench_1",
                "run_timestamp_utc": "2026-03-12T10:00:00+00:00",
                "run_status": "completed",
                "instance_id": "C1",
                "family": "C",
                "encoding": "wht_exact",
                "decoder": "oracle",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "seed": 42,
                "best_F": 2.0,
                "best_top_G_sampled_F": 2.0,
                "postselection_success": 0.5,
                "success_probability": 0.5,
                "metric_availability": json.dumps({"best_F": "available", "success_probability": "available"}),
            }
        ]
    )
    legacy.to_csv(results_root / "tables" / "matrix_b_summary.csv", index=False)
    (results_root / "reports" / "matrix_b_report.json").write_text(
        json.dumps({"matrix": "matrix_b", "record_count": 1}, indent=2)
    )

    stage_rows = [
        {
            "matrix": "matrix_b",
            "stage": "alpha_validation",
            "run_id": "stage_d_mix",
            "run_timestamp_utc": "2026-03-12T11:00:00+00:00",
            "run_status": "completed",
            "status_reason": "completed",
            "in_experiment_matrix": True,
            "run_error": None,
            "family": "C",
            "instance_id": "C1",
            "encoding": "wht_exact",
            "decoder": "oracle",
            "alpha_mode": "paper",
            "execution_mode": "mixture",
            "ell": 1,
            "m_active": 4,
            "trial_seed": 42,
            "canonical_trial_seed": 42,
            "seed": 42,
            "oracle_exact": True,
            "coherent_supported": True,
            "best_F": 3.0,
            "best_top_G_sampled_F": 3.0,
            "postselection_success": 0.7,
            "success_probability": 0.7,
            "candidate_probabilities": [0.75, 0.25],
            "metric_availability": {"best_F": "available", "success_probability": "available"},
        },
        {
            "matrix": "matrix_b",
            "stage": "alpha_validation",
            "run_id": "stage_d_coh",
            "run_timestamp_utc": "2026-03-12T11:00:01+00:00",
            "run_status": "completed",
            "status_reason": "completed",
            "in_experiment_matrix": True,
            "run_error": None,
            "family": "C",
            "instance_id": "C1",
            "encoding": "wht_exact",
            "decoder": "oracle",
            "alpha_mode": "paper",
            "execution_mode": "coherent",
            "ell": 1,
            "m_active": 4,
            "trial_seed": 42,
            "canonical_trial_seed": 42,
            "seed": 42,
            "oracle_exact": True,
            "coherent_supported": True,
            "best_F": 5.0,
            "best_top_G_sampled_F": 4.0,
            "postselection_success": 0.9,
            "success_probability": 0.9,
            "candidate_probabilities": [0.5, 0.5],
            "metric_availability": {"best_F": "available", "success_probability": "available"},
        },
        {
            "matrix": "matrix_b",
            "stage": "alpha_validation",
            "run_id": "stage_d_bad_decoder",
            "run_timestamp_utc": "2026-03-12T11:00:02+00:00",
            "run_status": "completed",
            "status_reason": "completed",
            "in_experiment_matrix": True,
            "run_error": None,
            "family": "C",
            "instance_id": "C1",
            "encoding": "wht_exact",
            "decoder": "bp1",
            "alpha_mode": "uniform",
            "execution_mode": "mixture",
            "ell": 1,
            "m_active": 4,
            "trial_seed": 42,
            "canonical_trial_seed": 42,
            "seed": 42,
            "oracle_exact": False,
            "coherent_supported": True,
            "best_F": 1.0,
            "best_top_G_sampled_F": 1.0,
            "postselection_success": 0.2,
            "success_probability": 0.2,
            "candidate_probabilities": [1.0, 0.0],
            "metric_availability": {"best_F": "available", "success_probability": "available"},
        },
    ]
    for row in stage_rows:
        (results_root / "matrix_b" / "runs" / f"{row['run_id']}.json").write_text(json.dumps(row, indent=2))


def test_seed_normalization_and_slot_id_stability() -> None:
    assert normalize_seed_for_hash(None) == "NULL"
    assert normalize_seed_for_hash(float("nan")) == "NULL"
    assert normalize_seed_for_hash("42") == "42"
    assert normalize_seed_for_hash(42.0) == "42"

    kwargs = {
        "stage": "benchmark_matrix",
        "matrix": "matrix_a",
        "family": "P",
        "instance_id": "P3",
        "encoding": "wht_truncated",
        "decoder": "bruteforce",
        "alpha_mode": "paper",
        "execution_mode": "mixture",
        "trial_seed": 42,
    }
    assert build_slot_id(**kwargs) == build_slot_id(**kwargs)


def test_duplicate_resolution_tie_break_order() -> None:
    candidates = [
        {
            "run_id": "z_run",
            "_canonical_status": "failed",
            "run_timestamp_utc": "2026-03-11T09:00:00+00:00",
            "_source_file_mtime": 100.0,
        },
        {
            "run_id": "a_run",
            "_canonical_status": "completed",
            "run_timestamp_utc": "2026-03-11T08:00:00+00:00",
            "_source_file_mtime": 99.0,
        },
        {
            "run_id": "b_run",
            "_canonical_status": "completed",
            "run_timestamp_utc": "2026-03-11T08:00:00+00:00",
            "_source_file_mtime": 99.0,
        },
    ]
    selected, run_ids, count = _resolve_duplicates(candidates)
    assert selected["run_id"] == "a_run"
    assert count == 3
    assert run_ids == ["a_run", "b_run", "z_run"]


def test_run_make_report_unobserved_slot_defaults_and_availability_invariants(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_minimal_matrix_a_inputs(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_a"])
    assert Path(out["run_manifest"]).exists()
    assert Path(out["resources_master"]).exists()
    assert Path(out["metrics_master"]).exists()

    metrics_json = json.loads((results_root / "metrics_master.json").read_text())
    rows = metrics_json["rows"]
    assert len(rows) == 3  # one observed + two planned-not-executed slots

    planned = [r for r in rows if r["observed_run"] is False]
    assert planned, "expected planned but unobserved rows"
    for r in planned:
        assert r["run_status"] == "not_applicable"
        assert r["status_reason_code"] == "planned_not_executed"
        assert r["availability_best_sampled_F"] == "not_applicable"
        assert r["best_sampled_F"] is None

    completed = [r for r in rows if r["run_status"] == "completed"]
    assert len(completed) == 1
    c = completed[0]
    assert c["availability_best_sampled_F"] == "available"
    assert c["metric_availability"]["best_sampled_F"] == "available"

    # Compatibility view is projected from master.
    compat = pd.read_csv(results_root / "tables" / "matrix_a_summary.csv")
    assert len(compat) == 3


def test_make_report_stage_trial_seed_and_by_stage_for_matrix_c(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_minimal_matrix_c_inputs_with_decoder_study(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_c"])
    assert Path(out["run_manifest"]).exists()
    assert Path(out["metrics_master"]).exists()

    metrics_path = Path(out["metrics_master"])
    if metrics_path.suffix == ".parquet":
        metrics = pd.read_parquet(metrics_path)
    else:
        payload = json.loads(metrics_path.read_text())
        metrics = pd.DataFrame(payload["rows"])
    assert "stage" in metrics.columns
    assert "trial_seed" in metrics.columns
    assert "canonical_trial_seed" in metrics.columns

    study = metrics[metrics["stage"] == "decoder_study"]
    assert not study.empty
    assert set(study["trial_seed"].dropna().astype(int).tolist()) == {9}
    assert set(study["canonical_trial_seed"].dropna().astype(int).tolist()) == {9}

    osd = study[study["decoder"] == "bp_osd_lite"].iloc[0]
    assert float(osd["gain_over_bp1"]) == 3.0
    assert float(osd["gain_over_bruteforce"]) == 2.0

    report = json.loads((results_root / "reports" / "matrix_c_report.json").read_text())
    assert "by_stage" in report
    assert "benchmark_matrix" in report["by_stage"]
    assert "decoder_study" in report["by_stage"]


def test_make_report_stage_d_derivations_and_guardrail_for_matrix_b(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_minimal_matrix_b_inputs_with_alpha_validation(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_b"])
    assert Path(out["metrics_master"]).exists()

    metrics_path = Path(out["metrics_master"])
    if metrics_path.suffix == ".parquet":
        metrics = pd.read_parquet(metrics_path)
    else:
        payload = json.loads(metrics_path.read_text())
        metrics = pd.DataFrame(payload["rows"])
    stage_d = metrics[metrics["stage"] == "alpha_validation"].copy()
    assert not stage_d.empty

    # Guardrail: non-oracle alpha_validation row should be dropped/quarantined.
    assert not ((stage_d["decoder"] != "oracle") & (stage_d["family"] == "C")).any()

    coherent = stage_d[
        (stage_d["instance_id"] == "C1")
        & (stage_d["alpha_mode"] == "paper")
        & (stage_d["execution_mode"] == "coherent")
    ].iloc[0]
    assert float(coherent["coherent_vs_mixture_delta_best_F"]) == 2.0
    assert float(coherent["coherent_vs_mixture_delta_best_top_G_sampled_F"]) == 1.0
    assert float(coherent["coherent_vs_mixture_delta_postselection_success"]) == pytest.approx(0.2)
    assert coherent["coherent_vs_mixture_distribution_distance"] is not None
    assert coherent["alpha_ranking_within_instance"] is not None
    assert coherent["baseline_selector_policy"] == "fixed_priority"
    assert coherent["baseline_min_pairs"] == 8
    assert coherent["baseline_selected_reason"] == "no_baseline_meets_threshold"
    assert coherent["baseline_fallback_used"] is False
    assert coherent["best_available_pair_count"] == 1
    assert coherent["is_stage_d_selected_baseline"] is False
    assert pd.isna(coherent["baseline_encoding_norm"])
    assert pd.isna(coherent["baseline_decoder_norm"])
    assert pd.isna(coherent["baseline_pair_count"])
    assert pd.isna(coherent["baseline_priority_rank"])


def test_make_report_preserves_unavailable_boundary_rows(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_minimal_matrix_a_boundary_inputs(results_root)

    run_make_report(results_root=results_root, matrices=["matrix_a"])
    metrics_payload = json.loads((results_root / "metrics_master.json").read_text())
    metrics = pd.DataFrame(metrics_payload["rows"])
    row = metrics.loc[
        (metrics["run_id"] == "matrix_a_P7_ilp_derived_bp1_uniform_mixture_seed42")
        & (metrics["observed_run"] == True)
    ].iloc[0]

    assert row["run_status"] == "unavailable"
    assert row["run_status_norm"] == "unavailable"
    assert bool(row["attempted_execution"]) is False
