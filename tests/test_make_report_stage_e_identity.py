"""Stage E identity and seed canonicalization tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, ".")

from scripts.make_report import run_make_report


def _load_rows(results_root: Path, out: dict) -> list[dict]:
    path = Path(out["metrics_master"])
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    return json.loads(path.read_text())["rows"]


def _write_minimal_matrix_a_identity_inputs(results_root: Path) -> None:
    (results_root / "matrix_a").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "schema_version": "1.0",
        "matrix": "matrix_a",
        "generated_at": "2026-03-12T00:00:00+00:00",
        "config": {
            "features": [3],
            "decoders": ["bruteforce"],
            "alpha_modes": ["uniform"],
            "seed": 42,
        },
    }
    (results_root / "matrix_a" / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    df = pd.DataFrame(
        [
            {
                "matrix": "matrix_a",
                "run_id": "identity_seed_row",
                "run_timestamp_utc": "2026-03-12T10:00:00+00:00",
                "run_status": "completed",
                "instance_id": "P3",
                "family": "P",
                "encoding": "wht_truncated",
                "decoder": "bruteforce",
                "alpha_mode": "uniform",
                "execution_mode": "mixture",
                "canonical_trial_seed": 11,
                "seed": 7,
                "best_sampled_F": 100.0,
                "metric_availability": json.dumps({"best_sampled_F": "available"}),
            }
        ]
    )
    df.to_csv(results_root / "tables" / "matrix_a_summary.csv", index=False)
    (results_root / "reports" / "matrix_a_report.json").write_text(
        json.dumps({"matrix": "matrix_a", "record_count": 1}, indent=2)
    )


def _write_matrix_a_paired_identity_inputs(results_root: Path) -> None:
    (results_root / "matrix_a").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "schema_version": "1.0",
        "matrix": "matrix_a",
        "generated_at": "2026-03-12T00:00:00+00:00",
        "config": {
            "features": [3],
            "decoders": ["bruteforce"],
            "alpha_modes": ["uniform"],
            "seed": 42,
        },
    }
    (results_root / "matrix_a" / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    df = pd.DataFrame(
        [
            {
                "matrix": "matrix_a",
                "stage": "benchmark_matrix",
                "run_id": "pair_wht",
                "run_timestamp_utc": "2026-03-12T10:00:00+00:00",
                "run_status": "completed",
                "instance_id": "P3",
                "family": "P",
                "encoding": "wht_truncated",
                "decoder": "bruteforce",
                "alpha_mode": "uniform",
                "execution_mode": "mixture",
                "trial_seed": 42,
                "best_sampled_F": 100.0,
                "topk_regret": 0.1,
                "decision_distortion": 0.2,
                "spearman_rho_F_G": 0.95,
                "retained_energy_eta": 0.8,
                "resource_total_qubits_est_with_decode": 5,
                "resource_total_gate_est_with_decode": 200,
                "resource_total_depth_est_with_decode": 80,
                "resource_decode_uncompute_confidence": "implementation_proxy",
            },
            {
                "matrix": "matrix_a",
                "stage": "benchmark_matrix",
                "run_id": "pair_ilp",
                "run_timestamp_utc": "2026-03-12T10:01:00+00:00",
                "run_status": "completed",
                "instance_id": "P3",
                "family": "P",
                "encoding": "ilp_derived",
                "decoder": "bruteforce",
                "alpha_mode": "uniform",
                "execution_mode": "mixture",
                "trial_seed": 42,
                "best_sampled_F": 101.0,
                "topk_regret": 0.2,
                "decision_distortion": 0.25,
                "spearman_rho_F_G": 0.9,
                "retained_energy_eta": 0.75,
                "resource_total_qubits_est_with_decode": 6,
                "resource_total_gate_est_with_decode": 250,
                "resource_total_depth_est_with_decode": 90,
                "resource_decode_uncompute_confidence": "implementation_proxy",
            },
        ]
    )
    df.to_csv(results_root / "tables" / "matrix_a_summary.csv", index=False)
    (results_root / "reports" / "matrix_a_report.json").write_text(
        json.dumps({"matrix": "matrix_a", "record_count": 2}, indent=2)
    )


def test_manifest_slot_required_in_run_manifest_rows(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_minimal_matrix_a_identity_inputs(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_a"])
    manifest = json.loads(Path(out["run_manifest"]).read_text())

    assert manifest["slots"], "expected non-empty run_manifest slots"
    assert all("manifest_slot" in slot for slot in manifest["slots"])


def test_canonical_trial_seed_precedence(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_minimal_matrix_a_identity_inputs(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_a"])
    rows = _load_rows(results_root, out)

    row = [r for r in rows if r.get("run_id") == "identity_seed_row"][0]
    assert row["canonical_trial_seed"] == 11


def test_metrics_master_persists_normalized_identity_columns_and_comparison_key(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_a_paired_identity_inputs(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_a"])
    rows = _load_rows(results_root, out)
    metrics = pd.DataFrame(rows)

    required_cols = [
        "matrix_norm",
        "stage_norm",
        "family_norm",
        "encoding_norm",
        "decoder_norm",
        "alpha_mode_norm",
        "execution_mode_norm",
        "comparison_key",
    ]
    for col in required_cols:
        assert col in metrics.columns

    pair_rows = metrics[metrics["run_id"].isin(["pair_wht", "pair_ilp"])].copy()
    assert len(pair_rows) == 2
    assert pair_rows["comparison_key"].notna().all()
    assert pair_rows["comparison_key"].nunique() == 1
    assert pair_rows["encoding_norm"].nunique() == 2


def test_stage_e_exports_project_normalized_identity_columns_and_comparison_key(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_a_paired_identity_inputs(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_a"])

    master_path = Path(out["quality_cost_master"])
    if master_path.suffix == ".parquet":
        master = pd.read_parquet(master_path)
    else:
        master = pd.DataFrame(json.loads(master_path.read_text())["rows"])

    required_cols = [
        "matrix_norm",
        "stage_norm",
        "family_norm",
        "encoding_norm",
        "decoder_norm",
        "alpha_mode_norm",
        "execution_mode_norm",
        "comparison_key",
    ]
    for col in required_cols:
        assert col in master.columns

    pair_rows = master[master["run_id"].isin(["pair_wht", "pair_ilp"])].copy()
    assert len(pair_rows) == 2
    assert pair_rows["comparison_key"].notna().all()
    assert pair_rows["comparison_key"].nunique() == 1
