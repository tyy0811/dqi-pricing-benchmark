"""Stage E quality-cost partition tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, ".")

from scripts.make_report import run_make_report


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return pd.DataFrame(rows)


def _write_matrix_a_inputs_with_quality_cost_fields(results_root: Path) -> None:
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
            "seed": 123,
        },
    }
    (results_root / "matrix_a" / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    df = pd.DataFrame(
        [
            {
                "matrix": "matrix_a",
                "run_id": "qc_completed",
                "run_timestamp_utc": "2026-03-12T10:00:00+00:00",
                "run_status": "completed",
                "instance_id": "P3",
                "family": "P",
                "encoding": "wht_truncated",
                "decoder": "bruteforce",
                "alpha_mode": "uniform",
                "execution_mode": "mixture",
                "trial_seed": 123,
                "best_sampled_F": 10.0,
                "best_top_G_sampled_F": 12.0,
                "topk_regret": 0.1,
                "decision_distortion": 0.2,
                "spearman_rho_F_G": 0.9,
                "retained_energy_eta": 0.8,
                "resource_total_qubits_est_with_decode": 5,
                "resource_total_gate_est_with_decode": 200,
                "resource_total_depth_est_with_decode": 80,
                "resource_decode_uncompute_confidence": "implementation_proxy",
                "metric_availability": json.dumps(
                    {
                        "best_sampled_F": "available",
                        "topk_regret": "available",
                        "decision_distortion": "available",
                        "spearman_rho_F_G": "available",
                        "retained_energy_eta": "available",
                    }
                ),
            }
        ]
    )
    df.to_csv(results_root / "tables" / "matrix_a_summary.csv", index=False)
    (results_root / "reports" / "matrix_a_report.json").write_text(
        json.dumps({"matrix": "matrix_a", "record_count": 1}, indent=2)
    )


def test_completed_comparable_rows_go_to_quality_cost_master_only(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_a_inputs_with_quality_cost_fields(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_a"])

    master = _load_table(Path(out["quality_cost_master"]))
    unavailable = _load_table(Path(out["quality_cost_unavailable"]))

    assert "qc_completed" in set(master["run_id"])
    assert "qc_completed" not in set(unavailable["run_id"])


def test_unavailable_rows_preserved_with_class_and_reason_fields(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_a_inputs_with_quality_cost_fields(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_a"])
    unavailable = _load_table(Path(out["quality_cost_unavailable"]))

    assert not unavailable.empty
    for col in ["unavailable_class", "unavailable_reason_code", "unavailable_reason"]:
        assert col in unavailable.columns
    assert "skipped" not in set(unavailable["run_status"])
    assert set(unavailable["run_status"]).issubset(
        {"completed", "failed", "not_applicable", "unavailable", "not_in_experiment_matrix"}
    )


def test_quality_cost_files_written_json_and_parquet_or_fallback_json(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_a_inputs_with_quality_cost_fields(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_a"])
    assert Path(out["quality_cost_master"]).exists()
    assert Path(out["quality_cost_unavailable"]).exists()
    assert out["quality_cost_master_mode"] in {"parquet", "json"}
    assert out["quality_cost_unavailable_mode"] in {"parquet", "json"}
