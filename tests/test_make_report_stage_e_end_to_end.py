"""Stage E end-to-end smoke test for make_report outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, ".")

from scripts.make_report import run_make_report


def _load_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    payload = json.loads(path.read_text())
    return payload.get("rows", payload if isinstance(payload, list) else [])


def _write_inputs(results_root: Path) -> None:
    (results_root / "matrix_a").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)
    (results_root / "paired_pricing").mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "schema_version": "1.0",
        "matrix": "matrix_a",
        "generated_at": "2026-03-12T00:00:00+00:00",
        "config": {
            "features": [3],
            "decoders": ["bruteforce"],
            "alpha_modes": ["uniform"],
            "seed": 1,
        },
    }
    (results_root / "matrix_a" / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    summary = pd.DataFrame(
        [
            {
                "matrix": "matrix_a",
                "run_id": "e2e_completed",
                "run_timestamp_utc": "2026-03-12T10:00:00+00:00",
                "run_status": "completed",
                "instance_id": "P3",
                "family": "P",
                "encoding": "wht_truncated",
                "decoder": "bruteforce",
                "alpha_mode": "uniform",
                "execution_mode": "mixture",
                "trial_seed": 1,
                "best_sampled_F": 5.0,
                "best_top_G_sampled_F": 6.0,
                "topk_regret": 0.2,
                "decision_distortion": 0.1,
                "spearman_rho_F_G": 0.9,
                "retained_energy_eta": 0.8,
                "resource_total_qubits_est_with_decode": 3,
                "resource_total_gate_est_with_decode": 20,
                "resource_total_depth_est_with_decode": 11,
                "resource_decode_uncompute_confidence": "implementation_proxy",
            }
        ]
    )
    summary.to_csv(results_root / "tables" / "matrix_a_summary.csv", index=False)
    (results_root / "reports" / "matrix_a_report.json").write_text(
        json.dumps({"matrix": "matrix_a", "record_count": 1}, indent=2)
    )


def test_make_report_emits_stage_e_outputs_and_preserves_unavailable_auditability(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_inputs(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_a"])

    master_path = Path(out["quality_cost_master"])
    unavailable_path = Path(out["quality_cost_unavailable"])
    assert master_path.exists()
    assert unavailable_path.exists()

    master_rows = _load_rows(master_path)
    unavailable_rows = _load_rows(unavailable_path)
    assert master_rows
    assert unavailable_rows  # planned-but-unobserved rows should land here

    for row in unavailable_rows:
        assert "unavailable_class" in row
        assert "unavailable_reason_code" in row
        assert "unavailable_reason" in row
