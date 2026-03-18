"""Required Stage E audit artifact tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, ".")

import scripts.make_report as make_report
from scripts.make_report import run_make_report


DUPLICATE_AUDIT_COLUMNS = [
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


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return pd.DataFrame(rows)


def _write_matrix_a_audit_inputs(results_root: Path) -> None:
    (results_root / "matrix_a").mkdir(parents=True, exist_ok=True)
    (results_root / "matrix_c").mkdir(parents=True, exist_ok=True)
    (results_root / "matrix_c" / "runs").mkdir(parents=True, exist_ok=True)
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
    matrix_c_manifest = {
        "schema_version": "1.0",
        "matrix": "matrix_c",
        "generated_at": "2026-03-12T00:00:00+00:00",
        "config": {
            "instance_ids": ["S1"],
            "alpha_modes": ["paper"],
            "decoders": ["bp1"],
            "seed": 7,
        },
    }
    (results_root / "matrix_c" / "manifest.json").write_text(json.dumps(matrix_c_manifest, indent=2))

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
                "best_sampled_F": 10.0,
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
                "run_id": "pair_ilp_a",
                "run_timestamp_utc": "2026-03-12T10:01:00+00:00",
                "run_status": "completed",
                "instance_id": "P3",
                "family": "P",
                "encoding": "ilp_derived",
                "decoder": "bruteforce",
                "alpha_mode": "uniform",
                "execution_mode": "mixture",
                "trial_seed": 42,
                "best_sampled_F": 11.0,
                "topk_regret": 0.2,
                "decision_distortion": 0.3,
                "spearman_rho_F_G": 0.9,
                "retained_energy_eta": 0.75,
                "resource_total_qubits_est_with_decode": 6,
                "resource_total_gate_est_with_decode": 250,
                "resource_total_depth_est_with_decode": 90,
                "resource_decode_uncompute_confidence": "implementation_proxy",
            },
            {
                "matrix": "matrix_a",
                "stage": "benchmark_matrix",
                "run_id": "pair_ilp_b",
                "run_timestamp_utc": "2026-03-12T10:02:00+00:00",
                "run_status": "completed",
                "instance_id": "P3",
                "family": "P",
                "encoding": "ilp_derived",
                "decoder": "bruteforce",
                "alpha_mode": "uniform",
                "execution_mode": "mixture",
                "trial_seed": 42,
                "best_sampled_F": 9.0,
                "topk_regret": 0.25,
                "decision_distortion": 0.35,
                "spearman_rho_F_G": 0.88,
                "retained_energy_eta": 0.72,
                "resource_total_qubits_est_with_decode": 6,
                "resource_total_gate_est_with_decode": 250,
                "resource_total_depth_est_with_decode": 90,
                "resource_decode_uncompute_confidence": "implementation_proxy",
            },
        ]
    )
    df.to_csv(results_root / "tables" / "matrix_a_summary.csv", index=False)
    pd.DataFrame(columns=["run_id"]).to_csv(results_root / "tables" / "matrix_c_summary.csv", index=False)
    (results_root / "reports" / "matrix_a_report.json").write_text(
        json.dumps({"matrix": "matrix_a", "record_count": 3}, indent=2)
    )
    (results_root / "reports" / "matrix_c_report.json").write_text(
        json.dumps({"matrix": "matrix_c", "record_count": 2}, indent=2)
    )

    for run_id, manifest_slot, verified in [
        ("dup_run_a", "slot-a", True),
        ("dup_run_b", "slot-b", True),
    ]:
        (results_root / "matrix_c" / "runs" / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "matrix": "matrix_c",
                    "stage": "decoder_study",
                    "run_id": run_id,
                    "manifest_slot": manifest_slot,
                    "run_timestamp_utc": "2026-03-12T12:00:00+00:00",
                    "run_status": "completed",
                    "status_reason_code": "completed",
                    "status_reason": "completed",
                    "family": "S",
                    "instance_id": "S1",
                    "encoding": "synthetic_structured_bv",
                    "decoder": "bp1",
                    "alpha_mode": "paper",
                    "execution_mode": "mixture",
                    "trial_seed": 7,
                    "canonical_trial_seed": 7,
                    "seed": 7,
                    "best_sampled_F": 9.0,
                    "topk_regret": 0.1,
                    "decision_distortion": 0.2,
                    "spearman_rho_F_G": 0.8,
                    "retained_energy_eta": 0.7,
                    "resource_total_qubits_est_with_decode": 5,
                    "resource_total_gate_est_with_decode": 100,
                    "resource_total_depth_est_with_decode": 20,
                    "resource_decode_uncompute_confidence": "implementation_proxy",
                    "has_verified_provenance": verified,
                },
                indent=2,
            )
        )


def test_run_make_report_writes_required_stage_e_audits_with_reconciliation_fields(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_matrix_a_audit_inputs(results_root)

    run_make_report(results_root=results_root, matrices=["matrix_a", "matrix_c"])

    pairing = _load_table(results_root / "matrix_a_pairing_audit.json")
    admission = _load_table(results_root / "quality_cost_admission_audit.json")
    duplicate = _load_table(results_root / "quality_cost_duplicate_audit.json")

    assert not pairing.empty
    assert not admission.empty
    assert not duplicate.empty

    for col in [
        "run_key",
        "stage_e_admission_key",
        "matrix_norm",
        "stage_norm",
        "family_norm",
        "encoding_norm",
        "decoder_norm",
        "alpha_mode_norm",
        "execution_mode_norm",
    ]:
        assert col in admission.columns
    assert "comparison_key" in admission.columns
    assert "counterpart_key" in admission.columns

    for col in [
        *DUPLICATE_AUDIT_COLUMNS,
    ]:
        assert col in duplicate.columns

    for col in [
        "run_key",
        "comparison_key",
        "matrix_norm",
        "stage_norm",
        "family_norm",
        "encoding_norm",
        "decoder_norm",
        "alpha_mode_norm",
        "execution_mode_norm",
    ]:
        assert col in pairing.columns


def test_run_make_report_fails_when_quality_cost_unavailable_violates_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results"
    _write_matrix_a_audit_inputs(results_root)

    def _bad_quality_cost_tables(*_args, **_kwargs):
        return (
            pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "run_id": "bad_row",
                        "instance_id": "P3",
                        "family": "P",
                        "encoding": "wht_truncated",
                        "decoder": "bruteforce",
                        "alpha_mode": "uniform",
                        "execution_mode": "mixture",
                        "unavailable_class": "missing_required_fields",
                        "unavailable_reason": "bad row",
                        "run_status": "not_applicable",
                        "resource_status": None,
                        "has_verified_provenance": False,
                        "missing_required_fields": [],
                        "missing_quality_fields": [],
                        "missing_resource_fields": [],
                    }
                ]
            ),
        )

    monkeypatch.setattr(make_report, "_build_quality_cost_tables", _bad_quality_cost_tables)

    with pytest.raises(ValueError, match="quality_cost_unavailable"):
        run_make_report(results_root=results_root, matrices=["matrix_a", "matrix_c"])


def test_run_make_report_preserves_duplicate_audit_schema_when_empty(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    (results_root / "matrix_a").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)
    (results_root / "matrix_a" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "matrix": "matrix_a",
                "generated_at": "2026-03-12T00:00:00+00:00",
                "config": {
                    "features": [3],
                    "decoders": ["bp1"],
                    "alpha_modes": ["paper"],
                    "seed": 42,
                },
            },
            indent=2,
        )
    )
    pd.DataFrame(
        [
            {
                "matrix": "matrix_a",
                "stage": "benchmark_matrix",
                "run_id": "only_row",
                "run_timestamp_utc": "2026-03-12T10:00:00+00:00",
                "run_status": "completed",
                "instance_id": "P3",
                "family": "P",
                "encoding": "wht_truncated",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "trial_seed": 42,
                "best_sampled_F": 10.0,
                "topk_regret": 0.1,
                "decision_distortion": 0.2,
                "spearman_rho_F_G": 0.9,
                "retained_energy_eta": 0.8,
                "resource_total_qubits_est_with_decode": 5,
                "resource_total_gate_est_with_decode": 100,
                "resource_total_depth_est_with_decode": 20,
                "resource_decode_uncompute_confidence": "implementation_proxy",
            }
        ]
    ).to_csv(results_root / "tables" / "matrix_a_summary.csv", index=False)
    (results_root / "reports" / "matrix_a_report.json").write_text(
        json.dumps({"matrix": "matrix_a", "record_count": 1}, indent=2)
    )

    run_make_report(results_root=results_root, matrices=["matrix_a"])

    payload = json.loads((results_root / "quality_cost_duplicate_audit.json").read_text())
    assert payload["rows"] == []
    assert payload["columns"] == DUPLICATE_AUDIT_COLUMNS
