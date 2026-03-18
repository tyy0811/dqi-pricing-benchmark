"""Regression tests for Qiskit structural-subset resource preservation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, ".")

from src.matrix_runner import _resource_fields
from src.resources_compare import _build_resources_block
from scripts.make_report import run_make_report


def _write_minimal_matrix_c_inputs_with_qiskit_subset_fields(results_root: Path) -> None:
    (results_root / "matrix_c").mkdir(parents=True, exist_ok=True)
    (results_root / "matrix_c" / "runs").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "reports").mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "schema_version": "1.0",
        "matrix": "matrix_c",
        "generated_at": "2026-03-16T00:00:00+00:00",
        "config": {
            "instance_ids": ["S_dense_low_collision"],
            "alpha_modes": ["paper"],
            "decoders": ["bp1"],
            "include_p_family": False,
            "seed": 9,
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
                "run_id": "legacy_bench_1",
                "run_timestamp_utc": "2026-03-16T10:00:00+00:00",
                "run_status": "completed",
                "instance_id": "S_dense_low_collision",
                "family": "S",
                "encoding": "synthetic_structured_bv",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "seed": 9,
                "best_F": 2.0,
                "best_top_G_sampled_F": 2.0,
                "topk_regret": 0.0,
                "metric_availability": json.dumps({"best_F": "available", "topk_regret": "available"}),
            }
        ]
    )
    legacy.to_csv(results_root / "tables" / "matrix_c_summary.csv", index=False)
    (results_root / "reports" / "matrix_c_report.json").write_text(
        json.dumps({"matrix": "matrix_c", "record_count": 1}, indent=2)
    )

    qiskit_subset_resource_report = {
        "transpiled_depth": 17,
        "cx_count": 39,
        "qubit_count": 9,
        "status": "ok",
    }
    qiskit_transpiled_structural_count = {
        "status": "ok",
        "n_operations": 102,
    }

    observed_row = {
        "matrix": "matrix_c",
        "stage": "decoder_study",
        "run_id": "matrix_c_qiskit_subset_case",
        "run_timestamp_utc": "2026-03-16T11:00:00+00:00",
        "run_status": "completed",
        "status_reason": "completed",
        "in_experiment_matrix": True,
        "run_error": None,
        "family": "S",
        "instance_id": "S_dense_low_collision",
        "encoding": "synthetic_structured_bv",
        "decoder": "bp1",
        "alpha_mode": "paper",
        "execution_mode": "mixture",
        "seed": 9,
        "best_sampled_F": 3.0,
        "best_F": 3.0,
        "topk_regret": 0.1,
        "decision_distortion": 0.05,
        "spearman_rho_F_G": 0.85,
        "retained_energy_eta": 0.72,
        "resource_total_qubits_est_with_decode": 9,
        "resource_total_gate_est_with_decode": 120,
        "resource_total_depth_est_with_decode": 30,
        "resource_decode_uncompute_confidence": "implementation_proxy",
        "qiskit_subset_resource_report": qiskit_subset_resource_report,
        "qiskit_transpiled_structural_count": qiskit_transpiled_structural_count,
        "qiskit_subset_transpiled_depth": 17,
        "qiskit_subset_cx_count": 39,
        "qiskit_subset_qubit_count": 9,
        "qiskit_subset_status": "ok",
        "metric_availability": {"best_sampled_F": "available", "best_F": "available"},
    }

    (results_root / "matrix_c" / "runs" / f"{observed_row['run_id']}.json").write_text(
        json.dumps(observed_row, indent=2)
    )


def test_resources_compare_preserves_qiskit_subset_payload_and_convenience_fields() -> None:
    qiskit_subset_resource_report = {
        "transpiled_depth": 17,
        "cx_count": 39,
        "qubit_count": 9,
        "status": "ok",
    }
    qiskit_transpiled_structural_count = {
        "status": "ok",
        "transpiled_gate_count": 102,
        "n_layers": 3,
    }

    resource_summary = {
        "implemented_resources": {"prefix": {"qubits": 4}},
        "analytic_estimates": {"decode_model_gate_est": 4.0},
        "total_estimated_resources": {"gates": 120},
        "total_gate_est_with_decode": 120,
        "total_depth_est_with_decode": 30,
        "total_qubits_est_with_decode": 9,
        "decode_uncompute_confidence": "implementation_proxy",
        "decode_uncompute_status": "implemented",
        "qiskit_subset_resource_report": qiskit_subset_resource_report,
        "qiskit_transpiled_structural_count": qiskit_transpiled_structural_count,
    }

    out, has_resources = _build_resources_block(resource_summary=resource_summary, execution_mode="mixture")

    assert has_resources is True
    assert out["qiskit_subset_resource_report"] == qiskit_subset_resource_report
    assert out["qiskit_transpiled_structural_count"] == qiskit_transpiled_structural_count
    assert out["qiskit_subset_transpiled_depth"] == 17
    assert out["qiskit_subset_cx_count"] == 39
    assert out["qiskit_subset_qubit_count"] == 9
    assert out["qiskit_subset_status"] == "ok"
    assert out["resource_status"]["transpiled_subset_available"] is True


def test_run_make_report_preserves_qiskit_subset_resource_fields_in_resources_master(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_minimal_matrix_c_inputs_with_qiskit_subset_fields(results_root)

    out = run_make_report(results_root=results_root, matrices=["matrix_c"])

    resources_payload = json.loads(Path(out["resources_master"]).read_text())
    resources_rows = resources_payload["rows"]

    observed = [
        row
        for row in resources_rows
        if row.get("run_id") == "matrix_c_qiskit_subset_case"
        and bool(row.get("observed_run"))
    ]
    assert len(observed) == 1
    row = observed[0]

    assert row["qiskit_subset_resource_report"] == {
        "transpiled_depth": 17,
        "cx_count": 39,
        "qubit_count": 9,
        "status": "ok",
    }
    assert row["qiskit_transpiled_structural_count"] == {"status": "ok", "n_operations": 102}
    assert row["qiskit_subset_transpiled_depth"] == 17
    assert row["qiskit_subset_cx_count"] == 39
    assert row["qiskit_subset_qubit_count"] == 9
    assert row["qiskit_subset_status"] == "ok"


def test_matrix_runner_resource_fields_preserves_qiskit_subset_data() -> None:
    """Test that _resource_fields() extracts Qiskit subset fields correctly."""
    qiskit_subset_resource_report = {
        "transpiled_depth": 25,
        "cx_count": 42,
        "qubit_count": 12,
        "status": "ok",
    }
    qiskit_transpiled_structural_count = {
        "status": "ok",
        "category": "qiskit_transpiled_structural_count",
        "qubit_count": 12,
        "transpiled_depth": 25,
        "cx_count": 42,
        "gate_counts": {"cx": 42, "rz": 15},
    }

    resource_summary = {
        "decode_uncompute_model": "lookup_table_reversible",
        "decode_uncompute_status": "analytic_estimate_conservative",
        "decode_uncompute_confidence": "analytic_upper_bound",
        "total_gate_est_with_decode": 500,
        "total_depth_est_with_decode": 100,
        "total_qubits_est_with_decode": 20,
        "prefix_gate_est": 50,
        "prefix_depth_est": 15,
        "decode_model_gate_est": 450,
        "decode_model_depth_est": 85,
        "weight_register_qubits": 2,
        "coherent_weight_register_gate_est": 16,
        "coherent_weight_register_depth_est": 5,
        "qiskit_subset_resource_report": qiskit_subset_resource_report,
        "qiskit_transpiled_structural_count": qiskit_transpiled_structural_count,
    }

    out = _resource_fields(resource_summary)

    # Check nested payloads are preserved
    assert out["resource_qiskit_subset_resource_report"] == qiskit_subset_resource_report
    assert out["resource_qiskit_transpiled_structural_count"] == qiskit_transpiled_structural_count
    assert out["qiskit_subset_resource_report"] == qiskit_subset_resource_report
    assert out["qiskit_transpiled_structural_count"] == qiskit_transpiled_structural_count

    # Check flattened convenience fields
    assert out["resource_qiskit_subset_transpiled_depth"] == 25.0
    assert out["resource_qiskit_subset_cx_count"] == 42.0
    assert out["resource_qiskit_subset_qubit_count"] == 12.0
    assert out["resource_qiskit_subset_status"] == "ok"
    assert out["qiskit_subset_transpiled_depth"] == 25
    assert out["qiskit_subset_cx_count"] == 42
    assert out["qiskit_subset_qubit_count"] == 12
    assert out["qiskit_subset_status"] == "ok"


def test_matrix_runner_resource_fields_handles_missing_qiskit_gracefully() -> None:
    """Test that _resource_fields() handles missing Qiskit data without error."""
    resource_summary = {
        "decode_uncompute_model": "bp1_classical_oracle_estimate",
        "decode_uncompute_status": "analytic_estimate_conservative",
        "decode_uncompute_confidence": "implementation_proxy",
        "total_gate_est_with_decode": 200,
        "total_depth_est_with_decode": 40,
        "total_qubits_est_with_decode": 15,
        # No Qiskit fields present
    }

    out = _resource_fields(resource_summary)

    # Check that Qiskit fields are None but don't cause errors
    assert out["qiskit_subset_resource_report"] is None
    assert out["qiskit_transpiled_structural_count"] is None
    assert out["qiskit_subset_transpiled_depth"] is None
    assert out["qiskit_subset_cx_count"] is None
    assert out["qiskit_subset_qubit_count"] is None
    assert out["qiskit_subset_status"] is None

    # Other fields should still be present
    assert out["resource_decode_uncompute_model"] == "bp1_classical_oracle_estimate"
    assert out["resource_total_gate_est_with_decode"] == 200.0
