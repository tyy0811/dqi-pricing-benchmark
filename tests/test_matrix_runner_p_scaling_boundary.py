"""Tests for P-family boundary/status helpers in matrix runner."""

from __future__ import annotations

import numpy as np

from src.matrix_runner import run_matrix_c, summarize_status_counts


def test_summarize_status_counts_by_instance() -> None:
    rows = [
        {"instance_id": "P6", "run_status": "completed"},
        {"instance_id": "P7", "run_status": "unavailable"},
        {"instance_id": "P7", "run_status": "failed"},
    ]
    summary = summarize_status_counts(rows)

    assert summary["P6"]["completed"] == 1
    assert summary["P6"]["unavailable"] == 0
    assert summary["P7"]["unavailable"] == 1
    assert summary["P7"]["failed"] == 1


def test_run_matrix_c_emits_p6_completed_and_p7_boundary_rows(monkeypatch, tmp_path) -> None:
    class _FakeProblem:
        def __init__(self, n_features: int):
            self.n_features = n_features
            self.n_bits = 2 * n_features

        def evaluate(self, x: int) -> float:
            return float((x % 17) + self.n_features)

    class _FakeDQIPipeline:
        def __init__(self, prob, *, k, ell, alpha_mode, execution_mode, decoder_mode):
            self.prob = prob
            self.k = k
            self.ell = ell
            self.alpha_mode = alpha_mode
            self.execution_mode = execution_mode
            self.decoder_mode = decoder_mode
            self.B = np.zeros((2, prob.n_bits), dtype=np.int8)
            self.v = np.zeros(2, dtype=np.int8)

        def run_with_metrics(self, n_samples: int, seed: int | None):
            del seed
            samples = np.arange(min(4, max(1, n_samples)), dtype=np.int64)
            f_values = np.linspace(10.0, 7.0, num=len(samples), dtype=np.float64)
            return {
                "F_values": f_values,
                "G_weighted_values": f_values + 1.0,
                "samples": samples,
                "best_F": float(np.max(f_values)),
                "best_G_weighted": float(np.max(f_values + 1.0)),
                "decoder_success_rate": 0.5,
                "success_prob": 1.0,
                "energy_fraction": 0.9,
            }

    def _fake_load_scaling_instance(n_features, instances_dir="data/instances"):
        del instances_dir
        return _FakeProblem(int(n_features)), False, tmp_path / f"pricing_{n_features}feat_{2*int(n_features)}bit.json"

    def _fake_resources(*args, **kwargs):
        del args, kwargs
        return {
            "decode_uncompute_model": "lookup_table_reversible",
            "decode_uncompute_status": "estimated",
            "decode_uncompute_confidence": "implementation_proxy",
            "total_gate_est_with_decode": 100.0,
            "total_depth_est_with_decode": 20.0,
            "total_qubits_est_with_decode": 31.0,
            "prefix_gate_est": 10.0,
            "prefix_depth_est": 3.0,
            "decode_model_gate_est": 5.0,
            "decode_model_depth_est": 2.0,
            "weight_register_qubits": 0.0,
            "coherent_weight_register_gate_est": 0.0,
            "coherent_weight_register_depth_est": 0.0,
        }

    monkeypatch.setattr("src.matrix_runner.load_scaling_instance", _fake_load_scaling_instance)
    monkeypatch.setattr("src.matrix_runner.DQIPipeline", _FakeDQIPipeline)
    monkeypatch.setattr("src.matrix_runner.dqi_resource_estimate", _fake_resources)

    out = run_matrix_c(
        output_root=tmp_path / "results",
        instance_ids=[],
        alpha_modes=["paper"],
        decoders=["bruteforce", "bp1"],
        p_features=[6, 7],
        include_p_family=True,
        n_dqi_samples=8,
        top_k_by_surrogate=4,
        seed=42,
    )
    rows = out["records"]
    p6_rows = [row for row in rows if row["instance_id"] == "P6"]
    p7_rows = [row for row in rows if row["instance_id"] == "P7"]

    assert p6_rows
    assert p7_rows
    assert all(
        {"run_id", "instance_id", "encoding", "decoder", "alpha_mode", "execution_mode", "ell"}.issubset(row)
        for row in p6_rows + p7_rows
    )
    assert all(row["run_status"] == "completed" for row in p6_rows)
    assert all(row["attempted_execution"] is True for row in p6_rows)
    assert all(row["run_status"] == "unavailable" for row in p7_rows)
    assert all(row["attempted_execution"] is False for row in p7_rows)
    assert all(row["boundary_reason"] == "qubit_count_exceeded" for row in p7_rows)
