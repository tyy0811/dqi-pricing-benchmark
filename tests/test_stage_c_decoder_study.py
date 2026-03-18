"""Smoke tests for Stage C decoder-study raw runner."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, ".")

from src.decoder_study_runner import run_decoder_study


def test_decoder_study_writes_append_only_raw_runs(tmp_path: Path) -> None:
    out = run_decoder_study(
        output_root=tmp_path / "results",
        p_instances=("P3",),
        s_instances=("S_dense_low_collision",),
        alpha_modes=("paper",),
        decoders=("bruteforce", "bp1", "bp_osd_lite", "oracle"),
        n_dqi_samples=20,
        trial_seed=7,
    )
    assert out["row_count"] > 0

    runs_dir = tmp_path / "results" / "matrix_c" / "runs"
    run_files = sorted(runs_dir.glob("*.json"))
    assert run_files, "expected append-only raw run files"

    for run_file in run_files:
        payload = json.loads(run_file.read_text())
        assert payload["matrix"] == "matrix_c"
        assert payload["stage"] == "decoder_study"
        assert "trial_seed" in payload
        assert "seed" in payload
        assert "gain_over_bp1" not in payload
        assert "gain_over_bruteforce" not in payload


def test_zero_decodable_syndromes_marked_not_applicable(tmp_path: Path) -> None:
    """Regression test: instances with zero decodable syndromes skip DQI execution.

    S_sparse_high_collision has decoder_success_rate=0.0 for bruteforce/oracle.
    These should be marked not_applicable with proper reason codes, not failed.
    """
    out = run_decoder_study(
        output_root=tmp_path / "results",
        p_instances=(),
        s_instances=("S_sparse_high_collision",),
        alpha_modes=("paper",),
        decoders=("bruteforce", "bp1", "oracle"),
        n_dqi_samples=16,
        trial_seed=42,
    )
    assert out["row_count"] == 3  # bruteforce, bp1, oracle

    runs_dir = tmp_path / "results" / "matrix_c" / "runs"
    run_files = sorted(runs_dir.glob("*S_sparse_high_collision*.json"))
    assert len(run_files) == 3

    results_by_decoder = {}
    for run_file in run_files:
        payload = json.loads(run_file.read_text())
        decoder = payload["decoder"]
        results_by_decoder[decoder] = payload

    # Bruteforce: 0% success rate → not_applicable with structural boundary reason
    bf = results_by_decoder["bruteforce"]
    assert bf["run_status"] == "not_applicable"
    assert bf["status_reason_code"] == "dqi_impossible_zero_decodable_syndromes"
    assert "zero decodable syndromes" in bf["run_error"]
    assert "structural property" in bf["run_error"]
    assert bf["best_sampled_F"] is None
    assert bf["metric_availability"]["best_sampled_F"] == "not_applicable"
    assert bf["metric_availability"]["spearman_rho_F_G"] == "not_applicable"

    # BP1: ~1% success rate → attempts execution (may complete or fail, but not structural skip)
    bp1 = results_by_decoder["bp1"]
    assert bp1["run_status"] in ("completed", "failed")  # BP1 attempts despite low success rate
    assert bp1.get("status_reason_code") != "dqi_impossible_zero_decodable_syndromes"

    # Oracle: skipped for different reason (instance too large)
    oracle = results_by_decoder["oracle"]
    assert oracle["run_status"] == "not_applicable"
    assert oracle["status_reason_code"] == "oracle_tiny_cap_exceeded"  # Different reason than bruteforce
