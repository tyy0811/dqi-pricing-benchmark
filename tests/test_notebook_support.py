"""Tests for notebook support utilities and feasibility diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, ".")

from src.problem_generator import default_instance, compute_feasibility_stats
from notebooks._helpers import (
    ensure_numeric_columns,
    ensure_optional_columns,
    ensure_run_status_norm,
    format_benchmark_row,
    format_benchmark_table,
    format_optimal_table,
    get_heuristic_dqi_report,
    load_metrics_master,
    load_run_manifest,
    normalize_run_status,
    repo_root,
    unavailable_panel_table,
    tvd,
)


def test_compute_feasibility_stats_fields():
    prob = default_instance()
    stats = compute_feasibility_stats(prob)

    expected = {
        "total_count",
        "valid_count",
        "valid_fraction",
        "invalid_tier_count",
        "bundle_violations",
        "luxury_cap_violations",
    }
    assert expected.issubset(stats.keys())


def test_compute_feasibility_stats_consistency():
    prob = default_instance()
    stats = compute_feasibility_stats(prob)

    assert stats["total_count"] == prob.n_configurations
    assert 0 <= stats["valid_count"] <= stats["total_count"]
    assert 0.0 <= stats["valid_fraction"] <= 1.0
    np.testing.assert_allclose(
        stats["valid_fraction"],
        stats["valid_count"] / stats["total_count"],
        rtol=1e-12,
    )
    for key in ["invalid_tier_count", "bundle_violations", "luxury_cap_violations"]:
        assert 0 <= stats[key] <= stats["total_count"]


def test_tvd_basic_properties():
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    np.testing.assert_allclose(tvd(p, q), 0.5, rtol=1e-12)
    np.testing.assert_allclose(tvd(p, p), 0.0, rtol=1e-12)


def test_format_optimal_table_schema():
    prob = default_instance()
    x_opt, _ = prob.brute_force_solve()
    table = format_optimal_table("5feat", prob, x_opt)

    assert list(table.columns) == ["instance", "feature", "tier", "price"]
    assert len(table) == prob.n_features


def test_get_heuristic_dqi_report_prefers_explicit_key():
    record = {
        "dqi_optimal": {"legacy": True},
        "dqi_heuristic_optimal": {"legacy": False},
    }
    report = get_heuristic_dqi_report(record)
    assert report["legacy"] is False


def test_get_heuristic_dqi_report_falls_back_to_legacy_key():
    record = {"dqi_optimal": {"approximation_ratio": 0.9}}
    report = get_heuristic_dqi_report(record)
    assert report["approximation_ratio"] == 0.9


def test_format_benchmark_row_includes_alpha_and_cp_sat_fields():
    record = {
        "problem": {"n_features": 3, "n_bits": 6},
        "parameters": {"k": 12, "ell": 2},
        "ground_truth": {"f_opt": 3000.0},
        "dqi_uniform": {"approximation_ratio": 1.0, "success_prob": 0.9, "m_constraints": 12},
        "dqi_heuristic_optimal": {"approximation_ratio": 0.95, "success_prob": 0.85},
        "alpha_diagnostics": {"alpha_l1_distance": 0.2, "alpha_l2_distance": 0.1, "alpha_effectively_equal": False},
        "baselines": {"cp_sat": {"status": "ok", "f": 3000.0, "matches_bruteforce_objective": True}},
    }
    row = format_benchmark_row(record)
    assert row["k"] == 12
    assert row["ell"] == 2
    assert row["m"] == 12
    assert row["alpha_l1_distance"] == 0.2
    assert row["alpha_l2_distance"] == 0.1
    assert row["cp_sat_status"] == "ok"
    assert row["cp_sat_matches_bruteforce_objective"] is True


def test_format_benchmark_table_schema():
    records = [
        {
            "problem": {"n_features": 3, "n_bits": 6},
            "parameters": {"k": 12, "ell": 2},
            "ground_truth": {"f_opt": 3000.0},
            "dqi_uniform": {"approximation_ratio": 1.0, "success_prob": 0.9, "m_constraints": 12},
            "dqi_optimal": {"approximation_ratio": 0.95, "success_prob": 0.85},
            "alpha_diagnostics": {"alpha_l1_distance": 0.2, "alpha_l2_distance": 0.1, "alpha_effectively_equal": False},
            "baselines": {"cp_sat": {"status": "ok", "f": 3000.0, "matches_bruteforce_objective": True}},
        }
    ]
    table = format_benchmark_table(records)
    assert len(table) == 1
    assert "dqi_heuristic_ratio" in table.columns


def test_load_metrics_master_optional_missing_returns_empty(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    df = load_metrics_master(root, required=False)
    assert df.empty


def test_load_run_manifest_optional_missing_returns_empty_dict(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    manifest = load_run_manifest(root, required=False)
    assert manifest == {}


def test_load_metrics_master_required_raises_when_missing(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    try:
        load_metrics_master(root, required=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError when required=True and no metrics file exists")


def test_load_run_manifest_required_raises_when_missing(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    try:
        load_run_manifest(root, required=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError when required=True and no manifest file exists")


def test_load_metrics_master_reads_json_rows_payload(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    payload = {"rows": [{"run_id": "r1", "run_status": "ok"}]}
    (root / "metrics_master.json").write_text(json.dumps(payload), encoding="utf-8")
    df = load_metrics_master(root, required=False)
    assert len(df) == 1
    assert df.loc[0, "run_status"] == "ok"


def test_normalize_run_status_accepts_scalar_and_series():
    assert normalize_run_status("ok") == "completed"
    assert normalize_run_status("completed") == "completed"
    assert normalize_run_status("failed") == "failed"
    assert normalize_run_status("unavailable") == "unavailable"
    assert normalize_run_status("not_in_experiment_matrix") == "not_in_experiment_matrix"
    assert normalize_run_status("skipped") == "not_applicable"
    assert normalize_run_status("unsupported") == "not_applicable"

    series = pd.Series(["ok", "completed", "failed", "unavailable", "not_in_experiment_matrix", "skipped"])
    normalized = normalize_run_status(series)
    assert normalized.tolist() == [
        "completed",
        "completed",
        "failed",
        "unavailable",
        "not_in_experiment_matrix",
        "not_applicable",
    ]


def test_ensure_numeric_columns_adds_missing_and_coerces():
    df = pd.DataFrame({"a": ["1.2", "x"], "b": [1, 2]})
    out = ensure_numeric_columns(df, ["a", "c"])
    assert "c" in out.columns
    assert out["a"].notna().sum() == 1
    assert out["a"].isna().sum() == 1
    assert out["c"].isna().all()


def test_ensure_optional_columns_materializes_numeric_and_object_fields():
    df = pd.DataFrame({"a": ["1.0", "x"]})
    out = ensure_optional_columns(
        df,
        numeric_cols=["a", "b"],
        object_defaults={"coherent_mode_record": np.nan, "metric_availability": "{}"},
    )
    assert set(["a", "b", "coherent_mode_record", "metric_availability"]).issubset(out.columns)
    assert out["a"].notna().sum() == 1
    assert out["b"].isna().all()
    assert out["coherent_mode_record"].isna().all()
    assert (out["metric_availability"] == "{}").all()


def test_ensure_run_status_norm_handles_missing_run_status_column():
    df = pd.DataFrame({"x": [1, 2]})
    out = ensure_run_status_norm(df)
    assert "run_status_norm" in out.columns
    assert out["run_status_norm"].tolist() == ["unavailable", "unavailable"]


def test_unavailable_panel_table_has_frozen_schema():
    panel = unavailable_panel_table("Panel A", "reason", "")
    assert list(panel.columns) == ["panel", "status", "reason", "details"]
    assert panel.loc[0, "status"] == "unavailable"


def test_repo_root_resolves_repo_with_required_markers():
    root = repo_root()
    assert (root / "src").is_dir()
    assert (root / "notebooks").is_dir()


def test_repo_root_error_lists_markers_and_visited_candidates(monkeypatch):
    repo_root.cache_clear()
    monkeypatch.setattr(
        "notebooks._helpers._candidate_root_paths",
        lambda: [Path("/tmp/nonexistent-a"), Path("/tmp/nonexistent-b")],
    )
    with pytest.raises(RuntimeError) as exc:
        repo_root()
    msg = str(exc.value)
    assert "Required markers" in msg
    assert "Optional markers" in msg
    assert "/tmp/nonexistent-a" in msg
    assert "/tmp/nonexistent-b" in msg
    repo_root.cache_clear()
