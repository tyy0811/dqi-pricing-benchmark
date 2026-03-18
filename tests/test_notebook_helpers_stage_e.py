"""Notebook helper loaders for Stage E artifacts."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, ".")

from notebooks._helpers import (
    load_quality_cost_master,
    load_quality_cost_unavailable,
    normalize_run_status,
)


def _has_parquet_engine() -> bool:
    return importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None


def test_load_quality_cost_master_prefers_parquet_fallback_json(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir(parents=True, exist_ok=True)

    (root / "quality_cost_master.json").write_text(json.dumps({"rows": [{"run_id": "b", "best_F": 2.0}]}, indent=2))
    if _has_parquet_engine():
        df = pd.DataFrame([{"run_id": "a", "best_F": 1.0}])
        df.to_parquet(root / "quality_cost_master.parquet", index=False)
        loaded = load_quality_cost_master(root)
        assert loaded.iloc[0]["run_id"] == "a"
        (root / "quality_cost_master.parquet").unlink()
    loaded_json = load_quality_cost_master(root)
    assert loaded_json.iloc[0]["run_id"] == "b"


def test_load_quality_cost_unavailable_prefers_parquet_fallback_json(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir(parents=True, exist_ok=True)

    (root / "quality_cost_unavailable.json").write_text(
        json.dumps({"rows": [{"run_id": "y", "unavailable_class": "duplicate_key"}]}, indent=2)
    )
    if _has_parquet_engine():
        df = pd.DataFrame([{"run_id": "x", "unavailable_class": "missing_quality_fields"}])
        df.to_parquet(root / "quality_cost_unavailable.parquet", index=False)
        loaded = load_quality_cost_unavailable(root)
        assert loaded.iloc[0]["run_id"] == "x"
        (root / "quality_cost_unavailable.parquet").unlink()
    loaded_json = load_quality_cost_unavailable(root)
    assert loaded_json.iloc[0]["run_id"] == "y"


def test_load_quality_cost_master_optional_missing_returns_empty(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir(parents=True, exist_ok=True)
    loaded = load_quality_cost_master(root, required=False)
    assert loaded.empty


def test_load_quality_cost_unavailable_optional_missing_returns_empty(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir(parents=True, exist_ok=True)
    loaded = load_quality_cost_unavailable(root, required=False)
    assert loaded.empty


def test_load_quality_cost_master_required_missing_raises(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir(parents=True, exist_ok=True)
    try:
        load_quality_cost_master(root, required=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError when required=True and master table is absent")


def test_load_quality_cost_unavailable_required_missing_raises(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir(parents=True, exist_ok=True)
    try:
        load_quality_cost_unavailable(root, required=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError when required=True and unavailable table is absent")


def test_normalize_run_status_maps_ok_to_completed() -> None:
    assert normalize_run_status("ok") == "completed"


def test_normalize_run_status_preserves_completed_and_not_in_experiment_matrix() -> None:
    values = pd.Series(["completed", "not_in_experiment_matrix"])
    normalized = normalize_run_status(values)
    assert normalized.tolist() == ["completed", "not_in_experiment_matrix"]


def test_normalize_run_status_maps_skipped_and_unsupported_to_not_applicable() -> None:
    values = pd.Series(["skipped", "unsupported"])
    normalized = normalize_run_status(values)
    assert normalized.tolist() == ["not_applicable", "not_applicable"]
