"""Runtime contract tests for Notebook 10 fixed-baseline Stage D flow."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from notebooks._helpers import (
    ensure_numeric_columns,
    ensure_optional_columns,
    ensure_run_status_norm,
    parse_metric_availability,
    unavailable_panel_table,
)


def _find_code_cell(path: str, marker: str) -> str:
    payload = json.loads(Path(path).read_text())
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            return src
    raise AssertionError(f"could not find code cell marker `{marker}` in {path}")


def _make_display_capture() -> tuple[list[Any], Callable[..., None]]:
    records: list[Any] = []

    def _display(obj: Any = None, *_args: Any, **_kwargs: Any) -> None:
        records.append(obj)

    return records, _display


def _has_unavailable_with_reason(records: list[Any], reason: str) -> bool:
    for item in records:
        if isinstance(item, pd.DataFrame) and {"panel", "status", "reason"}.issubset(item.columns):
            if (item["reason"] == reason).any():
                return True
    return False


def test_missing_baseline_metadata_renders_baseline_metadata_unavailable() -> None:
    load_cell = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        'filter_mode = "primary_stage_family"',
    )
    baseline_guard = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        "required_baseline_fields = [",
    )

    records, display = _make_display_capture()
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(
            [
                {
                    "run_id": "r1",
                    "matrix": "matrix_b",
                    "stage": "alpha_validation",
                    "family": "C",
                    "instance_id": "C1",
                    "encoding": "wht_exact",
                    "decoder": "oracle",
                    "alpha_mode": "paper",
                    "execution_mode": "mixture",
                    "run_status": "completed",
                    "counterpart_key": "k1",
                },
            ]
        ),
        "parse_metric_availability": parse_metric_availability,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "ensure_numeric_columns": ensure_numeric_columns,
        "unavailable_panel_table": unavailable_panel_table,
    }

    exec(load_cell, env)
    exec(baseline_guard, env)

    assert _has_unavailable_with_reason(records, "baseline_metadata_unavailable")


def test_notebook10_uses_counterpart_key_only_for_pairing() -> None:
    src = json.loads(Path("notebooks/10_alpha_finite_size_validation.ipynb").read_text())
    code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in src.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert "counterpart_key" in code
    assert "pair_candidates =" not in code
