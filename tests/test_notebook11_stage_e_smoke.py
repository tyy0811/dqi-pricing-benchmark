"""Notebook 11 Stage E wiring smoke tests."""

from __future__ import annotations

import json
from pathlib import Path


def _notebook_source(path: str) -> str:
    payload = json.loads(Path(path).read_text())
    parts = []
    for cell in payload.get("cells", []):
        if cell.get("cell_type") == "code":
            parts.append("".join(cell.get("source", [])))
    return "\n\n".join(parts)


def test_notebook11_missing_or_empty_quality_cost_master_renders_placeholder() -> None:
    src = _notebook_source("notebooks/11_decision_faithfulness_dashboard.ipynb")
    assert "load_quality_cost_master" in src
    assert "missing or empty" in src
    assert "quality_cost_master" in src


def test_notebook11_no_matrix_summary_or_report_for_comparable_panels() -> None:
    src = _notebook_source("notebooks/11_decision_faithfulness_dashboard.ipynb")
    assert "load_matrix_summary" not in src
    assert "load_matrix_report" not in src
    assert "matrix_a_summary" not in src
    assert "matrix_b_summary" not in src
    assert "matrix_c_summary" not in src


def test_notebook11_comparable_source_is_quality_cost_master_only() -> None:
    src = _notebook_source("notebooks/11_decision_faithfulness_dashboard.ipynb")
    assert "comparable_df = load_quality_cost_master" in src
    assert "focus = comparable_df.copy()" in src
    assert "comparable panels are sourced only from quality_cost_master".lower() in src.lower()
    # Comparable slices should not be rebuilt from cross-source joins/concats.
    assert ".merge(" not in src
    assert ".concat(" not in src


def test_notebook11_transparency_uses_quality_cost_unavailable_only() -> None:
    src = _notebook_source("notebooks/11_decision_faithfulness_dashboard.ipynb")
    assert "quality_unavailable_df = load_quality_cost_unavailable" in src
    assert "Locked Stage E transparency card: driven only by quality_cost_unavailable." in src
    assert "panel=\"Stage E transparency\"" in src
