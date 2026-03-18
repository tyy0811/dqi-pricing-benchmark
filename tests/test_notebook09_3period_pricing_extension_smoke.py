"""Notebook 09 3-period pricing extension smoke tests."""

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


def _notebook_markdown(path: str) -> str:
    payload = json.loads(Path(path).read_text())
    parts = []
    for cell in payload.get("cells", []):
        if cell.get("cell_type") == "markdown":
            parts.append("".join(cell.get("source", [])))
    return "\n\n".join(parts)


def test_notebook09_states_shared_segment_weight_shift_model() -> None:
    markdown = _notebook_markdown("notebooks/09_3period_pricing_extension.ipynb").lower()
    assert "shared segment" in markdown
    assert "period-varying demand" in markdown
    assert "weight shifts" in markdown


def test_notebook09_shows_static_vs_dynamic_comparison_table() -> None:
    src = _notebook_source("notebooks/09_3period_pricing_extension.ipynb")
    assert "build_multiperiod_scenario_matrix" in src
    assert "make_multiperiod_scenario_problem" in src
    assert "scenario_matrix_df" in src
    assert 'results" / "tables" / "3period_scenario_matrix.csv' in src
    assert "verdict_df" in src
    assert "comparison_df" in src
    assert '"scenario_label"' in src
    assert '"dynamic_revenue"' in src
    assert '"static_revenue"' in src
    assert '"delta"' in src
    assert '"inventory_used"' in src
    assert '"package_changed_across_periods"' in src
    assert '"verdict"' in src
    assert "instance_summary_df" in src
    assert "weight_shift_df" in src


def test_bmw_reviewer_summary_exists_with_scope_note() -> None:
    path = Path("docs/BMW_reviewer_summary.md")
    text = path.read_text().lower() if path.exists() else ""
    assert "pricing pipeline works" in text
    assert "beats wht" in text
    assert "decoder robustness matters" in text
    assert "positive dynamic-vs-static uplift" in text
    assert "structural subset parity/resource evidence" in text
    assert "not full decode-inclusive dqi parity" in text


def test_notebook09_concludes_dynamic_pricing_adjacent_only() -> None:
    markdown = _notebook_markdown("notebooks/09_3period_pricing_extension.ipynb").lower()
    assert "dynamic-pricing-adjacent" in markdown
    assert "shared horizon inventory" in markdown
    assert "no dqi integration" in markdown
    assert "selected for regime coverage and interpretability" in markdown
