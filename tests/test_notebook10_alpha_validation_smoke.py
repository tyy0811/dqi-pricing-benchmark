"""Notebook 10 alpha appendix wiring smoke tests."""

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


def test_notebook10_uses_alpha_completeness_gate_and_compact_columns() -> None:
    src = _notebook_source("notebooks/10_alpha_finite_size_validation.ipynb")
    assert "has_complete_alpha_metrics" in src
    assert "top1_regret" in src
    assert "success_probability" in src
    assert "distribution_metric" in src


def test_notebook10_omits_incomplete_rows_from_main_appendix_table() -> None:
    src = _notebook_source("notebooks/10_alpha_finite_size_validation.ipynb")
    assert "apply(has_complete_alpha_metrics" in src
    assert "fillna(" not in src


def test_notebook10_describes_compact_appendix_role() -> None:
    markdown = _notebook_markdown("notebooks/10_alpha_finite_size_validation.ipynb").lower()
    assert "compact reviewer-facing appendix" in markdown
    assert "complete rows only" in markdown
    assert "not a deep validation notebook" in markdown
