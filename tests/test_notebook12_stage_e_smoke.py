"""Notebook 12 Stage E wiring smoke tests."""

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


def test_notebook12_reads_quality_cost_master_for_pareto() -> None:
    src = _notebook_source("notebooks/12_quality_vs_cost.ipynb")
    assert "load_quality_cost_master" in src
    assert "load_quality_cost_unavailable" in src
    assert "quality_master" in src
    assert "Pareto" in src


def test_notebook12_handles_empty_or_missing_quality_cost_master() -> None:
    src = _notebook_source("notebooks/12_quality_vs_cost.ipynb")
    assert "missing or empty" in src


def test_notebook12_no_matrix_summary_concat_or_smoke_dependency() -> None:
    src = _notebook_source("notebooks/12_quality_vs_cost.ipynb")
    assert "load_matrix_summary" not in src
    assert "load_matrix_report" not in src
    assert "matrix_a_summary" not in src
    assert "matrix_b_summary" not in src
    assert "matrix_c_summary" not in src
    assert "quality_vs_cost_smoke.json" not in src


def test_notebook12_no_identity_changing_master_join_ops() -> None:
    src = _notebook_source("notebooks/12_quality_vs_cost.ipynb")
    assert ".merge(" not in src
    assert ".concat(" not in src
    assert ".join(" not in src
    assert "groupby(" not in src


def test_notebook12_has_explicit_log_plot_positivity_guard() -> None:
    src = _notebook_source("notebooks/12_quality_vs_cost.ipynb")
    assert "No rows with positive gates; log-scale plot is unavailable." in src
    assert "Requires strictly positive `gates` values for log-scale plotting." in src
    assert "ax.set_xscale(\"log\")" in src


def test_notebook12_defaults_to_exact_reference_mainline_mixture_first() -> None:
    src = _notebook_source("notebooks/12_quality_vs_cost.ipynb")
    assert "EXACT_REFERENCE" in src
    assert "mixture" in src
    assert "bp1" in src
    assert "bruteforce" in src
    assert "coherent" in src
    assert "wht" in src
    assert "sort_values" in src


def test_notebook12_has_cumulative_exclusion_audit() -> None:
    src = _notebook_source("notebooks/12_quality_vs_cost.ipynb")
    # Check for cumulative audit structure
    assert "input_rows" in src
    assert "output_rows" in src
    assert "rows_removed" in src
    assert "reason_label" in src
    # Check for context panel
    assert "exclusion_reason" in src
    assert "Excluded Exact-Reference Rows" in src or "excluded_exact_ref" in src
