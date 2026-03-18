"""Static smoke tests for Notebook 13 scaling analysis."""

from __future__ import annotations

import json
from pathlib import Path


def _all_text(path: str) -> str:
    payload = json.loads(Path(path).read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))


def _markdown_text(path: str) -> str:
    payload = json.loads(Path(path).read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )


def test_notebook13_has_required_section_headers() -> None:
    markdown = _markdown_text("notebooks/13_scaling_analysis.ipynb")
    for header in [
        "## 1. Instance Summary: P3 through P7",
        "## 2. Encoding Quality vs. Scale",
        "## 3. Decoder Scaling (Exact Statevector Only)",
        "## 4. Resource Estimation (Analytic)",
        "## 5. Execution Timing and Tractability Boundary",
        "## 6. Scaling Verdicts",
        "## 7. Freeze Outputs",
    ]:
        assert header in markdown


def test_notebook13_states_provenance_split_and_canonical_plot_policy() -> None:
    text = _all_text("notebooks/13_scaling_analysis.ipynb")
    assert "execution coverage comes from the canonical master" in text
    assert "encoding diagnostics come from frozen paired artifacts / truth-table diagnostics" in text
    assert "canonical_m_value = 15" in text
    assert "diagnostic only" in text
    assert "P7 (m=20, diagnostic only)" in text


def test_notebook13_saves_plots_and_calls_out_boundary_rows() -> None:
    text = _all_text("notebooks/13_scaling_analysis.ipynb")
    assert 'ROOT / "results" / "plots"' in text
    assert "boundary_reason" in text
    assert "estimated_memory_gb" in text
    assert "generic unavailable coverage records without strong boundary provenance" in text or "validated executed coverage boundary at P7" in text


def test_notebook13_section5_states_execution_layer_not_stage_e() -> None:
    text = _all_text("notebooks/13_scaling_analysis.ipynb")
    assert "this section reports the executed pricing scaling slice" in text.lower()
    assert "it is not the stage e comparability slice used for reviewer-facing quality-vs-cost panels" in text.lower()
