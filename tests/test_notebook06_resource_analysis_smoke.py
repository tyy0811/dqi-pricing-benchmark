"""Notebook 06 resource-analysis presentation smoke tests."""

from __future__ import annotations

import json
from pathlib import Path


def _notebook_markdown(path: str) -> str:
    payload = json.loads(Path(path).read_text())
    parts = []
    for cell in payload.get("cells", []):
        if cell.get("cell_type") == "markdown":
            parts.append("".join(cell.get("source", [])))
    return "\n\n".join(parts)


def test_notebook06_separates_analytic_pipeline_from_qiskit_subset() -> None:
    markdown = _notebook_markdown("notebooks/06_resource_analysis.ipynb").lower()
    assert "analytic full-pipeline estimates" in markdown
    assert "qiskit structural subset counts" in markdown
    assert "does not validate the full decode-inclusive pipeline" in markdown
    assert "decode-inclusive analytic estimate and qiskit structural subset counts are distinct validation scopes" in markdown


def test_notebook06_has_compact_scope_split_summary_table() -> None:
    payload = json.loads(Path("notebooks/06_resource_analysis.ipynb").read_text())
    code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert "summary_cols" in code
    assert '"analytic_scope"' in code
    assert '"qiskit_scope"' in code
    assert '"decode_included?"' in code or '"decode_included"' in code
    assert '"comparable?"' in code or '"comparable"' in code
