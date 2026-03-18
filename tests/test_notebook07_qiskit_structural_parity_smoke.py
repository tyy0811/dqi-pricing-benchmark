"""Notebook 07 structural subset parity smoke tests."""

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


def test_notebook07_handles_unavailable_qiskit_distribution_without_crashing() -> None:
    src = _notebook_source("notebooks/07_qiskit_structural_parity.ipynb")
    assert 'if qiskit_prob is None:' in src
    assert 'np.nan' in src
    assert 'Qiskit subset distribution unavailable.' in src


def test_notebook07_checks_distribution_shape_before_tvd() -> None:
    src = _notebook_source("notebooks/07_qiskit_structural_parity.ipynb")
    assert 'numpy_prob.shape != qiskit_prob.shape' in src
    assert 'Distribution shape mismatch:' in src


def test_notebook07_preserves_subset_only_scope_and_explicit_conclusion() -> None:
    payload = json.loads(Path("notebooks/07_qiskit_structural_parity.ipynb").read_text())
    markdown = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )
    assert "subset-only parity evidence" in markdown.lower()
    assert "validated tiny-case subset comparison" in markdown.lower()
    assert "family-s structural/resource context" in markdown.lower()
    assert "near-zero tvd applies only to this tiny case" in markdown.lower()
    assert "not a general parity validation result" in markdown.lower()
    assert "structural-subset-only validation" in markdown.lower()
    assert "decode/uncompute is excluded" in markdown.lower()
    assert "not full end-to-end parity evidence" in markdown.lower()


def test_notebook07_uses_canonical_subset_apis_and_diagnostic_tables() -> None:
    src = _notebook_source("notebooks/07_qiskit_structural_parity.ipynb")
    assert "run_qiskit_statevector" not in src
    assert "qiskit_subset_validation_report" in src
    assert "run_dqi_subset_reference" in src
    assert "qiskit_subset_resource_report" in src
    assert "side_by_side_distribution" in src
    assert "tiny_B" in src
    assert "tiny_comparison_df" in src
    assert "artifact_comparison_df" in src
    assert "tiny_case_note_df" in src
    assert "family_s_scope_df" in src
    assert 'subset_resource["scope"]["scope"]' in src
    assert 'subset_resource["scope"]["includes"]' in src
    assert 'subset_resource["scope"]["excludes"]' in src
    assert '"metric": "raw_tvd"' in src
    assert 'subset_resource["gate_counts"]' in src
    assert "subset parity passed" in src or "subset parity failed" in src
    assert "syndrome_register_qubit_order" in src
    assert "max_abs_diff" in src
    assert "reference_distribution_df" in src
    assert "qiskit_distribution_df" in src
