"""Notebook 08 paired-encoding wiring smoke tests."""

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


def _code_cell(path: str, marker: str) -> str:
    payload = json.loads(Path(path).read_text())
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            return src
    raise AssertionError(f"could not find cell with marker {marker!r}")


def test_notebook08_prefers_master_with_legacy_matrix_a_fallback() -> None:
    src = _notebook_source("notebooks/08a_paired_encoding_comparison.ipynb")
    assert "load_metrics_master" in src
    assert "load_run_manifest" in src
    assert "load_matrix_summary" in src
    assert "load_matrix_report" in src
    assert "master_artifacts_available" in src


def test_notebook08_uses_normalized_status_and_no_raw_ok_filtering() -> None:
    src = _notebook_source("notebooks/08a_paired_encoding_comparison.ipynb")
    assert "ensure_run_status_norm(" in src
    assert 'run_status"] != "ok"' not in src
    assert 'run_status"] == "ok"' not in src
    assert 'focus["run_status_norm"] != "completed"' in src


def test_notebook08_prefers_canonical_comparison_key_pairing() -> None:
    cell = _code_cell("notebooks/08a_paired_encoding_comparison.ipynb", "pair_key_candidates = [")
    assert 'pair_keys = ["comparison_key"]' in cell
    assert "pairing_debug_summary_df" in cell
    assert '"slot_id"' not in cell
    assert 'run_status_norm"] == "completed"' in cell
    assert "No usable intersection pairing keys were found for WHT/ILP matching." in cell
    assert "unavailable_panel_table" in cell


def test_notebook08_pairing_contract_reports_audit_and_fallback_reason() -> None:
    cell = _code_cell("notebooks/08a_paired_encoding_comparison.ipynb", "pair_key_candidates = [")
    assert "pairing_audit_df" in cell
    assert "active_key_strategy" in cell
    assert "fallback_reason" in cell
    assert "dropped_unmatched_wht_rows" in cell
    assert "dropped_unmatched_ilp_rows" in cell
    assert "comparison_key_rejected_reason" in cell
    assert "comparison_key_missing_values" in cell
    assert "comparison_key_non_unique_within_candidate_slice" in cell


def test_notebook08_filters_to_pricing_mixture_completed_rows() -> None:
    src = _notebook_source("notebooks/08a_paired_encoding_comparison.ipynb")
    assert 'focus["family"].fillna("P").astype(str).eq("P")' in src
    assert 'focus["execution_mode_norm"] == "mixture"' in src
    assert 'focus["run_status_norm"] == "completed"' in src


def test_notebook08_includes_explicit_p7_exclusion_note() -> None:
    payload = json.loads(Path("notebooks/08a_paired_encoding_comparison.ipynb").read_text())
    text = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert "P7 (n=14) is excluded from this paired comparison" in text
    assert "current canonical master" in text
    assert "Notebook 13" in text


def test_notebook08b_contains_coherent_family_c_validation() -> None:
    src = _notebook_source("notebooks/08b_coherent_validation.ipynb")
    assert "family_c_df" in src
    assert "coherent" in src.lower()
