"""Notebook 04 pricing-pipeline wiring smoke tests."""

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


def test_notebook04_uses_repo_root_bootstrap_no_sys_path_insert() -> None:
    src = _notebook_source("notebooks/04_pricing_pipeline.ipynb")
    assert "repo_root" in src
    assert "sys.path.insert(" not in src


def test_notebook04_emits_paired_and_legacy_outputs() -> None:
    src = _notebook_source("notebooks/04_pricing_pipeline.ipynb")
    assert "solve_pricing_pair" in src
    assert 'ROOT / "results" / "paired_pricing"' in src
    assert 'ROOT / "data" / "instances"' in src
    assert "surrogate_artifact_to_dict(legacy_wht)" in src
    assert "paired_pricing_artifact_to_dict(paired)" in src


def test_notebook04_validates_both_encodings_and_keeps_cp_sat_bruteforce_section() -> None:
    src = _notebook_source("notebooks/04_pricing_pipeline.ipynb")
    assert 'set(loaded["encodings"].keys()) == {"wht_truncated", "ilp_derived"}' in src
    assert "REQUIRED_COMPARATIVE_SUBSET" in src
    assert "solve_pricing_with_cp_sat" in src
    assert "brute_force_solver" in src


def test_notebook04_points_to_separate_notebook09_extension() -> None:
    markdown = _notebook_markdown("notebooks/04_pricing_pipeline.ipynb").lower()
    assert "notebook 09" in markdown
    assert "3-period" in markdown
