"""Consistency smoke tests for exploratory notebooks 03 and 05."""

from __future__ import annotations

import json
from pathlib import Path


def _all_text(path: str) -> str:
    payload = json.loads(Path(path).read_text())
    parts = []
    for cell in payload.get("cells", []):
        parts.append("".join(cell.get("source", [])))
    return "\n\n".join(parts)


def test_notebook03_uses_repo_root_and_exploratory_labels() -> None:
    src = _all_text("notebooks/03_paper_faithful_dqi.ipynb")
    assert "repo_root" in src
    assert "sys.path.insert(" not in src
    assert "Exploratory scope note" in src
    assert "not a canonical reporting layer" in src
    assert "unavailable_panel_table" in src


def test_notebook05_uses_repo_root_prototype_labels_and_guards() -> None:
    src = _all_text("notebooks/05_decoder_comparison.ipynb")
    assert "repo_root" in src
    assert "sys.path.insert(" not in src
    assert "prototype" in src.lower()
    assert "distinct from canonical Stage C reporting" in src
    assert "raise FileNotFoundError" not in src
    assert "unavailable_panel_table" in src


def test_notebook05_keeps_bruteforce_and_bp1_paths() -> None:
    src = _all_text("notebooks/05_decoder_comparison.ipynb")
    assert "BoundedDistanceDecoder" in src
    assert "BP1Decoder" in src
