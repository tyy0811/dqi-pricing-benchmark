"""Notebook 09 decoder-phase wiring smoke tests."""

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


def test_notebook09_loader_handles_missing_or_null_stage_and_metric_availability() -> None:
    src = _notebook_source("notebooks/09_decoder_phase_diagram.ipynb")
    assert "stage_match_with_null_tolerance" in src
    assert "stage_present_no_decoder_values_using_matrix_family" in src
    assert '"metric_availability": "{}"' in src
    assert "metric_availability_dict" in src


def test_notebook09_uses_normalized_status_buckets() -> None:
    src = _notebook_source("notebooks/09_decoder_phase_diagram.ipynb")
    assert "run_status_norm" in src
    assert '"skipped"' not in src
    assert 'focus_for_status["run_status_norm"].isin(["not_applicable", "unavailable"])' in src


def test_notebook09_guards_structure_and_gain_fields_before_use() -> None:
    src = _notebook_source("notebooks/09_decoder_phase_diagram.ipynb")
    for token in [
        "structure_collision_rate",
        "structure_rank_deficiency",
        "structure_ambiguous_syndrome_count",
        "structure_short_cycle_proxy",
        "structure_distance_proxy",
        "gain_over_bp1",
        "gain_over_bruteforce",
        "best_sampled_F",
        "postselection_success",
    ]:
        assert token in src
