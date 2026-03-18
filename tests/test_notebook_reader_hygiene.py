"""Hygiene checks for standardized notebook reader bootstrap and status usage."""

from __future__ import annotations

import json
import re
from pathlib import Path

TARGET_NOTEBOOKS = [
    Path("notebooks/06_resource_analysis.ipynb"),
    Path("notebooks/08_paired_encoding_comparison.ipynb"),
    Path("notebooks/09_decoder_phase_diagram.ipynb"),
    Path("notebooks/10_alpha_finite_size_validation.ipynb"),
    Path("notebooks/11_decision_faithfulness_dashboard.ipynb"),
    Path("notebooks/12_quality_vs_cost.ipynb"),
]


def _code_cells(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return ["".join(c.get("source", [])) for c in payload.get("cells", []) if c.get("cell_type") == "code"]


def _all_code(path: Path) -> str:
    return "\n\n".join(_code_cells(path))


def test_notebooks_have_no_sys_path_insert_or_cwd_root_probing() -> None:
    for nb in TARGET_NOTEBOOKS:
        src = _all_code(nb)
        assert "sys.path.insert(" not in src, nb
        assert "Path.cwd().resolve()" not in src, nb
        assert "ROOT = Path.cwd().resolve()" not in src, nb
        assert "if not (ROOT / \"src\").exists()" not in src, nb


def test_notebooks_bootstrap_via_repo_root() -> None:
    for nb in TARGET_NOTEBOOKS:
        first = _code_cells(nb)[0]
        assert "repo_root" in first, nb
        assert "ROOT = repo_root()" in first, nb


def test_notebooks_use_status_materialization_helper_not_raw_status_filters() -> None:
    raw_status_filter = re.compile(r"run_status\"\]\s*(==|!=|\.isin\()")
    for nb in TARGET_NOTEBOOKS:
        src = _all_code(nb)
        assert "ensure_run_status_norm(" in src, nb
        assert "normalize_run_status(" not in src, nb
        assert raw_status_filter.search(src) is None, nb


def test_notebook06_reader_only_no_artifact_writer_patterns() -> None:
    src = _all_code(Path("notebooks/06_resource_analysis.ipynb"))
    banned = [
        "resources_scaling.json",
        "export_resources_json",
        "build_surrogate_artifact",
        "small_instance",
        "default_instance",
    ]
    for token in banned:
        assert token not in src


def test_notebook06_no_same_model_to_cross_model_fallback_and_explicit_diagnostic_label() -> None:
    src = _all_code(Path("notebooks/06_resource_analysis.ipynb"))
    assert "plot_source = same_model if not same_model.empty else cross_model" not in src
    assert "gain_col = \"gain_over_bp1\" if not same_model.empty else \"gain_over_bruteforce\"" not in src
    assert "cross-model, non-authoritative, not used for claims" in src
