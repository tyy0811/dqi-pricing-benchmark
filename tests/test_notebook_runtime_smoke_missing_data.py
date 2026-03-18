"""Runtime-oriented smoke tests for analysis notebooks under missing/empty inputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

plt.switch_backend("Agg")

from notebooks._helpers import (
    normalize_run_status,
    ensure_run_status_norm,
    parse_metric_availability,
    placeholder_plot,
    unavailable_or_numeric,
    unavailable_panel_table,
    ensure_numeric_columns,
    ensure_optional_columns,
)


def _find_code_cell(path: str, marker: str) -> str:
    payload = json.loads(Path(path).read_text())
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            return src
    raise AssertionError(f"could not find code cell marker `{marker}` in {path}")


def _make_display_capture() -> tuple[list[Any], Callable[..., None]]:
    records: list[Any] = []

    def _display(obj: Any = None, *_args: Any, **_kwargs: Any) -> None:
        records.append(obj)

    return records, _display


def _has_unavailable_panel(records: list[Any], panel: str) -> bool:
    for item in records:
        if isinstance(item, pd.DataFrame) and {"panel", "status", "reason"}.issubset(item.columns):
            if (item["panel"] == panel).any():
                return True
    return False


@pytest.fixture(params=["ok", "completed"], ids=["old_ok", "new_completed"])
def status_variant(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_normalize_run_status_variant_to_completed(status_variant: str) -> None:
    out_scalar = normalize_run_status(status_variant)
    out_series = normalize_run_status(pd.Series([status_variant])).iloc[0]
    assert out_scalar == "completed"
    assert out_series == "completed"


def test_normalize_run_status_preserves_canonical_labels() -> None:
    normalized = normalize_run_status(
        pd.Series(["completed", "failed", "not_in_experiment_matrix"])
    )
    assert normalized.tolist() == ["completed", "failed", "not_in_experiment_matrix"]


def test_notebook10_empty_stage_d_slice_renders_placeholder_no_crash() -> None:
    load_cell = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        'filter_mode = "primary_stage_family"',
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(
            [
                {
                    "matrix": "matrix_a",
                    "stage": "benchmark_matrix",
                    "family": "P",
                    "run_status": "completed",
                    "execution_mode": "mixture",
                }
            ]
        ),
        "parse_metric_availability": parse_metric_availability,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "ensure_numeric_columns": ensure_numeric_columns,
        "unavailable_panel_table": unavailable_panel_table,
    }

    exec(load_cell, env)

    assert "focus" in env and env["focus"].empty
    assert _has_unavailable_panel(records, "Alpha finite-size focus")


def test_notebook10_missing_optional_coherent_columns_do_not_crash() -> None:
    load_cell = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        'filter_mode = "primary_stage_family"',
    )
    coherent_panel = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        'panel="Coherent coverage"',
    )
    delta_panel = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        "delta_cols = [",
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "matrix_b_report": {},
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(
            [
                {
                    "run_id": "r_mix",
                    "matrix": "matrix_b",
                    "stage": "alpha_validation",
                    "family": "C",
                    "instance_id": "C1",
                    "encoding": "wht_exact",
                    "decoder": "bp1",
                    "alpha_mode": "a0",
                    "execution_mode": "mixture",
                    "run_status": "completed",
                    "best_F": 0.9,
                },
                {
                    "run_id": "r_coh",
                    "matrix": "matrix_b",
                    "stage": "alpha_validation",
                    "family": "C",
                    "instance_id": "C1",
                    "encoding": "wht_exact",
                    "decoder": "bp1",
                    "alpha_mode": "a0",
                    "execution_mode": "coherent",
                    "run_status": "completed",
                    "best_F": 0.91,
                },
            ]
        ),
        "parse_metric_availability": parse_metric_availability,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "ensure_numeric_columns": ensure_numeric_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)
    exec(coherent_panel, env)
    exec(delta_panel, env)

    assert "coherent_mode_record" in env["focus"].columns
    for col in [
        "coherent_vs_mixture_delta_best_F",
        "coherent_vs_mixture_delta_best_top_G_sampled_F",
        "coherent_vs_mixture_delta_postselection_success",
        "coherent_vs_mixture_distribution_distance",
    ]:
        assert col in env["focus"].columns
    assert "delta_panel" in env
    assert isinstance(env["delta_panel"], pd.DataFrame)
    # Either summary tables or unavailable panels are acceptable as long as execution does not crash.
    assert records


def test_notebook10_mixed_ok_completed_statuses_produce_completed_rows() -> None:
    load_cell = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        'filter_mode = "primary_stage_family"',
    )
    alpha_panel = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        'panel="Alpha comparison"',
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(
            [
                {
                    "run_id": "r_old",
                    "matrix": "matrix_b",
                    "stage": "alpha_validation",
                    "family": "C",
                    "instance_id": "C1",
                    "encoding": "wht_exact",
                    "decoder": "bp1",
                    "alpha_mode": "a0",
                    "execution_mode": "mixture",
                    "run_status": "ok",
                    "best_F": 0.90,
                    "postselection_success": 0.20,
                },
                {
                    "run_id": "r_new",
                    "matrix": "matrix_b",
                    "stage": "alpha_validation",
                    "family": "C",
                    "instance_id": "C1",
                    "encoding": "wht_exact",
                    "decoder": "bp1",
                    "alpha_mode": "a1",
                    "execution_mode": "mixture",
                    "run_status": "completed",
                    "best_F": 0.92,
                    "postselection_success": 0.25,
                },
            ]
        ),
        "parse_metric_availability": parse_metric_availability,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "ensure_numeric_columns": ensure_numeric_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    orig_show = plt.show
    try:
        plt.show = lambda *_a, **_k: None
        exec(load_cell, env)
        exec(alpha_panel, env)
    finally:
        plt.show = orig_show

    assert "focus" in env and not env["focus"].empty
    assert env["focus"]["run_status_norm"].eq("completed").all()
    assert "ok_rows" in env
    assert len(env["ok_rows"]) == 2
    assert records


def test_notebook10_distributional_summary_renders_when_fields_present() -> None:
    load_cell = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        'filter_mode = "primary_stage_family"',
    )
    alpha_panel = _find_code_cell(
        "notebooks/10_alpha_finite_size_validation.ipynb",
        'panel="Alpha comparison"',
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(
            [
                {
                    "run_id": "r_uniform",
                    "matrix": "matrix_b",
                    "stage": "alpha_validation",
                    "family": "C",
                    "instance_id": "C1",
                    "encoding": "wht_exact",
                    "decoder": "oracle",
                    "alpha_mode": "uniform",
                    "execution_mode": "mixture",
                    "run_status": "completed",
                    "best_sampled_F": 0.9,
                    "postselection_success": 0.11,
                    "candidate_entropy": 0.91,
                    "candidate_top1_probability": 0.40,
                    "candidate_top2_probability": 0.24,
                    "ell": 1,
                    "m_active": 4,
                },
                {
                    "run_id": "r_paper",
                    "matrix": "matrix_b",
                    "stage": "alpha_validation",
                    "family": "C",
                    "instance_id": "C1",
                    "encoding": "wht_exact",
                    "decoder": "oracle",
                    "alpha_mode": "paper",
                    "execution_mode": "mixture",
                    "run_status": "completed",
                    "best_sampled_F": 0.9,
                    "postselection_success": 0.12,
                    "candidate_entropy": 0.88,
                    "candidate_top1_probability": 0.43,
                    "candidate_top2_probability": 0.19,
                    "ell": 1,
                    "m_active": 4,
                },
                {
                    "run_id": "r_heuristic",
                    "matrix": "matrix_b",
                    "stage": "alpha_validation",
                    "family": "C",
                    "instance_id": "C1",
                    "encoding": "wht_exact",
                    "decoder": "oracle",
                    "alpha_mode": "heuristic",
                    "execution_mode": "mixture",
                    "run_status": "completed",
                    "best_sampled_F": 0.9,
                    "postselection_success": 0.13,
                    "candidate_entropy": 0.93,
                    "candidate_top1_probability": 0.33,
                    "candidate_top2_probability": 0.29,
                    "ell": 1,
                    "m_active": 4,
                },
            ]
        ),
        "parse_metric_availability": parse_metric_availability,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "ensure_numeric_columns": ensure_numeric_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    orig_show = plt.show
    try:
        plt.show = lambda *_a, **_k: None
        exec(load_cell, env)
        exec(alpha_panel, env)
    finally:
        plt.show = orig_show

    assert "distributional_summary" in env
    summary = env["distributional_summary"]
    assert isinstance(summary, pd.DataFrame)
    assert not summary.empty
    for col in [
        "mean_candidate_entropy",
        "mean_candidate_top1_probability",
        "mean_candidate_top2_probability",
    ]:
        assert col in summary.columns
    assert summary[["mean_candidate_entropy"]].notna().any().any()


def test_notebook09_loader_metrics_present_matrix_c_absent_no_crash() -> None:
    loader = _find_code_cell(
        "notebooks/09_decoder_phase_diagram.ipynb",
        'source_details = {',
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(
            [
                {
                    "matrix": "matrix_a",
                    "stage": "decoder_study",
                    "family": "P",
                    "execution_mode": "mixture",
                    "run_status": "completed",
                }
            ]
        ),
        "load_matrix_summary": lambda *_a, **_k: pd.DataFrame(),
        "load_matrix_report": lambda *_a, **_k: {},
        "parse_metric_availability": parse_metric_availability,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "unavailable_panel_table": unavailable_panel_table,
    }

    exec(loader, env)

    assert "focus" in env
    assert env["focus"].empty
    assert _has_unavailable_panel(records, "Decoder phase diagram input")


def test_notebook09_loader_metrics_empty_no_crash() -> None:
    loader = _find_code_cell(
        "notebooks/09_decoder_phase_diagram.ipynb",
        'source_details = {',
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(),
        "load_matrix_summary": lambda *_a, **_k: pd.DataFrame(),
        "load_matrix_report": lambda *_a, **_k: {},
        "parse_metric_availability": parse_metric_availability,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "unavailable_panel_table": unavailable_panel_table,
    }

    exec(loader, env)

    assert "focus" in env
    assert env["focus"].empty
    assert _has_unavailable_panel(records, "Decoder phase diagram input")


def test_notebook09_loader_stage_present_but_null_keeps_rows_and_normalizes_status() -> None:
    loader = _find_code_cell(
        "notebooks/09_decoder_phase_diagram.ipynb",
        'source_details = {',
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(
            [
                {
                    "matrix": "matrix_c",
                    "stage": np.nan,
                    "family": "P",
                    "decoder": "bp1",
                    "execution_mode": "mixture",
                    "run_status": "ok",
                    "structure_collision_rate": 0.2,
                    "best_sampled_F": 0.9,
                }
            ]
        ),
        "load_matrix_summary": lambda *_a, **_k: pd.DataFrame(),
        "load_matrix_report": lambda *_a, **_k: {},
        "parse_metric_availability": parse_metric_availability,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "unavailable_panel_table": unavailable_panel_table,
    }

    exec(loader, env)

    assert "focus" in env
    assert not env["focus"].empty
    assert env["focus"]["run_status_norm"].eq("completed").all()
    assert "metric_availability_dict" in env["focus"].columns
    assert env["focus"]["metric_availability_dict"].apply(lambda x: isinstance(x, dict)).all()
    assert not _has_unavailable_panel(records, "Decoder phase diagram input")


def test_notebook09_missing_structure_gain_columns_inserted_as_nan() -> None:
    structure_panel = _find_code_cell(
        "notebooks/09_decoder_phase_diagram.ipynb",
        "required_numeric_cols = [",
    )
    corr_panel = _find_code_cell(
        "notebooks/09_decoder_phase_diagram.ipynb",
        "corr_targets = [",
    )
    records, display = _make_display_capture()

    focus = pd.DataFrame(
        {
            "run_status_norm": ["completed", "completed", "failed"],
            "decoder": ["bp1", "oracle", "bp1"],
            "run_id": ["r1", "r2", "r3"],
        }
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "display": display,
        "focus": focus,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "ensure_numeric_columns": ensure_numeric_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
        "unavailable_or_numeric": unavailable_or_numeric,
    }

    exec(structure_panel, env)
    exec(corr_panel, env)

    ok_rows = env["ok_rows"]
    for col in [
        "structure_collision_rate",
        "structure_rank_deficiency",
        "gain_over_bp1",
        "gain_over_bruteforce",
    ]:
        assert col in ok_rows.columns
        assert ok_rows[col].isna().all()
    assert "corr_df" in env
    assert isinstance(env["corr_df"], pd.DataFrame)
    rendered_structure_panel = any(
        isinstance(item, pd.DataFrame) and ("structure_regime" in item.columns)
        for item in records
    )
    assert rendered_structure_panel or _has_unavailable_panel(records, "Structure regime comparison")


def test_notebook08_loader_uses_legacy_fallback_when_master_unavailable(tmp_path: Path) -> None:
    load_cell = _find_code_cell(
        "notebooks/08_paired_encoding_comparison.ipynb",
        "run_manifest = load_run_manifest",
    )
    records, display = _make_display_capture()

    legacy_summary = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "matrix": "matrix_a",
                "instance_id": "P3",
                "family": "P",
                "stage": "benchmark_matrix",
                "encoding": "wht_truncated",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "run_status": "ok",
                "best_sampled_F": 0.9,
            },
            {
                "run_id": "r2",
                "matrix": "matrix_a",
                "instance_id": "P3",
                "family": "P",
                "stage": "benchmark_matrix",
                "encoding": "ilp_derived",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "run_status": "completed",
                "best_sampled_F": 0.91,
            },
        ]
    )

    (tmp_path / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "tables" / "matrix_a_summary.csv").write_text("run_id\n", encoding="utf-8")

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "ROOT": tmp_path,
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {"slots": []},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(),
        "load_matrix_summary": lambda *_a, **_k: legacy_summary.copy(),
        "load_matrix_report": lambda *_a, **_k: {},
        "parse_metric_availability": parse_metric_availability,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)

    assert "focus" in env
    assert not env["focus"].empty
    assert env["focus"]["run_status_norm"].isin(["completed"]).all()
    assert env["selection_mode"] in {
        "stage_from_source",
        "stage_family_from_source",
        "legacy_manifest_scope_fallback",
        "legacy_fixed_instance_fallback",
        "legacy_instance_fallback",
        "stage_family_from_metrics_master",
        "stage_family_manifest_aligned",
    }


def test_notebook08_dynamic_pairing_keys_handle_missing_alpha_mode() -> None:
    match_cell = _find_code_cell(
        "notebooks/08_paired_encoding_comparison.ipynb",
        "pair_key_candidates = [",
    )
    records, display = _make_display_capture()

    focus = pd.DataFrame(
        [
            {
                "instance_id": "P3",
                "decoder": "bp1",
                "encoding": "wht_truncated",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
            },
            {
                "instance_id": "P3",
                "decoder": "bp1",
                "encoding": "ilp_derived",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
            },
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "focus": focus,
        "ensure_run_status_norm": ensure_run_status_norm,
        "unavailable_panel_table": unavailable_panel_table,
    }

    exec(match_cell, env)

    assert "matched_keys" in env
    assert not env["matched_keys"].empty
    assert "pair_keys" in env and env["pair_keys"]
    assert not _has_unavailable_panel(records, "Matched coverage")


def test_notebook08_uses_comparison_key_pairs_when_available() -> None:
    load_cell = _find_code_cell(
        "notebooks/08_paired_encoding_comparison.ipynb",
        "run_manifest = load_run_manifest",
    )
    match_cell = _find_code_cell(
        "notebooks/08_paired_encoding_comparison.ipynb",
        "pair_key_candidates = [",
    )
    metric_cell = _find_code_cell(
        "notebooks/08_paired_encoding_comparison.ipynb",
        "metric_panel = [",
    )
    delta_cell = _find_code_cell(
        "notebooks/08_paired_encoding_comparison.ipynb",
        'if mix.empty or "encoding" not in mix.columns:',
    )
    records, display = _make_display_capture()

    metrics_master = pd.DataFrame(
        [
            {
                "run_id": "wht_row",
                "matrix": "matrix_a",
                "stage": "benchmark_matrix",
                "family": "P",
                "instance_id": "P3",
                "encoding": "wht_truncated",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "comparison_key": "ck-1",
                "run_status": "completed",
                "best_sampled_F": 1.0,
                "top1_regret": 0.2,
                "topk_regret": 0.3,
                "best_top_G_sampled_F": 1.2,
                "spearman_rho_F_G": 0.8,
            },
            {
                "run_id": "ilp_row",
                "matrix": "matrix_a",
                "stage": "benchmark_matrix",
                "family": "P",
                "instance_id": "P3",
                "encoding": "ilp_derived",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "comparison_key": "ck-1",
                "run_status": "completed",
                "best_sampled_F": 1.5,
                "top1_regret": 0.1,
                "topk_regret": 0.15,
                "best_top_G_sampled_F": 1.6,
                "spearman_rho_F_G": 0.9,
            },
        ]
    )

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {"slots": []},
        "load_metrics_master": lambda *_a, **_k: metrics_master.copy(),
        "load_matrix_summary": lambda *_a, **_k: pd.DataFrame(),
        "load_matrix_report": lambda *_a, **_k: {},
        "parse_metric_availability": parse_metric_availability,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_numeric_columns": ensure_numeric_columns,
        "ensure_optional_columns": ensure_optional_columns,
        "unavailable_or_numeric": unavailable_or_numeric,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
        "plt": plt,
    }

    exec(load_cell, env)
    exec(match_cell, env)
    exec(metric_cell, env)
    exec(delta_cell, env)

    assert env["pair_keys"] == ["comparison_key"]
    assert "pairing_debug_summary_df" in env
    assert int(env["pairing_debug_summary_df"]["matched_pairs"].iloc[0]) == 1
    assert "delta_df" in env and not env["delta_df"].empty
    assert env["delta_df"]["paired_n"].max() > 0
    assert not _has_unavailable_panel(records, "Paired delta table")


def test_notebook08_rejects_duplicate_comparison_key_and_skips_pairing() -> None:
    match_cell = _find_code_cell(
        "notebooks/08_paired_encoding_comparison.ipynb",
        "pair_key_candidates = [",
    )
    records, display = _make_display_capture()

    focus = pd.DataFrame(
        [
            {
                "instance_id": "P6",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
                "family": "P",
                "stage": "benchmark_matrix",
                "ell": 2,
                "encoding": "wht_truncated",
                "comparison_key": "ck-dup",
            },
            {
                "instance_id": "P6",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
                "family": "P",
                "stage": "benchmark_matrix",
                "ell": 2,
                "encoding": "wht_truncated",
                "comparison_key": "ck-dup",
            },
            {
                "instance_id": "P6",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
                "family": "P",
                "stage": "benchmark_matrix",
                "ell": 2,
                "encoding": "ilp_derived",
                "comparison_key": "ck-dup",
            },
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "focus": focus,
        "ensure_run_status_norm": ensure_run_status_norm,
        "unavailable_panel_table": unavailable_panel_table,
    }

    exec(match_cell, env)

    assert env["pairing_unavailable_reason"] == "Duplicate completed WHT or ILP rows found for at least one final pairing key."
    assert env["pair_keys"]
    assert env["pairing_audit_df"]["active_key_strategy"].iloc[0] == "fallback_composite_key"
    assert env["pairing_audit_df"]["fallback_reason"].iloc[0] == "comparison_key_non_unique_within_candidate_slice"
    assert _has_unavailable_panel(records, "Matched coverage")


def test_notebook08_falls_back_when_comparison_key_is_partially_missing() -> None:
    match_cell = _find_code_cell(
        "notebooks/08_paired_encoding_comparison.ipynb",
        "pair_key_candidates = [",
    )
    records, display = _make_display_capture()

    focus = pd.DataFrame(
        [
            {
                "instance_id": "P6",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
                "family": "P",
                "stage": "benchmark_matrix",
                "ell": 2,
                "encoding": "wht_truncated",
                "comparison_key": np.nan,
            },
            {
                "instance_id": "P6",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
                "family": "P",
                "stage": "benchmark_matrix",
                "ell": 2,
                "encoding": "ilp_derived",
                "comparison_key": "ck-1",
            },
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "focus": focus,
        "ensure_run_status_norm": ensure_run_status_norm,
        "unavailable_panel_table": unavailable_panel_table,
    }

    exec(match_cell, env)

    assert env["pair_keys"] == [
        "instance_id",
        "decoder",
        "alpha_mode",
        "execution_mode_norm",
        "ell",
        "stage",
        "family",
    ]
    assert env["pairing_audit_df"]["active_key_strategy"].iloc[0] == "fallback_composite_key"
    assert env["pairing_audit_df"]["fallback_reason"].iloc[0] == "comparison_key_missing_values"
    assert int(env["pairing_audit_df"]["matched_pairs"].iloc[0]) == 1
    assert not _has_unavailable_panel(records, "Matched coverage")


def test_notebook08_pairing_audit_shows_p7_unmatched_rows_when_no_completed_match_exists() -> None:
    match_cell = _find_code_cell(
        "notebooks/08_paired_encoding_comparison.ipynb",
        "pair_key_candidates = [",
    )
    records, display = _make_display_capture()

    focus = pd.DataFrame(
        [
            {
                "instance_id": "P6",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
                "family": "P",
                "stage": "benchmark_matrix",
                "ell": 2,
                "encoding": "wht_truncated",
                "comparison_key": "ck-p6",
            },
            {
                "instance_id": "P6",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
                "family": "P",
                "stage": "benchmark_matrix",
                "ell": 2,
                "encoding": "ilp_derived",
                "comparison_key": "ck-p6",
            },
            {
                "instance_id": "P7",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
                "family": "P",
                "stage": "benchmark_matrix",
                "ell": 2,
                "encoding": "wht_truncated",
                "comparison_key": "ck-p7",
            },
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "focus": focus,
        "ensure_run_status_norm": ensure_run_status_norm,
        "unavailable_panel_table": unavailable_panel_table,
    }

    exec(match_cell, env)

    assert int(env["pairing_audit_df"]["matched_pairs"].iloc[0]) == 1
    assert int(env["pairing_audit_df"]["dropped_unmatched_wht_rows"].iloc[0]) == 1
    assert int(env["pairing_audit_df"]["dropped_unmatched_ilp_rows"].iloc[0]) == 0
    assert not _has_unavailable_panel(records, "Matched coverage")


def test_notebook13_section1_preserves_frozen_cp_sat_and_fills_missing_fields() -> None:
    section1_cell = _find_code_cell(
        "notebooks/13_scaling_analysis.ipynb",
        "section1_rows = []",
    )
    records, display = _make_display_capture()

    instance_payloads = {
        "P3": {
            "features": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "solution": {"f_opt": 3000.0},
            "statistics": {"n_feasible": 23, "n_configurations": 64},
            "cp_sat": {
                "status": "optimal",
                "time": 0.01,
                "model_type": "frozen",
                "n_variables": 64,
            },
        },
        "P6": {
            "features": [{"name": f"F{i}"} for i in range(6)],
            "solution": {"f_opt": 3360.0},
            "statistics": {"n_feasible": 520, "n_configurations": 4096},
        },
    }
    fallback_calls: list[str] = []

    def fake_cp_sat_fallback(instance_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        fallback_calls.append(instance_id)
        return {
            "status": "optimal",
            "time": 1.25,
            "model_type": "ortools_cp_sat_one_hot_exact",
            "n_variables": payload["statistics"]["n_configurations"],
        }

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "Markdown": lambda text: text,
        "instance_payloads": instance_payloads,
        "cp_sat_fallback": fake_cp_sat_fallback,
    }

    exec(section1_cell, env)

    assert "section1_df" in env
    assert set(env["section1_df"]["Instance"]) == {"P3", "P6"}
    p3 = env["section1_df"][env["section1_df"]["Instance"] == "P3"].iloc[0]
    p6 = env["section1_df"][env["section1_df"]["Instance"] == "P6"].iloc[0]
    assert p3["CP-SAT time (s)"] == 0.01
    assert p3["Model type"] == "frozen"
    assert p6["CP-SAT time (s)"] == 1.25
    assert fallback_calls == ["P6"]


def test_notebook13_section2_prefers_paired_report_metrics_and_marks_provenance(tmp_path: Path) -> None:
    section2_cell = _find_code_cell(
        "notebooks/13_scaling_analysis.ipynb",
        "def _instance_slug(instance_id: str) -> str:",
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "display": display,
        "Markdown": lambda text: text,
        "plots_dir": tmp_path,
        "canonical_m_value": 15,
        "section1_df": pd.DataFrame(
            [{"Instance": "P3", "n (bits)": 6, "Features": 3}]
        ),
        "paired_artifacts": {
            "pricing_3feat_6bit": {
                "backend_a_name": "wht_truncated",
                "backend_b_name": "ilp_derived",
                "backend_a_metrics": {
                    "faithfulness": {
                        "spearman_rho_F_G": 0.42,
                        "retained_energy_eta": 0.71,
                    },
                    "core_decision": {
                        "decision_distortion": 0.19,
                        "top1_regret": 0.11,
                        "topk_regret": 0.16,
                    },
                },
                "backend_b_metrics": {
                    "faithfulness": {
                        "spearman_rho_F_G": 0.95,
                        "retained_energy_eta": 1.0,
                    },
                    "core_decision": {
                        "decision_distortion": 0.0,
                        "top1_regret": 0.0,
                        "topk_regret": 0.0,
                    },
                },
                "summary": {
                    "winner": "mainline_exact_reference",
                    "reviewer_verdict": "ilp_derived_exact",
                },
            }
        },
        "loaded_probs": {"P3": None},
        "p7_diag_artifact_m20": None,
    }

    orig_show = plt.show
    try:
        plt.show = lambda *_a, **_k: None
        exec(section2_cell, env)
    finally:
        plt.show = orig_show

    assert "section2_df" in env
    section2_df = env["section2_df"]
    assert set(section2_df["Encoding"]) == {"wht_truncated", "ilp_derived"}
    assert section2_df["Spearman rho(F,G)"].notna().all()
    assert section2_df["Metric provenance"].eq("paired_report").all()
    assert section2_df["Availability"].eq("available").all()


def test_notebook13_section2_truth_table_fallback_flattens_nested_decision_metrics(tmp_path: Path) -> None:
    from notebooks._helpers import load_instance
    from src.encoding_compare import _evaluate_g_over_full_domain, _select_candidates_by_g
    from src.metrics_decision import compute_decision_metrics
    from src.pricing_ilp import solve_pricing_pair

    section2_cell = _find_code_cell(
        "notebooks/13_scaling_analysis.ipynb",
        "def _instance_slug(instance_id: str) -> str:",
    )
    records, display = _make_display_capture()
    prob = load_instance(Path("data/instances/pricing_3feat_6bit.json"))

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "display": display,
        "Markdown": lambda text: text,
        "plots_dir": tmp_path,
        "canonical_m_value": 15,
        "section1_df": pd.DataFrame(
            [{"Instance": "P3", "n (bits)": 6, "Features": 3}]
        ),
        "paired_artifacts": {
            "pricing_3feat_6bit": solve_pricing_pair(
                prob,
                instance_id="pricing_3feat_6bit",
                k=15,
                source_metadata={
                    "instance_id": "pricing_3feat_6bit",
                    "pricing_instance_path": "data/instances/pricing_3feat_6bit.json",
                },
            )
        },
        "loaded_probs": {"P3": prob},
        "p7_diag_artifact_m20": None,
        "_evaluate_g_over_full_domain": _evaluate_g_over_full_domain,
        "_select_candidates_by_g": _select_candidates_by_g,
        "compute_decision_metrics": compute_decision_metrics,
    }

    orig_show = plt.show
    try:
        plt.show = lambda *_a, **_k: None
        exec(section2_cell, env)
    finally:
        plt.show = orig_show

    section2_df = env["section2_df"]
    assert section2_df["Metric provenance"].eq("truth_table_fallback").all()
    assert section2_df["Availability"].eq("available").all()
    assert section2_df["Spearman rho(F,G)"].notna().all()
    assert section2_df["Retained energy eta"].notna().all()
    assert section2_df["Top-1 regret"].notna().all()


def test_notebook13_section3_uses_wall_clock_placeholder_and_coverage_note() -> None:
    section3_cell = _find_code_cell(
        "notebooks/13_scaling_analysis.ipynb",
        "section3_candidates = pd.DataFrame()",
    )
    records, display = _make_display_capture()

    metrics_master = pd.DataFrame(
        [
            {
                "instance_id": "P6",
                "family": "P",
                "run_status_norm": "completed",
                "decoder": "bp1",
                "ell": 2,
                "postselection_success": 0.25,
                "best_sampled_F": 3200.0,
                "top1_regret": 0.0,
                "wall_clock_s": np.nan,
            },
            {
                "instance_id": "P6",
                "family": "P",
                "run_status_norm": "completed",
                "decoder": "bruteforce",
                "ell": 2,
                "postselection_success": 1.0,
                "best_sampled_F": 3360.0,
                "top1_regret": 0.0,
                "wall_clock_s": 12.5,
            },
            {
                "instance_id": "P7",
                "family": "P",
                "run_status_norm": "unavailable",
                "decoder": "bp1",
                "ell": 2,
            },
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "Markdown": lambda text: text,
        "metrics_master_df": metrics_master,
        "instance_summary_df": pd.DataFrame(
            [{"Instance": "P6", "n (bits)": 12}, {"Instance": "P7", "n (bits)": 14}]
        ),
    }

    exec(section3_cell, env)

    assert "section3_display_df" in env
    assert "—" in env["section3_display_df"]["Wall-clock (s)"].tolist()
    assert "section3_coverage_note" in env
    assert "P7" in env["section3_coverage_note"]


def test_notebook13_section5_surfaces_boundary_only_p7_rows() -> None:
    section5_cell = _find_code_cell(
        "notebooks/13_scaling_analysis.ipynb",
        "section5_completed = pd.DataFrame()",
    )
    records, display = _make_display_capture()

    metrics_master = pd.DataFrame(
        [
            {
                "instance_id": "P6",
                "family": "P",
                "execution_mode_norm": "mixture",
                "run_status_norm": "completed",
                "encoding": "wht_truncated_pricing",
                "decoder": "bp1",
                "ell": 2,
                "wall_clock_s": 10.0,
                "peak_memory_mb": 100.0,
            },
            {
                "instance_id": "P7",
                "family": "P",
                "execution_mode_norm": "mixture",
                "run_status_norm": "unavailable",
                "encoding": "wht_truncated_pricing",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "ell": 2,
                "boundary_reason": "statevector_memory_limit",
                "estimated_qubits": 29,
                "estimated_memory_gb": 32.0,
            },
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "Markdown": lambda text: text,
        "metrics_master_df": metrics_master,
        "instance_summary_df": pd.DataFrame(
            [{"Instance": "P6", "n (bits)": 12}, {"Instance": "P7", "n (bits)": 14}]
        ),
    }

    exec(section5_cell, env)

    assert "section5_boundary_df" in env and not env["section5_boundary_df"].empty
    assert env["section5_boundary_df"]["Instance"].tolist() == ["P7"]
    assert "section5_boundary_note" in env
    assert "execution/resource boundary records" in env["section5_boundary_note"] or "P7" in env["section5_boundary_note"]


def test_notebook13_section5_uses_only_canonical_execution_slice() -> None:
    section5_cell = _find_code_cell(
        "notebooks/13_scaling_analysis.ipynb",
        "section5_completed = pd.DataFrame()",
    )
    records, display = _make_display_capture()

    metrics_master = pd.DataFrame(
        [
            {
                "instance_id": "P6",
                "family": "P",
                "execution_mode_norm": "mixture",
                "encoding": "wht_truncated_pricing",
                "decoder": "bp1",
                "run_status_norm": "completed",
                "ell": 1,
                "wall_clock_s": 20.0,
                "peak_memory_mb": 4000.0,
                "attempted_execution": True,
            },
            {
                "instance_id": "P6",
                "family": "P",
                "execution_mode_norm": "mixture",
                "encoding": "wht_truncated_pricing",
                "decoder": "bruteforce",
                "run_status_norm": "completed",
                "ell": 1,
                "wall_clock_s": 8.0,
                "peak_memory_mb": 4100.0,
                "attempted_execution": True,
            },
            {
                "instance_id": "P6",
                "family": "P",
                "execution_mode_norm": "mixture",
                "encoding": "wht_truncated_pricing",
                "decoder": "oracle",
                "run_status_norm": "completed",
                "ell": 1,
                "wall_clock_s": 7.0,
                "peak_memory_mb": 4100.0,
                "attempted_execution": True,
            },
            {
                "instance_id": "P6",
                "family": "P",
                "execution_mode_norm": "mixture",
                "encoding": "ilp_derived",
                "decoder": "bp1",
                "run_status_norm": "completed",
                "ell": 1,
                "wall_clock_s": 3.0,
                "peak_memory_mb": 10.0,
                "attempted_execution": True,
            },
            {
                "instance_id": "P6",
                "family": "P",
                "execution_mode_norm": "mixture",
                "encoding": "wht_truncated",
                "decoder": "bp1",
                "run_status_norm": "completed",
                "ell": 1,
                "wall_clock_s": 9.0,
                "peak_memory_mb": 10.0,
                "attempted_execution": True,
            },
            {
                "instance_id": "P6",
                "family": "P",
                "execution_mode_norm": "coherent",
                "encoding": "wht_truncated_pricing",
                "decoder": "bp1",
                "run_status_norm": "completed",
                "ell": 1,
                "wall_clock_s": 99.0,
                "peak_memory_mb": 999.0,
                "attempted_execution": True,
            },
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "Markdown": lambda text: text,
        "metrics_master_df": metrics_master,
        "instance_summary_df": pd.DataFrame([{"Instance": "P6", "n (bits)": 12}]),
    }

    exec(section5_cell, env)

    completed = env["section5_completed_df"]
    assert set(completed["Encoding"]) == {"wht_truncated_pricing"}
    assert set(completed["Decoder"]) == {"bp1", "bruteforce"}


def test_notebook13_section5_splits_explicit_boundary_from_generic_unavailable() -> None:
    section5_cell = _find_code_cell(
        "notebooks/13_scaling_analysis.ipynb",
        "section5_completed = pd.DataFrame()",
    )
    records, display = _make_display_capture()

    metrics_master = pd.DataFrame(
        [
            {
                "instance_id": "P7",
                "family": "P",
                "execution_mode_norm": "mixture",
                "encoding": "wht_truncated_pricing",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "ell": 1,
                "run_status_norm": "unavailable",
                "attempted_execution": False,
                "boundary_reason": "qubit_count_exceeded",
                "estimated_qubits": 35,
                "estimated_memory_gb": 64.0,
            },
            {
                "instance_id": "P7",
                "family": "P",
                "execution_mode_norm": "mixture",
                "encoding": "wht_truncated_pricing",
                "decoder": "bruteforce",
                "alpha_mode": "uniform",
                "ell": 1,
                "run_status_norm": "unavailable",
            },
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "Markdown": lambda text: text,
        "metrics_master_df": metrics_master,
        "instance_summary_df": pd.DataFrame([{"Instance": "P7", "n (bits)": 14}]),
    }

    exec(section5_cell, env)

    assert "section5_explicit_boundary_df" in env
    assert "section5_generic_unavailable_df" in env
    assert env["section5_explicit_boundary_df"]["boundary_reason"].tolist() == ["qubit_count_exceeded"]
    assert env["section5_generic_unavailable_df"]["boundary_reason"].tolist() == ["—"]


def test_notebook13_section6_uses_coverage_boundary_language_when_boundary_metadata_is_sparse() -> None:
    verdict_cell = _find_code_cell(
        "notebooks/13_scaling_analysis.ipynb",
        "verdict_lines = []",
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "Markdown": lambda text: text,
        "section2_df": pd.DataFrame(),
        "section3_display_df": pd.DataFrame(),
        "section5_completed_df": pd.DataFrame([{"Instance": "P6"}]),
        "section5_boundary_df": pd.DataFrame([{"Instance": "P7"}]),
        "section5_explicit_boundary_df": pd.DataFrame(),
        "section5_generic_unavailable_df": pd.DataFrame([{"Instance": "P7"}]),
        "section5_claim_strength": "validated_executed_coverage_boundary",
    }

    exec(verdict_cell, env)

    rendered = "\n".join(str(item) for item in records if isinstance(item, str))
    assert "validated executed coverage boundary" in rendered.lower()
    assert "tractability boundary" not in rendered.lower()


def test_notebook13_section6_avoids_nan_verdict_prose() -> None:
    verdict_cell = _find_code_cell(
        "notebooks/13_scaling_analysis.ipynb",
        "verdict_lines = []",
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "display": display,
        "Markdown": lambda text: text,
        "section2_df": pd.DataFrame(
            [
                {
                    "Instance": "P3",
                    "n": 6,
                    "Encoding": "wht_truncated",
                    "Spearman rho(F,G)": np.nan,
                    "Retained energy eta": np.nan,
                    "Top-1 regret": np.nan,
                    "Availability": "unavailable",
                    "Unavailable reason": "paired_report_missing_and_no_truth_table_fallback",
                },
                {
                    "Instance": "P7",
                    "n": 14,
                    "Encoding": "ilp_derived",
                    "Spearman rho(F,G)": np.nan,
                    "Retained energy eta": np.nan,
                    "Top-1 regret": np.nan,
                    "Availability": "unavailable",
                    "Unavailable reason": "paired_report_missing_and_no_truth_table_fallback",
                },
            ]
        ),
        "section3_display_df": pd.DataFrame(),
        "section5_boundary_df": pd.DataFrame([{"Instance": "P7"}]),
        "section5_completed_df": pd.DataFrame(),
    }

    exec(verdict_cell, env)

    rendered = "\n".join(str(item) for item in records if isinstance(item, str))
    assert "nan" not in rendered.lower()
    assert "coverage is too thin" in rendered or "no diagnostic rows are available" in rendered


def test_notebook06_resources_master_missing_degrades_to_placeholders() -> None:
    load_cell = _find_code_cell(
        "notebooks/06_resource_analysis.ipynb",
        "try:\n    resources_master = load_resources_master",
    )
    structure_cell = _find_code_cell(
        "notebooks/06_resource_analysis.ipynb",
        'PANEL_SAME = "Structure vs decode (same-model authoritative)"',
    )
    records, display = _make_display_capture()

    def _raise_missing(*_a: Any, **_k: Any) -> pd.DataFrame:
        raise FileNotFoundError("missing resources_master")

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(),
        "load_resources_master": _raise_missing,
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)
    exec(structure_cell, env)

    assert env["resources_master"].empty
    assert _has_unavailable_panel(records, "Resource estimation")
    assert _has_unavailable_panel(records, "Structure vs decode (same-model authoritative)")
    assert _has_unavailable_panel(records, "Structure vs decode (cross-model diagnostic)")


def test_notebook06_metrics_master_missing_degrades_structure_decode_panel() -> None:
    load_cell = _find_code_cell(
        "notebooks/06_resource_analysis.ipynb",
        "try:\n    resources_master = load_resources_master",
    )
    structure_cell = _find_code_cell(
        "notebooks/06_resource_analysis.ipynb",
        'PANEL_SAME = "Structure vs decode (same-model authoritative)"',
    )
    records, display = _make_display_capture()

    resources_df = pd.DataFrame(
        [
            {
                "slot_id": "s1",
                "stage": "decoder_study",
                "run_status": "completed",
                "resource_decode_model_gate_est": 10.0,
            }
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(),
        "load_resources_master": lambda *_a, **_k: resources_df.copy(),
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)
    exec(structure_cell, env)

    assert _has_unavailable_panel(records, "Structure vs decode (same-model authoritative)")
    assert _has_unavailable_panel(records, "Structure vs decode (cross-model diagnostic)")


def test_notebook06_same_model_missing_renders_placeholder_even_if_cross_model_present() -> None:
    load_cell = _find_code_cell(
        "notebooks/06_resource_analysis.ipynb",
        "try:\n    resources_master = load_resources_master",
    )
    structure_cell = _find_code_cell(
        "notebooks/06_resource_analysis.ipynb",
        'PANEL_SAME = "Structure vs decode (same-model authoritative)"',
    )
    records, display = _make_display_capture()

    resources_df = pd.DataFrame(
        [
            {
                "slot_id": "s1",
                "stage": "decoder_study",
                "matrix": "matrix_c",
                "instance_id": "S_sparse_low_collision",
                "encoding": "synthetic_structured_bv",
                "decoder": "bp_osd",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "run_status": "completed",
                "resource_total_gate_est_with_decode": 123.0,
                "resource_total_depth_est_with_decode": 55.0,
            }
        ]
    )
    metrics_df = pd.DataFrame(
        [
            {
                "slot_id": "s1",
                "stage": "decoder_study",
                "matrix": "matrix_c",
                "instance_id": "S_sparse_low_collision",
                "encoding": "synthetic_structured_bv",
                "decoder": "bp_osd",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "run_status": "completed",
                "structure_collision_rate": 0.2,
                "gain_over_bp1": np.nan,
                "gain_over_bruteforce": 0.8,
            }
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: metrics_df.copy(),
        "load_resources_master": lambda *_a, **_k: resources_df.copy(),
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_numeric_columns": ensure_numeric_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)
    exec(structure_cell, env)

    # Main panel must not fall back to cross-model gain metric.
    assert _has_unavailable_panel(records, "Structure vs decode (same-model authoritative)")
    # Separate cross-model diagnostic panel should still be rendered.
    assert _has_unavailable_panel(records, "Structure vs decode (cross-model diagnostic)") or any(
        hasattr(item, "get_axes")
        and item.get_axes()
        and "cross-model" in str(item.get_axes()[0].get_title()).lower()
        for item in records
    )


def test_notebook06_duplicate_aligned_identity_renders_placeholder_without_dedupe() -> None:
    load_cell = _find_code_cell(
        "notebooks/06_resource_analysis.ipynb",
        "try:\n    resources_master = load_resources_master",
    )
    structure_cell = _find_code_cell(
        "notebooks/06_resource_analysis.ipynb",
        'PANEL_SAME = "Structure vs decode (same-model authoritative)"',
    )
    records, display = _make_display_capture()

    resources_df = pd.DataFrame(
        [
            {
                "slot_id": "dup",
                "stage": "decoder_study",
                "instance_id": "S_sparse_low_collision",
                "encoding": "synthetic_structured_bv",
                "decoder": "bp_osd",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "run_status": "completed",
            },
            {
                "slot_id": "dup",
                "stage": "decoder_study",
                "instance_id": "S_sparse_low_collision",
                "encoding": "synthetic_structured_bv",
                "decoder": "bp_osd",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "run_status": "completed",
            },
        ]
    )
    metrics_df = pd.DataFrame(
        [
            {
                "slot_id": "dup",
                "stage": "decoder_study",
                "instance_id": "S_sparse_low_collision",
                "encoding": "synthetic_structured_bv",
                "decoder": "bp_osd",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "run_status": "completed",
                "structure_collision_rate": 0.25,
                "gain_over_bp1": 0.9,
            }
        ]
    )
    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: metrics_df.copy(),
        "load_resources_master": lambda *_a, **_k: resources_df.copy(),
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_numeric_columns": ensure_numeric_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)
    exec(structure_cell, env)

    reason_hits = []
    for item in records:
        if isinstance(item, pd.DataFrame) and {"panel", "reason"}.issubset(item.columns):
            reason_hits.extend(item["reason"].astype(str).tolist())
    assert any("duplicate_aligned_identity" in reason for reason in reason_hits)


@pytest.mark.parametrize(
    "master_df",
    [pd.DataFrame(), pd.DataFrame(columns=["matrix", "encoding"])],
    ids=["missing_master", "empty_master"],
)
def test_notebook11_master_missing_or_empty_renders_placeholders(master_df: pd.DataFrame) -> None:
    load_cell = _find_code_cell(
        "notebooks/11_decision_faithfulness_dashboard.ipynb",
        "comparable_df = load_quality_cost_master",
    )
    shared_panel = _find_code_cell(
        "notebooks/11_decision_faithfulness_dashboard.ipynb",
        "shared_panel_metrics = [",
    )
    quality_panel = _find_code_cell(
        "notebooks/11_decision_faithfulness_dashboard.ipynb",
        'panel="Faithfulness vs quality"',
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(),
        "load_quality_cost_master": lambda *_a, **_k: master_df.copy(),
        "load_quality_cost_unavailable": lambda *_a, **_k: pd.DataFrame(),
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "parse_metric_availability": parse_metric_availability,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)
    exec(shared_panel, env)
    exec(quality_panel, env)

    assert _has_unavailable_panel(records, "Shared metric panel")
    assert _has_unavailable_panel(records, "Faithfulness vs quality")


def test_notebook11_transparency_renders_when_master_empty_and_unavailable_present() -> None:
    load_cell = _find_code_cell(
        "notebooks/11_decision_faithfulness_dashboard.ipynb",
        "comparable_df = load_quality_cost_master",
    )
    transparency = _find_code_cell(
        "notebooks/11_decision_faithfulness_dashboard.ipynb",
        "status_cols = [",
    )
    records, display = _make_display_capture()

    unavailable_df = pd.DataFrame(
        [
            {"unavailable_class": "missing_quality_fields", "unavailable_reason_code": "missing_gate"},
            {"unavailable_class": "missing_quality_fields", "unavailable_reason_code": "missing_gate"},
        ]
    )

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "load_run_manifest": lambda *_a, **_k: {},
        "load_metrics_master": lambda *_a, **_k: pd.DataFrame(),
        "load_quality_cost_master": lambda *_a, **_k: pd.DataFrame(),
        "load_quality_cost_unavailable": lambda *_a, **_k: unavailable_df.copy(),
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "parse_metric_availability": parse_metric_availability,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)
    exec(transparency, env)

    assert any(
        isinstance(item, pd.DataFrame) and ("count" in item.columns or "unavailable_class" in item.columns)
        for item in records
    )


@pytest.mark.parametrize(
    "master_df",
    [pd.DataFrame(), pd.DataFrame(columns=["best_F", "gates"])],
    ids=["missing_master", "empty_master"],
)
def test_notebook12_master_missing_or_empty_renders_placeholders(master_df: pd.DataFrame) -> None:
    load_cell = _find_code_cell(
        "notebooks/12_quality_vs_cost.ipynb",
        "quality_master = load_quality_cost_master",
    )
    comparable_panel = _find_code_cell(
        "notebooks/12_quality_vs_cost.ipynb",
        "view_cols = [",
    )
    pareto_panel = _find_code_cell(
        "notebooks/12_quality_vs_cost.ipynb",
        'panel="Pareto table"',
    )
    records, display = _make_display_capture()

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "load_quality_cost_master": lambda *_a, **_k: master_df.copy(),
        "load_quality_cost_unavailable": lambda *_a, **_k: pd.DataFrame(),
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)
    exec(comparable_panel, env)
    exec(pareto_panel, env)

    assert _has_unavailable_panel(records, "Stage E quality-cost comparable table")
    assert _has_unavailable_panel(records, "Pareto table")


def test_notebook12_transparency_renders_when_master_empty_and_unavailable_present() -> None:
    load_cell = _find_code_cell(
        "notebooks/12_quality_vs_cost.ipynb",
        "quality_master = load_quality_cost_master",
    )
    slice_panel = _find_code_cell(
        "notebooks/12_quality_vs_cost.ipynb",
        "slice_specs = [",
    )
    records, display = _make_display_capture()

    unavailable_df = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "matrix": "matrix_a",
                "unavailable_class": "missing_quality_fields",
                "unavailable_reason_code": "missing_gate",
            }
        ]
    )

    env: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "ROOT": Path("."),
        "display": display,
        "Markdown": lambda text: text,
        "load_quality_cost_master": lambda *_a, **_k: pd.DataFrame(),
        "load_quality_cost_unavailable": lambda *_a, **_k: unavailable_df.copy(),
        "normalize_run_status": normalize_run_status,
        "ensure_run_status_norm": ensure_run_status_norm,
        "ensure_optional_columns": ensure_optional_columns,
        "unavailable_panel_table": unavailable_panel_table,
        "placeholder_plot": placeholder_plot,
    }

    exec(load_cell, env)
    exec(slice_panel, env)

    assert any(
        isinstance(item, pd.DataFrame) and ("count" in item.columns or "unavailable_class" in item.columns)
        for item in records
    )
