"""Tests for shared decision metrics panel."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


def test_compute_decision_metrics_has_required_sections_and_keys():
    from src.metrics_decision import compute_decision_metrics

    out = compute_decision_metrics(
        F_values=[1.0, 2.0, 3.0],
        G_values=[1.2, 2.1, 2.9],
        f_opt=3.0,
        k_eval=2,
        decode_success=0.9,
        postselection_success=0.8,
        candidate_source_label="unit_test",
    )

    assert "metrics_schema_version" in out
    for section in [
        "core_decision",
        "faithfulness",
        "pipeline",
        "timing",
        "metadata",
        "metric_goal",
        "metric_availability",
    ]:
        assert section in out

    required = {
        "best_sampled_F",
        "best_sampled_G",
        "top1_regret",
        "topk_regret",
        "optimum_recall_at_k",
        "precision_at_k_high_F",
        "best_top_G_sampled_F",
        "surrogate_best_vs_true_best_sampled_gap",
        "spearman_rho_F_G",
        "retained_energy_eta",
        "decision_distortion",
        "decode_success",
        "postselection_success",
    }
    flat = {**out["core_decision"], **out["faithfulness"], **out["pipeline"]}
    assert required.issubset(flat.keys())


def test_unavailable_metrics_are_present_as_none():
    from src.metrics_decision import compute_decision_metrics

    out = compute_decision_metrics(
        F_values=[1.0],
        G_values=[1.0],
        f_opt=1.0,
        k_eval=5,
        candidate_source_label="minimal",
    )
    assert "precision_at_k_high_F" in out["core_decision"]
    assert out["core_decision"]["precision_at_k_high_F"] is None
    assert "postselection_success" in out["pipeline"]
    assert out["pipeline"]["postselection_success"] is None
