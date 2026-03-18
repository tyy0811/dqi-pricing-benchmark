"""Tests for paired encoding comparison runner."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from src.problem_generator import small_instance


def test_paired_encoding_compare_report_is_json_serializable_and_schema_compatible():
    from src.encoding_compare import run_paired_encoding_comparison

    prob = small_instance(3)
    report = run_paired_encoding_comparison(
        prob=prob,
        k_wht=8,
        alpha_mode="paper",
        execution_mode="mixture",
        decoder_mode="bruteforce",
        n_samples=40,
        seed=11,
        top_k_eval=10,
    )

    required = {
        "comparison_id",
        "comparison_metadata",
        "instance_metadata",
        "backend_a_name",
        "backend_b_name",
        "backend_a_artifact_summary",
        "backend_b_artifact_summary",
        "artifacts_compatible_for_paired_run",
        "compatibility_notes",
        "backend_a_metrics",
        "backend_b_metrics",
        "metric_deltas",
    }
    assert required.issubset(report.keys())
    assert set(report["backend_a_metrics"].keys()) == set(report["backend_b_metrics"].keys())
    json.dumps(report, default=float)


def test_artifact_compatibility_predicate_detects_mismatch():
    from src.encoding_compare import artifacts_compatible_for_paired_run

    a = {
        "instance_hash": "abc",
        "codeword_mapping": {0: "standard", 1: "premium", 2: "luxury", 3: "invalid"},
        "bit_order": "msb_first",
        "bitstring_convention": "feature_major_msb_first",
        "objective_semantics": "maximize_F_over_feature_major_msb_first_bitstrings",
    }
    b = dict(a)
    b["bit_order"] = "lsb_first"

    ok, notes = artifacts_compatible_for_paired_run(
        a,
        b,
        alpha_mode="paper",
        execution_mode="mixture",
        decoder_mode="bruteforce",
    )
    assert ok is False
    assert notes


def test_resolve_encoding_reviewer_status_handles_complete_partial_and_unavailable():
    from src.encoding_compare import resolve_encoding_reviewer_status

    complete = resolve_encoding_reviewer_status(
        {
            "summary": {
                "winner": "mainline_exact_reference",
                "reviewer_verdict": "ilp_derived_exact",
                "metrics_used": ["top1_regret", "topk_regret"],
            }
        }
    )
    assert complete["default_backend"] == "ilp_derived_exact"
    assert complete["formal_verdict"] == "mainline_exact_reference"
    assert complete["status"] == "available"
    assert complete["reason"] == "paired_comparison_complete"
    assert complete["metrics_used"] == ["top1_regret", "topk_regret"]

    partial = resolve_encoding_reviewer_status(
        {
            "summary": {
                "reviewer_verdict": "ilp_derived_exact",
            }
        }
    )
    assert partial["status"] == "unavailable"
    assert partial["reason"] == "paired_comparison_partial"
    assert partial["default_backend"] == "ilp_derived_exact"
    assert partial["formal_verdict"] is None

    unavailable = resolve_encoding_reviewer_status(
        {
            "status": "unavailable",
            "reason": "paired_comparison_disabled",
            "metrics_used": [],
        }
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "paired_comparison_disabled"


def test_paired_encoding_verdict_mainline_exact_reference_when_all_decisive_metrics_favor_exact():
    """Exact-reference wins when decisive deltas strictly favor it on every criterion."""
    from src.encoding_compare import paired_encoding_verdict

    verdict = paired_encoding_verdict(
        {
            "top1_regret": -1.0,
            "topk_regret": -0.5,
            "optimum_recall_at_k": -0.2,
            "decision_distortion": -0.3,
        }
    )
    assert verdict == "mainline_exact_reference"


def test_paired_encoding_verdict_tie_when_decisive_metrics_are_mixed():
    """Mixed decisive-sign signals should resolve to tie."""
    from src.encoding_compare import paired_encoding_verdict

    verdict = paired_encoding_verdict(
        {
            "top1_regret": -1.0,
            "topk_regret": 0.5,
            "optimum_recall_at_k": -0.1,
            "decision_distortion": 0.0,
        }
    )
    assert verdict == "tie"


def test_paired_encoding_verdict_tie_when_a_decisive_metric_is_missing():
    """Missing decisive deltas are treated as unavailable/neutral under current contract."""
    from src.encoding_compare import paired_encoding_verdict

    verdict = paired_encoding_verdict(
        {
            "top1_regret": -1.0,
            "topk_regret": None,
            "optimum_recall_at_k": -0.2,
            "decision_distortion": -0.3,
        }
    )
    assert verdict == "tie"


def test_paired_encoding_verdict_ignores_non_decisive_best_sampled_F():
    """Non-decisive metrics such as best_sampled_F should not force a directional verdict."""
    from src.encoding_compare import paired_encoding_verdict

    verdict = paired_encoding_verdict(
        {
            "top1_regret": 0.0,
            "topk_regret": 0.0,
            "optimum_recall_at_k": 0.0,
            "decision_distortion": 0.0,
            "best_sampled_F": -1.0,
        }
    )
    assert verdict == "tie"


def test_flatten_paired_metrics_for_encoding_reads_nested_backend_blocks():
    from src.encoding_compare import flatten_paired_metrics_for_encoding

    report = {
        "backend_a_name": "wht_truncated",
        "backend_b_name": "ilp_derived",
        "backend_a_metrics": {
            "faithfulness": {
                "spearman_rho_F_G": 0.41,
                "retained_energy_eta": 0.76,
            },
            "core_decision": {
                "decision_distortion": 0.12,
                "top1_regret": 0.05,
                "topk_regret": 0.08,
            },
        },
        "backend_b_metrics": {
            "faithfulness": {
                "spearman_rho_F_G": 0.93,
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

    wht = flatten_paired_metrics_for_encoding(report, "wht_truncated")
    ilp = flatten_paired_metrics_for_encoding(report, "ilp_derived")

    assert wht["spearman_rho_F_G"] == 0.41
    assert wht["retained_energy_eta"] == 0.76
    assert wht["decision_distortion"] == 0.12
    assert wht["top1_regret"] == 0.05
    assert wht["topk_regret"] == 0.08
    assert wht["source_backend"] == "backend_a_metrics"

    assert ilp["spearman_rho_F_G"] == 0.93
    assert ilp["retained_energy_eta"] == 1.0
    assert ilp["decision_distortion"] == 0.0
    assert ilp["top1_regret"] == 0.0
    assert ilp["topk_regret"] == 0.0
    assert ilp["source_backend"] == "backend_b_metrics"


def test_flatten_paired_metrics_for_encoding_reports_unavailable_when_backend_missing():
    from src.encoding_compare import flatten_paired_metrics_for_encoding

    missing = flatten_paired_metrics_for_encoding(
        {"backend_a_name": "wht_truncated"},
        "ilp_derived",
    )

    assert missing["availability"] == "unavailable"
    assert missing["reason"] == "encoding_backend_missing_from_paired_report"
    assert missing["spearman_rho_F_G"] is None


def test_flatten_decision_metrics_reads_nested_compute_decision_metrics_output():
    from src.encoding_compare import flatten_decision_metrics

    nested = {
        "core_decision": {
            "decision_distortion": 0.04,
            "top1_regret": 0.02,
            "topk_regret": 0.03,
        },
        "faithfulness": {
            "spearman_rho_F_G": 0.81,
            "retained_energy_eta": 0.92,
        },
    }

    flat = flatten_decision_metrics(nested)

    assert flat["spearman_rho_F_G"] == 0.81
    assert flat["retained_energy_eta"] == 0.92
    assert flat["decision_distortion"] == 0.04
    assert flat["top1_regret"] == 0.02
    assert flat["topk_regret"] == 0.03
