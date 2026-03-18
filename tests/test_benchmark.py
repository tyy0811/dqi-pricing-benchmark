"""
test_benchmark.py — Tests for benchmark result transparency fields.

Run: python -m pytest tests/test_benchmark.py -v
"""

import json
import sys
sys.path.insert(0, ".")

import pytest

from src.benchmark import run_benchmark, run_multiperiod_pricing_benchmark
from src.benchmark import load_problem_from_json, load_scaling_instance
from src.problem_generator import small_instance
from src.classical_baselines import cp_sat_solver_result
from src.problem_generator import default_instance
from src.matrix_runner import run_matrix_a
from tests._multiperiod_helpers import make_tiny_multiperiod_problem


def test_load_problem_from_frozen_json():
    """Frozen JSON instance should reconstruct a PricingProblem."""
    prob = load_problem_from_json("data/instances/pricing_3feat_6bit.json")
    assert prob.n_features == 3
    assert prob.n_bits == 6


def test_load_scaling_instance_prefers_frozen():
    """Scaling loader should use frozen instances when present."""
    prob, from_frozen, path = load_scaling_instance(4)
    assert from_frozen is True
    assert prob.n_features == 4
    assert path.name == "pricing_4feat_8bit.json"


def test_load_scaling_instance_fallback_when_missing(tmp_path):
    """Missing frozen files should fall back to generated instances."""
    prob, from_frozen, path = load_scaling_instance(3, instances_dir=tmp_path / "missing")
    assert from_frozen is False
    assert prob.n_features == 3
    assert path.name == "pricing_3feat_6bit.json"


def test_load_scaling_instance_generates_p6_and_p7_when_frozen_missing(tmp_path):
    """Scaling loader should construct larger strict-superset instances when absent."""
    p6, from_frozen_6, path6 = load_scaling_instance(6, instances_dir=tmp_path / "missing")
    p7, from_frozen_7, path7 = load_scaling_instance(7, instances_dir=tmp_path / "missing")

    assert from_frozen_6 is False
    assert from_frozen_7 is False
    assert p6.n_features == 6
    assert p7.n_features == 7
    assert path6.name == "pricing_6feat_12bit.json"
    assert path7.name == "pricing_7feat_14bit.json"


def test_run_matrix_a_marks_p7_as_preflight_boundary_when_qubits_exceed_threshold(monkeypatch, tmp_path):
    def _fake_load_scaling_instance(n_features, instances_dir="data/instances"):
        prob = small_instance(3)
        return prob, False, tmp_path / f"pricing_{n_features}feat_{2*n_features}bit.json"

    def _fake_run_benchmark(*args, **kwargs):
        pytest.fail("run_benchmark should not execute for preflight-skipped P7")

    def _fake_paired(*args, **kwargs):
        pytest.fail("run_paired_encoding_comparison should not execute for preflight-skipped P7")

    def _fake_resources(*args, **kwargs):
        return {
            "decode_uncompute_model": "lookup_table_reversible",
            "decode_uncompute_status": "estimated",
            "decode_uncompute_confidence": "implementation_proxy",
            "total_gate_est_with_decode": 100,
            "total_depth_est_with_decode": 20,
            "total_qubits_est_with_decode": 31,
            "prefix_gate_est": 10,
            "prefix_depth_est": 3,
            "decode_model_gate_est": 5,
            "decode_model_depth_est": 2,
            "weight_register_qubits": 0,
            "coherent_weight_register_gate_est": 0,
            "coherent_weight_register_depth_est": 0,
        }

    monkeypatch.setattr("src.matrix_runner.load_scaling_instance", _fake_load_scaling_instance)
    monkeypatch.setattr("src.matrix_runner.run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr("src.matrix_runner.run_paired_encoding_comparison", _fake_paired)
    monkeypatch.setattr("src.matrix_runner.dqi_resource_estimate", _fake_resources)

    out = run_matrix_a(output_root=tmp_path / "results", features=[7], smoke=False)
    rows = out["records"]

    assert rows
    assert all(row["instance_id"] == "P7" for row in rows)
    assert all(row["run_status"] == "unavailable" for row in rows)
    assert all(row["attempted_execution"] is False for row in rows)
    assert all(row["boundary_reason"] == "qubit_count_exceeded" for row in rows)
    assert all(row["estimated_qubits"] == 31 for row in rows)


def test_benchmark_reports_weighted_and_unweighted_metrics():
    """Benchmark output should include both F-side and surrogate-side summaries."""
    prob = small_instance(3)
    results = run_benchmark(
        prob,
        k=8,
        ell=2,
        n_dqi_samples=40,
        sa_iter=300,
        top_k_by_surrogate=10,
        seed=7,
    )

    assert "surrogate" in results
    assert "spearman_rho" in results["surrogate"]
    assert "spearman_rho_weighted" in results["surrogate"]
    assert "structure" in results
    for key in [
        "row_weight_mean",
        "row_weight_max",
        "col_weight_mean",
        "col_weight_max",
        "syndrome_density",
        "number_of_unique_syndromes_decodable_at_ell",
        "decoder_collision_rate",
        "estimated_lookup_table_size",
    ]:
        assert key in results["structure"]
    assert "resources" in results
    for key in [
        "decode_uncompute_model",
        "decode_uncompute_status",
        "decode_uncompute_gate_est",
        "decode_uncompute_depth_est",
        "total_gate_est_with_decode",
        "total_depth_est_with_decode",
        "decode_cost_drivers",
        "qiskit_transpiled_structural_count",
    ]:
        assert key in results["resources"]

    expected_modes = {
        "dqi_uniform_mixture": ("uniform", "mixture"),
        "dqi_paper_mixture": ("paper", "mixture"),
        "dqi_heuristic_mixture": ("heuristic", "mixture"),
    }
    for dqi_key, (alpha_mode, execution_mode) in expected_modes.items():
        dqi = results[dqi_key]

        for key in [
            "best_F",
            "best_x",
            "best_G_weighted",
            "best_G_unweighted",
            "success_prob",
            "decoder_mode",
            "decoder_success_rate",
            "decode_exact_recovery_rate",
            "ambiguous_syndrome_rate",
            "flip_count_stats",
            "confidence_stats",
            "alpha",
            "alpha_mode",
            "alpha_vector",
            "alpha_source_label",
            "alpha_norm",
            "alpha_is_paper_derived",
            "execution_mode",
            "coherent_supported",
            "coherent_exact_small_instance",
            "weight_register_bits",
            "decode_semantics",
            "state_model",
            "postselection_kind",
            "approximation_ratio",
            "unique_samples",
            "top_by_weighted_surrogate",
            "gap_summary",
        ]:
            assert key in dqi
        assert dqi["alpha_mode"] == alpha_mode
        assert dqi["execution_mode"] == execution_mode
        assert dqi["decoder_mode"] == "bruteforce"
        assert len(dqi["alpha_vector"]) == results["parameters"]["ell"] + 1
        assert dqi["alpha_source_label"]
        assert isinstance(dqi["alpha_source_label"], str)
        assert isinstance(dqi["alpha_is_paper_derived"], bool)
        assert dqi["alpha_is_paper_derived"] is (alpha_mode == "paper")
        assert abs(dqi["alpha_norm"] - 1.0) < 1e-12

        top = dqi["top_by_weighted_surrogate"]
        assert top["k"] == 10
        assert len(top["x"]) == top["k"]
        assert len(top["F_values"]) == top["k"]
        assert len(top["G_weighted_values"]) == top["k"]
        assert len(top["G_unweighted_values"]) == top["k"]
        assert len(top["satisfied_counts"]) == top["k"]
        assert top["F_summary"]["max"] is not None

        gaps = dqi["gap_summary"]
        assert "at_true_optimum" in gaps
        assert "at_best_weighted_surrogate_sampled" in gaps
        assert gaps["at_true_optimum"]["G_weighted"] is not None
        assert gaps["at_true_optimum"]["G_unweighted"] is not None
        assert gaps["at_best_weighted_surrogate_sampled"] is not None

    coherent = results["dqi_paper_coherent"]
    if coherent.get("status") == "unsupported":
        for key in [
            "status",
            "reason",
            "execution_mode",
            "coherent_supported",
            "coherent_exact_small_instance",
            "coherent_comparison_attempted",
            "alpha_mode",
            "k",
            "ell",
            "n_bits",
            "m",
            "joint_num_bits",
            "joint_state_size",
            "cap_limit",
            "alpha_vector",
            "weight_register_bits",
            "decoder_mode",
            "decode_semantics",
            "state_model",
            "postselection_kind",
        ]:
            assert key in coherent
        assert coherent["execution_mode"] == "coherent"
        assert coherent["coherent_supported"] is False
    else:
        assert coherent["alpha_mode"] == "paper"
        assert coherent["execution_mode"] == "coherent"
        assert coherent["coherent_supported"] is True
        assert coherent["coherent_exact_small_instance"] is True

    assert "alpha_diagnostics" in results
    alpha_diag = results["alpha_diagnostics"]
    for key in [
        "alpha_uniform",
        "alpha_paper",
        "alpha_heuristic",
        "alpha_l2_distance",
        "alpha_l1_distance",
        "alpha_exactly_equal",
        "alpha_effectively_equal",
    ]:
        assert key in alpha_diag

    assert "dqi_pairwise_diagnostics" in results
    pair_diag = results["dqi_pairwise_diagnostics"]
    for key in [
        "delta_best_F_paper_vs_heuristic",
        "delta_success_prob_paper_vs_heuristic",
        "delta_best_F_paper_mixture_vs_coherent",
        "tvd_paper_mixture_vs_coherent",
        "same_top_sample_flag_paper_mixture_vs_coherent",
        "coherent_comparison_attempted",
    ]:
        assert key in pair_diag

    # Backward-compatible aliases should map to mixture payloads.
    assert results["dqi_uniform"] == results["dqi_uniform_mixture"]
    assert results["dqi_paper"] == results["dqi_paper_mixture"]
    assert results["dqi_heuristic"] == results["dqi_heuristic_mixture"]

    assert results["dqi_optimal"] is not results["dqi_heuristic_mixture"]
    assert results["dqi_optimal"] != results["dqi_heuristic_mixture"]
    assert results["dqi_optimal"]["_compat_source_key"] == "dqi_heuristic_mixture"
    assert "_compat_note" in results["dqi_optimal"]

    assert "dqi_heuristic_optimal" in results
    assert results["dqi_heuristic_optimal"]["_compat_source_key"] == "dqi_heuristic_mixture"

    assert "dqi_run_diagnostics" in results
    run_diag = results["dqi_run_diagnostics"]
    assert "headline_metric_equality" in run_diag
    assert "explanation" in run_diag
    assert "success_prob_uniform" in run_diag
    assert "success_prob_heuristic" in run_diag
    assert "success_prob_equal" in run_diag
    assert "metrics_that_differ" in run_diag
    assert isinstance(run_diag["metrics_that_differ"], list)

    assert "dqi_paper_mixture_bruteforce" in results
    assert "dqi_paper_mixture_bp1" in results
    assert "decoder_sensitivity_paper_mixture" in results
    assert results["dqi_paper_mixture_bruteforce"]["decoder_mode"] == "bruteforce"
    assert results["dqi_paper_mixture_bp1"]["decoder_mode"] == "bp1"
    decoder_diag = results["decoder_sensitivity_paper_mixture"]
    assert decoder_diag["alpha_mode"] == "paper"
    assert decoder_diag["execution_mode"] == "mixture"
    for decoder_mode in ["bruteforce", "bp1"]:
        by_decoder = decoder_diag["by_decoder"][decoder_mode]
        assert "best_F" in by_decoder
        assert "success_prob" in by_decoder
        assert "decode_success_rate" in by_decoder

    assert "cp_sat" in results["baselines"]


def test_run_multiperiod_pricing_benchmark_returns_separate_compact_payload():
    prob = make_tiny_multiperiod_problem()
    out = run_multiperiod_pricing_benchmark(prob)

    assert set(out) >= {
        "horizon_revenue",
        "static_vs_dynamic_delta",
        "per_period_configuration",
        "capacity_usage",
        "solve_time",
    }
    assert len(out["per_period_configuration"]) == 3
    assert out["capacity_usage"]["inventory_consumed_per_period"]
    assert out["capacity_usage"]["remaining_inventory_after_period"]


def test_single_period_cp_sat_contract_remains_unchanged():
    prob = default_instance()
    out = cp_sat_solver_result(prob, instance_id="single_period_guard")

    assert "objective_value" in out
    assert "optimized_objective_value" in out
    assert "x_opt" in out
    assert "constraint_summary" in out


def test_benchmark_can_emit_stage1_paired_encoding_section():
    """Benchmark should optionally include Stage 1 paired encoding comparison."""
    prob = small_instance(3)
    out = run_benchmark(
        prob,
        k=8,
        ell=2,
        n_dqi_samples=40,
        sa_iter=300,
        seed=5,
        include_stage1_paired_encoding=True,
    )
    assert "stage1_paired_encoding" in out
    paired = out["stage1_paired_encoding"]
    assert paired["backend_a_name"] == "wht_truncated"
    assert paired["backend_b_name"] == "ilp_exact_reference"


def test_benchmark_saves_and_loads_reviewer_facing_encoding_recommendation_metadata(tmp_path):
    """Reviewer-facing encoding recommendation metadata must roundtrip through benchmark JSON I/O."""
    prob = small_instance(3)
    result = run_benchmark(
        prob,
        k=8,
        ell=2,
        n_dqi_samples=40,
        sa_iter=300,
        seed=31,
        include_stage1_paired_encoding=True,
    )

    recommendation = result["encoding_backend_recommendation"]
    assert recommendation
    for key in [
        "status",
        "reason",
        "formal_verdict",
        "default_reviewer_backend",
        "recommendation_basis",
    ]:
        assert key in recommendation

    paired = result["stage1_paired_encoding"]
    assert paired
    assert "summary" in paired
    paired_summary = paired["summary"]
    assert "winner" in paired_summary
    assert "reviewer_verdict" in paired_summary
    assert "metrics_used" in paired_summary

    output_path = tmp_path / "benchmark_3feat.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)

    with open(output_path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)

    assert reloaded["encoding_backend_recommendation"] == recommendation
    assert reloaded["stage1_paired_encoding"]["summary"] == paired_summary
    assert (
        reloaded["stage1_paired_encoding"]["summary"]["reviewer_verdict"]
        == paired_summary["reviewer_verdict"]
    )

    if recommendation["status"] != "available":
        assert reloaded["encoding_backend_recommendation"]["status"] == recommendation["status"]
        assert reloaded["encoding_backend_recommendation"]["reason"] == recommendation["reason"]


def test_benchmark_dqi_report_exposes_shared_decision_metrics_panel():
    """DQI benchmark reports should carry shared grouped decision metrics."""
    prob = small_instance(3)
    out = run_benchmark(
        prob,
        k=8,
        ell=2,
        n_dqi_samples=40,
        sa_iter=300,
        seed=13,
    )
    panel = out["dqi_paper_mixture"]["decision_metrics"]
    assert panel["metrics_schema_version"] == "1.0"
    assert "core_decision" in panel
    assert "faithfulness" in panel
    assert "pipeline" in panel


def test_benchmark_includes_embedded_quality_vs_cost_section():
    """Benchmark payload should embed canonical quality-vs-cost summary."""
    prob = small_instance(3)
    out = run_benchmark(
        prob,
        k=8,
        ell=2,
        n_dqi_samples=30,
        sa_iter=200,
        seed=19,
    )
    assert "quality_vs_cost" in out
    qvc = out["quality_vs_cost"]
    assert qvc["schema_version"] == "1"
    assert "metadata" in qvc
    assert "rows" in qvc
    assert "pareto" in qvc
    assert "objective" in qvc["pareto"]
    assert "row_ids" in qvc["pareto"]


def test_benchmark_quality_vs_cost_rows_have_required_keys():
    """Rows should carry required quality/resource contract keys."""
    prob = small_instance(3)
    out = run_benchmark(
        prob,
        k=8,
        ell=2,
        n_dqi_samples=30,
        sa_iter=200,
        seed=23,
    )
    rows = out["quality_vs_cost"]["rows"]
    assert len(rows) >= 1
    row = rows[0]
    assert "row_id" in row
    assert "row_status" in row
    assert "quality" in row
    assert "resources" in row
    for key in [
        "best_F",
        "F_star",
        "best_F_over_F_star",
        "best_G",
        "top1_regret",
        "topk_regret",
        "optimum_recall_at_k",
        "best_top_G_sampled_F",
        "decision_distortion",
        "decode_success",
        "postselection_success",
    ]:
        assert key in row["quality"]
    assert "coherent_alpha_overhead" in row["resources"]
