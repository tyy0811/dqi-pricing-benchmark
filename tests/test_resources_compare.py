"""Tests for Stage 3 quality-vs-cost join layer."""

import json
import sys

sys.path.insert(0, ".")

from src.resources_compare import build_quality_vs_cost_summary


def _minimal_results_fixture() -> dict:
    return {
        "problem": {
            "n_features": 3,
            "n_bits": 6,
            "n_configurations": 64,
        },
        "parameters": {
            "k": 8,
            "ell": 2,
            "n_dqi_samples": 40,
            "top_k_by_surrogate": 10,
            "seed": 7,
        },
        "ground_truth": {
            "x_opt": 3,
            "x_opt_bits": "000011",
            "f_opt": 10.0,
        },
        "structure": {
            "row_weight_mean": 2.0,
            "row_weight_max": 3,
            "col_weight_mean": 1.5,
            "col_weight_max": 2,
            "syndrome_density": 0.25,
            "number_of_unique_syndromes_decodable_at_ell": 8,
            "decoder_collision_rate": 0.1,
            "estimated_lookup_table_size": 16,
            "ambiguous_syndrome_count": 2,
            "rank_B_gf2": 4,
            "rank_deficiency": 2,
            "decodable_syndrome_fraction": 0.125,
            "m": 8,
            "n": 6,
            "paper_mixture_success_prob": 0.7,
        },
        "resources_by_decoder_model": {
            "bruteforce": {
                "decode_uncompute_model": "lookup_table_reversible",
                "decode_uncompute_status": "analytic_estimate",
                "decode_uncompute_confidence": "analytic_upper_bound",
                "decode_uncompute_gate_est": 120,
                "decode_uncompute_depth_est": 30,
                "total_gate_est": 180,
                "total_depth_est": 20,
                "total_gate_est_with_decode": 300,
                "total_depth_est_with_decode": 50,
                "total_qubits_est_with_decode": 14,
                "weight_register_qubits": 2,
                "weight_register_overhead_status": "analytic_upper_bound",
                "coherent_weight_register_gate_est": 24,
                "coherent_weight_register_depth_est": 10,
                "qiskit_transpiled_structural_count": {"status": "not_included"},
                "implemented_resources": {"phase_kick": {"gate_est": 8}},
                "analytic_estimates": {"decode_uncompute": {"status": "analytic_estimate"}},
                "total_estimated_resources": {
                    "total_with_decode": {
                        "gate_est": 300,
                        "depth_est": 50,
                        "qubits_est": 14,
                    }
                },
                "prefix_gate_est": 180,
                "prefix_depth_est": 20,
                "decode_model_gate_est": 120,
                "decode_model_depth_est": 30,
                "rank": 4,
                "rank_deficiency": 2,
                "decodable_syndrome_fraction": 0.125,
            },
            "bp1": {
                "decode_uncompute_model": "bp1_classical_oracle_estimate",
                "decode_uncompute_status": "analytic_estimate",
                "decode_uncompute_confidence": "implementation_proxy",
                "decode_uncompute_gate_est": 80,
                "decode_uncompute_depth_est": 18,
                "total_gate_est": 180,
                "total_depth_est": 20,
                "total_gate_est_with_decode": 260,
                "total_depth_est_with_decode": 38,
                "total_qubits_est_with_decode": 12,
                "weight_register_qubits": 2,
                "weight_register_overhead_status": "analytic_upper_bound",
                "coherent_weight_register_gate_est": 24,
                "coherent_weight_register_depth_est": 10,
                "qiskit_transpiled_structural_count": {"status": "not_included"},
                "implemented_resources": {"phase_kick": {"gate_est": 8}},
                "analytic_estimates": {"decode_uncompute": {"status": "analytic_estimate"}},
                "total_estimated_resources": {
                    "total_with_decode": {
                        "gate_est": 260,
                        "depth_est": 38,
                        "qubits_est": 12,
                    }
                },
                "prefix_gate_est": 180,
                "prefix_depth_est": 20,
                "decode_model_gate_est": 80,
                "decode_model_depth_est": 18,
                "rank": 4,
                "rank_deficiency": 2,
                "decodable_syndrome_fraction": 0.125,
            },
        },
        "dqi_paper_mixture": {
            "alpha_mode": "paper",
            "execution_mode": "mixture",
            "decoder_mode": "bruteforce",
            "coherent_supported": True,
            "best_F": 9.0,
            "best_G": 3.0,
            "success_prob": 0.7,
            "decision_metrics": {
                "core_decision": {
                    "top1_regret": 1.0,
                    "topk_regret": 0.5,
                    "optimum_recall_at_k": 0.25,
                    "best_top_G_sampled_F": 9.5,
                    "decision_distortion": 0.1,
                },
                "faithfulness": {
                    "spearman_rho_F_G": 0.62,
                    "retained_energy_eta": 0.73,
                },
                "pipeline": {
                    "decode_success": 0.91,
                    "postselection_success": 0.7,
                },
            },
        },
        "dqi_paper_mixture_bp1": {
            "alpha_mode": "paper",
            "execution_mode": "mixture",
            "decoder_mode": "bp1",
            "coherent_supported": True,
            "best_F": 8.5,
            "best_G": 2.7,
            "success_prob": 0.66,
            "decision_metrics": {
                "core_decision": {
                    "top1_regret": 1.5,
                    "topk_regret": 0.9,
                    "optimum_recall_at_k": 0.20,
                    "best_top_G_sampled_F": 9.1,
                    "decision_distortion": 0.15,
                },
                "faithfulness": {
                    "spearman_rho_F_G": 0.58,
                    "retained_energy_eta": 0.71,
                },
                "pipeline": {
                    "decode_success": 0.86,
                    "postselection_success": 0.66,
                },
            },
        },
    }


def test_quality_vs_cost_schema_has_required_top_level_keys():
    out = build_quality_vs_cost_summary(_minimal_results_fixture())
    assert out["schema_version"] == "1"
    assert "rows" in out and isinstance(out["rows"], list)
    assert out["row_count"] == len(out["rows"])
    assert "metadata" in out
    assert out["metadata"]["canonical_source"] == "embedded_run_benchmark_payload"
    assert "pareto" in out
    assert "objective" in out["pareto"]
    assert "maximize" in out["pareto"]["objective"]
    assert "minimize" in out["pareto"]["objective"]


def test_quality_vs_cost_row_has_required_quality_and_resource_keys():
    row = build_quality_vs_cost_summary(_minimal_results_fixture())["rows"][0]
    for key in [
        "best_F",
        "F_star",
        "best_F_over_F_star",
        "best_G",
        "decode_success",
        "postselection_success",
    ]:
        assert key in row["quality"]
    assert "resources" in row
    assert "coherent_alpha_overhead" in row["resources"]
    assert "resource_status" in row["resources"]


def test_quality_vs_cost_row_id_is_deterministic_and_sorted():
    out_a = build_quality_vs_cost_summary(_minimal_results_fixture())
    out_b = build_quality_vs_cost_summary(_minimal_results_fixture())
    ids_a = [r["row_id"] for r in out_a["rows"]]
    ids_b = [r["row_id"] for r in out_b["rows"]]
    assert ids_a == ids_b
    assert ids_a == sorted(ids_a)


def test_quality_vs_cost_missing_optional_run_is_resilient():
    payload = _minimal_results_fixture()
    payload.pop("dqi_paper_mixture_bp1", None)
    out = build_quality_vs_cost_summary(payload)
    assert isinstance(out["rows"], list)
    assert out["row_count"] == len(out["rows"])
    assert len(out["rows"]) >= 1


def test_quality_vs_cost_payload_is_json_serializable():
    out = build_quality_vs_cost_summary(_minimal_results_fixture())
    json.dumps(out)
