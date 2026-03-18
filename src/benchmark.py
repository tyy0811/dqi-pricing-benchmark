"""
benchmark.py — Benchmark orchestration for DQI pricing experiments.

Runs DQI and classical baselines on frozen instances, collects metrics,
and generates comparison reports.
"""

from __future__ import annotations

import json
import time
import copy
import hashlib
import math
from pathlib import Path
from typing import Optional

import numpy as np

from src.problem_generator import PricingProblem, default_instance, scaling_instance, small_instance
from src.classical_baselines_multiperiod import run_multiperiod_baselines
from src.dqi_pipeline import DQIPipeline, compare_candidate_distributions_tvd
from src.classical_baselines import run_all_baselines
from src.encoding_compare import (
    EXPLORATORY_WHT_BACKEND,
    PRIMARY_EXACT_REFERENCE_BACKEND,
    resolve_encoding_reviewer_status,
    run_paired_encoding_comparison,
)
from src.family_s_manifest import load_family_s_family
from src.metrics_decision import compute_decision_metrics
from src.resources_compare import build_quality_vs_cost_summary
from src.reduction import build_surrogate, surrogate_faithfulness, surrogate_objective, count_satisfied
from src.resources import dqi_resource_estimate
from src.structure_metrics import compute_structure_metrics
from src.dqi_state import (
    COHERENT_STATE_SIZE_CAP,
    decoder_diagnostics,
    coherent_supported_for_instance,
    joint_num_bits_coherent,
    joint_state_size_coherent,
    run_dqi_statevector,
    weight_register_bits,
)
from src.weights import (
    heuristic_alpha_weights,
    max_safe_ell,
    paper_alpha_weights,
    uniform_weights,
    validate_dqi_parameters,
)


def _stable_encoding_artifact_id(*, B: np.ndarray, v: np.ndarray, term_weights: np.ndarray) -> str:
    """Return deterministic hash for surrogate encoding payload."""
    payload = {
        "B": np.asarray(B, dtype=np.int8).tolist(),
        "v": np.asarray(v, dtype=np.int8).tolist(),
        "term_weights": np.asarray(term_weights, dtype=np.float64).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_problem_from_json(path: str | Path) -> PricingProblem:
    """Load a PricingProblem from a frozen JSON instance file."""
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    return PricingProblem.from_dict(data)


def load_scaling_instance(
    n_features: int,
    instances_dir: str | Path = "data/instances",
) -> tuple[PricingProblem, bool, Path]:
    """Load scaling instance from frozen JSON when available, else generate."""
    instance_path = Path(instances_dir) / f"pricing_{n_features}feat_{2*n_features}bit.json"
    if instance_path.exists():
        return load_problem_from_json(instance_path), True, instance_path

    if n_features in {3, 4, 5, 6, 7}:
        return scaling_instance(n_features), False, instance_path
    if n_features == 5:
        return default_instance(), False, instance_path
    return small_instance(n_features), False, instance_path


def run_multiperiod_pricing_benchmark(prob) -> dict:
    """Run the separate fixed-3-period static-vs-dynamic benchmark surface."""
    baselines = run_multiperiod_baselines(prob)
    dynamic = baselines["dynamic_cp_sat"]
    static = baselines["static_fixed_config"]
    return {
        "horizon_revenue": float(dynamic["horizon_revenue"]),
        "static_vs_dynamic_delta": float(
            dynamic["horizon_revenue"] - static["horizon_revenue"]
        ),
        "per_period_configuration": dynamic["per_period_configuration"],
        "capacity_usage": dynamic["capacity_usage"],
        "solve_time": float(dynamic["solve_time_s"]),
        "baseline_comparison": {
            "dynamic_cp_sat": float(dynamic["horizon_revenue"]),
            "static_fixed_config": float(static["horizon_revenue"]),
        },
    }


def _build_dqi_report(
    dqi_results: dict,
    pipeline: DQIPipeline,
    x_opt: int,
    f_opt: float,
    elapsed: float,
    top_k_by_surrogate: int,
) -> dict:
    """Build a compact, JSON-serializable report for one DQI run."""
    samples = np.asarray(dqi_results['samples'])
    f_values = np.asarray(dqi_results['F_values'], dtype=np.float64)
    g_weighted_values = np.asarray(dqi_results['G_weighted_values'], dtype=np.float64)
    g_unweighted_values = np.asarray(dqi_results['G_unweighted_values'])
    satisfied_counts = np.asarray(dqi_results['satisfied_counts'])

    n = len(samples)
    k_eff = min(max(0, top_k_by_surrogate), n)
    order = np.argsort(g_weighted_values)[::-1] if n > 0 else np.array([], dtype=np.int64)
    top_idx = order[:k_eff]
    decision_metrics = compute_decision_metrics(
        F_values=f_values,
        G_values=g_weighted_values,
        f_opt=float(f_opt),
        k_eval=int(top_k_by_surrogate),
        sampled_x=samples.tolist(),
        optimum_x=[int(x_opt)],
        decode_success=dqi_results.get("decoder_success_rate"),
        postselection_success=dqi_results.get("success_prob"),
        retained_energy_eta=dqi_results.get("energy_fraction"),
        candidate_source_label=(
            f"dqi_{dqi_results.get('alpha_mode', 'unknown')}_{dqi_results.get('execution_mode', 'unknown')}"
        ),
        runtime_s=float(elapsed),
    )

    if k_eff > 0:
        top_f = f_values[top_idx]
        top_summary = {
            'k': int(k_eff),
            'x': [int(samples[i]) for i in top_idx],
            'F_values': [float(f_values[i]) for i in top_idx],
            'G_weighted_values': [float(g_weighted_values[i]) for i in top_idx],
            'G_unweighted_values': [int(g_unweighted_values[i]) for i in top_idx],
            'satisfied_counts': [int(satisfied_counts[i]) for i in top_idx],
            'F_summary': {
                'mean': float(np.mean(top_f)),
                'median': float(np.median(top_f)),
                'max': float(np.max(top_f)),
            },
        }

        best_g_idx = int(order[0])
        x_g_best = int(samples[best_g_idx])
        fg_summary = {
            'x': x_g_best,
            'F': float(f_values[best_g_idx]),
            'G_weighted': float(g_weighted_values[best_g_idx]),
            'G_unweighted': int(g_unweighted_values[best_g_idx]),
            'num_satisfied': int(satisfied_counts[best_g_idx]),
            'F_gap_to_true_opt': float(f_opt - f_values[best_g_idx]),
        }
    else:
        top_summary = {
            'k': 0,
            'x': [],
            'F_values': [],
            'G_weighted_values': [],
            'G_unweighted_values': [],
            'satisfied_counts': [],
            'F_summary': {
                'mean': None,
                'median': None,
                'max': None,
            },
        }
        fg_summary = None

    fx_summary = {
        'x': int(x_opt),
        'F': float(f_opt),
        'G_weighted': float(pipeline.evaluate_G_weighted(int(x_opt))),
        'G_unweighted': int(pipeline.evaluate_G_unweighted(int(x_opt))),
        'num_satisfied': int(pipeline.evaluate_num_satisfied(int(x_opt))),
    }

    return {
        'weight_mode': dqi_results.get('alpha_mode'),
        'best_F': dqi_results['best_F'],
        'best_x': dqi_results['best_x'],
        'approximation_ratio': dqi_results['approximation_ratio'],
        'unique_samples': dqi_results['unique_samples'],
        'best_G': dqi_results['best_G'],  # backward-compatible weighted alias
        'best_G_weighted': dqi_results['best_G_weighted'],
        'best_G_unweighted': dqi_results['best_G_unweighted'],
        'best_satisfied_count': dqi_results['best_satisfied_count'],
        'm_constraints': int(pipeline.m),
        'success_prob': dqi_results['success_prob'],
        'alpha': [float(a) for a in np.asarray(dqi_results['alpha'], dtype=np.float64)],
        'alpha_mode': dqi_results['alpha_mode'],
        'alpha_vector': [float(a) for a in np.asarray(dqi_results['alpha_vector'], dtype=np.float64)],
        'alpha_source_label': dqi_results['alpha_source_label'],
        'alpha_norm': float(dqi_results['alpha_norm']),
        'alpha_is_paper_derived': bool(dqi_results['alpha_is_paper_derived']),
        'execution_mode': dqi_results['execution_mode'],
        'coherent_supported': bool(dqi_results['coherent_supported']),
        'coherent_exact_small_instance': bool(dqi_results['coherent_exact_small_instance']),
        'weight_register_bits': int(dqi_results['weight_register_bits']),
        'decode_semantics': dqi_results['decode_semantics'],
        'state_model': dqi_results['state_model'],
        'postselection_kind': dqi_results['postselection_kind'],
        'decoder_mode': dqi_results['decoder_mode'],
        'decoder_success_rate': dqi_results['decoder_success_rate'],
        'decode_exact_recovery_rate': dqi_results['decode_exact_recovery_rate'],
        'ambiguous_syndrome_rate': dqi_results['ambiguous_syndrome_rate'],
        'flip_count_stats': dqi_results['flip_count_stats'],
        'confidence_stats': dqi_results['confidence_stats'],
        'probabilities': [float(p) for p in np.asarray(dqi_results['probabilities'], dtype=np.float64)],
        'time': elapsed,
        'decision_metrics': decision_metrics,
        'top_by_weighted_surrogate': top_summary,
        'gap_summary': {
            'at_true_optimum': fx_summary,
            'at_best_weighted_surrogate_sampled': fg_summary,
        },
    }


def _best_sample_with_tiebreak(dqi_results: dict) -> int:
    """Pick best sampled candidate by F, then G_weighted, then lexicographic x."""
    samples = np.asarray(dqi_results["samples"], dtype=np.int64)
    f_vals = np.asarray(dqi_results["F_values"], dtype=np.float64)
    g_vals = np.asarray(dqi_results["G_weighted_values"], dtype=np.float64)

    order = sorted(
        range(len(samples)),
        key=lambda i: (
            -float(f_vals[i]),
            -float(g_vals[i]),
            int(samples[i]),
        ),
    )
    return int(samples[order[0]])


def _build_coherent_unsupported_report(
    *,
    alpha_vector: np.ndarray,
    k: int,
    ell: int,
    n_bits: int,
    m: int,
    decoder_mode: str,
    cap_limit: int,
) -> dict:
    """Build stable payload for unsupported coherent runs."""
    joint_bits = joint_num_bits_coherent(m=m, n=n_bits, ell=ell)
    joint_size = joint_state_size_coherent(m=m, n=n_bits, ell=ell)
    return {
        "status": "unsupported",
        "reason": "coherent_hilbert_cap_exceeded",
        "execution_mode": "coherent",
        "alpha_mode": "paper",
        "k": int(k),
        "ell": int(ell),
        "n_bits": int(n_bits),
        "m": int(m),
        "coherent_supported": False,
        "coherent_exact_small_instance": False,
        "coherent_comparison_attempted": False,
        "joint_num_bits": int(joint_bits),
        "joint_state_size": int(joint_size),
        "cap_limit": int(cap_limit),
        "alpha_vector": [float(a) for a in np.asarray(alpha_vector, dtype=np.float64)],
        "weight_register_bits": int(weight_register_bits(ell)),
        "decoder_mode": decoder_mode,
        "decode_semantics": "projective_surrogate",
        "state_model": "coherent_joint_state",
        "postselection_kind": "projective_surrogate",
        "decoder_success_rate": None,
        "decode_exact_recovery_rate": None,
        "ambiguous_syndrome_rate": None,
        "flip_count_stats": None,
        "confidence_stats": None,
    }


def _build_pairwise_diagnostics(
    *,
    paper_mix_raw: dict,
    heuristic_mix_raw: dict,
    paper_mix_raw_prob: dict,
    paper_coherent_report: dict,
    paper_coherent_raw: dict | None,
) -> dict:
    """Build benchmark-level pairwise diagnostics."""
    coherent_comparison_attempted = paper_coherent_report.get("status") != "unsupported"
    if coherent_comparison_attempted:
        tvd_out = compare_candidate_distributions_tvd(paper_mix_raw_prob, paper_coherent_raw or {})
        tvd_value = tvd_out["tvd"] if tvd_out.get("ok", False) else None
        delta_best_coherent = float(
            paper_mix_raw["best_F"] - paper_coherent_report["best_F"]
        )
        same_top = (
            _best_sample_with_tiebreak(paper_mix_raw)
            == _best_sample_with_tiebreak(paper_coherent_raw or {})
        )
    else:
        tvd_value = None
        delta_best_coherent = None
        same_top = None

    return {
        "delta_best_F_paper_vs_heuristic": float(
            paper_mix_raw["best_F"] - heuristic_mix_raw["best_F"]
        ),
        "delta_success_prob_paper_vs_heuristic": float(
            paper_mix_raw["success_prob"] - heuristic_mix_raw["success_prob"]
        ),
        "delta_best_F_paper_mixture_vs_coherent": delta_best_coherent,
        "tvd_paper_mixture_vs_coherent": tvd_value,
        "same_top_sample_flag_paper_mixture_vs_coherent": same_top,
        "coherent_comparison_attempted": coherent_comparison_attempted,
    }


def _build_alpha_diagnostics(
    alpha_uniform: np.ndarray,
    alpha_paper: np.ndarray,
    alpha_heuristic: np.ndarray,
) -> dict:
    """Build diagnostics comparing all alpha modes."""
    delta_uniform_heuristic = alpha_uniform - alpha_heuristic
    delta_uniform_paper = alpha_uniform - alpha_paper
    delta_paper_heuristic = alpha_paper - alpha_heuristic
    return {
        'alpha_uniform': [float(a) for a in alpha_uniform],
        'alpha_paper': [float(a) for a in alpha_paper],
        'alpha_heuristic': [float(a) for a in alpha_heuristic],
        'alpha_l2_distance': float(np.linalg.norm(delta_uniform_heuristic, ord=2)),
        'alpha_l1_distance': float(np.linalg.norm(delta_uniform_heuristic, ord=1)),
        'alpha_l2_distance_uniform_paper': float(np.linalg.norm(delta_uniform_paper, ord=2)),
        'alpha_l1_distance_uniform_paper': float(np.linalg.norm(delta_uniform_paper, ord=1)),
        'alpha_l2_distance_paper_heuristic': float(np.linalg.norm(delta_paper_heuristic, ord=2)),
        'alpha_l1_distance_paper_heuristic': float(np.linalg.norm(delta_paper_heuristic, ord=1)),
        'alpha_exactly_equal': bool(np.array_equal(alpha_uniform, alpha_heuristic)),
        'alpha_effectively_equal': bool(np.allclose(alpha_uniform, alpha_heuristic, atol=1e-12, rtol=1e-10)),
    }


def _build_decoder_sensitivity_summary(
    *,
    bruteforce_report: dict,
    bp1_report: dict,
) -> dict:
    """Summarize decoder sensitivity for paper-alpha mixture runs."""
    return {
        "alpha_mode": "paper",
        "execution_mode": "mixture",
        "by_decoder": {
            "bruteforce": {
                "best_F": bruteforce_report.get("best_F"),
                "success_prob": bruteforce_report.get("success_prob"),
                "decode_success_rate": bruteforce_report.get("decoder_success_rate"),
            },
            "bp1": {
                "best_F": bp1_report.get("best_F"),
                "success_prob": bp1_report.get("success_prob"),
                "decode_success_rate": bp1_report.get("decoder_success_rate"),
            },
        },
        "delta_best_F_bp1_minus_bruteforce": (
            float(bp1_report.get("best_F", 0.0) - bruteforce_report.get("best_F", 0.0))
        ),
        "delta_success_prob_bp1_minus_bruteforce": (
            float(
                bp1_report.get("success_prob", 0.0)
                - bruteforce_report.get("success_prob", 0.0)
            )
        ),
        "delta_decode_success_rate_bp1_minus_bruteforce": (
            float(
                bp1_report.get("decoder_success_rate", 0.0)
                - bruteforce_report.get("decoder_success_rate", 0.0)
            )
        ),
    }


def _build_dqi_key_diagnostics(dqi_optimal: dict, dqi_heuristic: dict) -> dict:
    """Document compatibility-key semantics for reviewers."""
    dqi_opt_core = {k: v for k, v in dqi_optimal.items() if not k.startswith("_compat_")}
    return {
        'dqi_optimal_is_backward_compat_key': True,
        'dqi_optimal_refers_to': 'dqi_heuristic_mixture',
        'dqi_optimal_same_object_as_heuristic': bool(dqi_optimal is dqi_heuristic),
        'dqi_optimal_core_payload_equals_heuristic': bool(dqi_opt_core == dqi_heuristic),
        'dqi_optimal_has_compatibility_metadata': bool(any(k.startswith("_compat_") for k in dqi_optimal)),
    }


def _build_dqi_run_diagnostics(
    dqi_uniform: dict,
    dqi_paper: dict,
    dqi_heuristic: dict,
    alpha_diag: dict,
) -> dict:
    """Compare three DQI alpha modes and explain outcomes."""
    headline_metrics = [
        'best_F',
        'approximation_ratio',
        'best_G_weighted',
        'best_G_unweighted',
        'best_x',
    ]
    mode_runs = {
        'uniform': dqi_uniform,
        'paper': dqi_paper,
        'heuristic': dqi_heuristic,
    }
    headline_equal = {}
    differing_metrics = []
    for metric in headline_metrics:
        values = {mode: mode_runs[mode].get(metric) for mode in mode_runs}
        all_equal = len(set(values.values())) == 1
        headline_equal[metric] = all_equal
        if not all_equal:
            differing_metrics.append(metric)

    success_probs = {
        mode: float(mode_runs[mode].get('success_prob', 0.0))
        for mode in mode_runs
    }
    success_prob_equal = len(set(success_probs.values())) == 1
    if not success_prob_equal:
        differing_metrics.append('success_prob')

    if alpha_diag.get('alpha_effectively_equal', False):
        explanation = (
            "Uniform and heuristic alpha vectors are effectively identical, "
            "so similar DQI outcomes are expected."
        )
    elif all(headline_equal.values()) and success_prob_equal:
        explanation = (
            "Alpha vectors differ, but these frozen instances are insensitive on "
            "headline metrics and success probability."
        )
    else:
        explanation = (
            "At least one alpha mode produces distinct headline outcomes on this instance."
        )

    return {
        'headline_metric_equality': headline_equal,
        'success_prob_uniform': success_probs['uniform'],
        'success_prob_paper': success_probs['paper'],
        'success_prob_heuristic': success_probs['heuristic'],
        'success_prob_equal': success_prob_equal,
        'metrics_that_differ': differing_metrics,
        'explanation': explanation,
    }


def _alpha_vector_for_mode(
    *,
    alpha_mode: str,
    B: np.ndarray,
    v: np.ndarray,
    ell: int,
) -> np.ndarray:
    """Return normalized alpha vector for supported alpha modes."""
    if alpha_mode == "uniform":
        return uniform_weights(ell)
    if alpha_mode == "paper":
        return paper_alpha_weights(ell)
    if alpha_mode == "heuristic":
        return heuristic_alpha_weights(B, v, ell)
    raise ValueError(
        f"Invalid alpha_mode={alpha_mode!r}. Must be one of ['uniform', 'paper', 'heuristic']."
    )


def _decode_model_for_decoder(decoder_mode: str) -> str:
    if decoder_mode == "bruteforce":
        return "lookup_table_reversible"
    if decoder_mode == "bp1":
        return "bp1_classical_oracle_estimate"
    raise ValueError(
        f"Invalid decoder_mode={decoder_mode!r}. Must be one of ['bruteforce', 'bp1']."
    )


def _paired_metric(metrics: dict | None, section: str, key: str) -> float | None:
    """Safely extract one paired-comparison metric."""
    if not isinstance(metrics, dict):
        return None
    block = metrics.get(section)
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    return float(value) if value is not None else None


def _is_metric_complete(value) -> bool:
    """Return True when a scalar metric is present and finite."""
    if value is None:
        return False
    if isinstance(value, (float, np.floating)):
        return not math.isnan(float(value))
    return True


def _build_dqi_display_row(source_key: str, report: dict, label: str) -> dict:
    """Build reviewer-facing display row for one DQI report."""
    return {
        "source_key": source_key,
        "display_label": label,
        "kind": "dqi",
        "best_F": report.get("best_F"),
        "approximation_ratio": report.get("approximation_ratio"),
        "success_prob": report.get("success_prob"),
        "decoder_mode": report.get("decoder_mode"),
        "execution_mode": report.get("execution_mode"),
        "alpha_mode": report.get("alpha_mode"),
    }


def _build_paired_encoding_display_row(paired: dict) -> dict:
    """Build reviewer-facing row for the paired encoding comparison."""
    status = resolve_encoding_reviewer_status(paired)
    backend_b_metrics = paired.get("backend_b_metrics") if isinstance(paired, dict) else None
    backend_a_metrics = paired.get("backend_a_metrics") if isinstance(paired, dict) else None
    return {
        "source_key": "stage1_paired_encoding",
        "display_label": PRIMARY_EXACT_REFERENCE_BACKEND,
        "kind": "paired_encoding",
        "backend_a_name": paired.get("backend_a_reviewer_name", EXPLORATORY_WHT_BACKEND),
        "backend_b_name": paired.get("backend_b_reviewer_name", PRIMARY_EXACT_REFERENCE_BACKEND),
        "comparison_id": paired.get("comparison_id"),
        "formal_verdict": status["formal_verdict"],
        "reviewer_verdict": status["default_backend"],
        "best_F": _paired_metric(backend_b_metrics, "core_decision", "best_sampled_F"),
        "best_G": _paired_metric(backend_b_metrics, "core_decision", "best_sampled_G"),
        "top1_regret": _paired_metric(backend_b_metrics, "core_decision", "top1_regret"),
        "topk_regret": _paired_metric(backend_b_metrics, "core_decision", "topk_regret"),
        "optimum_recall_at_k": _paired_metric(backend_b_metrics, "core_decision", "optimum_recall_at_k"),
        "spearman_rho_F_G": _paired_metric(backend_b_metrics, "faithfulness", "spearman_rho_F_G"),
        "retained_energy_eta": _paired_metric(backend_b_metrics, "faithfulness", "retained_energy_eta"),
        "decision_distortion": _paired_metric(backend_b_metrics, "core_decision", "decision_distortion"),
        "exploratory_best_F": _paired_metric(backend_a_metrics, "core_decision", "best_sampled_F"),
    }


def _build_encoding_backend_recommendation(
    paired: dict | None,
    *,
    paired_requested: bool,
    is_pricing_instance: bool,
) -> dict:
    """Emit mandatory backend recommendation block for reviewer-facing pricing summaries."""
    if not is_pricing_instance:
        status_blob = {"status": "not_applicable", "reason": "non_pricing_instance"}
    elif not paired_requested:
        status_blob = {"status": "unavailable", "reason": "paired_comparison_disabled"}
    elif not isinstance(paired, dict):
        status_blob = {"status": "unavailable", "reason": "paired_comparison_missing"}
    else:
        status_blob = paired
    resolved = resolve_encoding_reviewer_status(status_blob)

    recommendation_basis = (
        "conservative_dominance_decision_metrics"
        if resolved["status"] == "available"
        else "paired_encoding_comparison"
    )

    return {
        "status": resolved["status"],
        "reason": resolved["reason"],
        "formal_verdict": resolved["formal_verdict"],
        "default_reviewer_backend": resolved["default_backend"],
        "recommendation_basis": recommendation_basis,
    }


def should_render_coherent_appendix(
    coherent_report: dict | None,
    *,
    enabled: bool,
) -> bool:
    """Coherent is appendix-only and renders only when enabled and complete."""
    if not enabled or not isinstance(coherent_report, dict):
        return False
    if coherent_report.get("status") == "unsupported":
        return False
    required = (
        coherent_report.get("best_F"),
        coherent_report.get("approximation_ratio"),
        coherent_report.get("success_prob"),
    )
    return all(_is_metric_complete(value) for value in required)


def _build_reviewer_display(results: dict) -> dict:
    """Build reviewer-facing headline and appendix blocks without changing payload keys."""
    headline_rows: list[dict] = []
    appendix_rows: list[dict] = []

    paper_mix = results.get("dqi_paper_mixture")
    if isinstance(paper_mix, dict):
        headline_rows.append(
            _build_dqi_display_row("dqi_paper_mixture", paper_mix, "dqi_paper_mixture")
        )

    paper_bp1 = results.get("dqi_paper_mixture_bp1")
    if isinstance(paper_bp1, dict):
        headline_rows.append(
            _build_dqi_display_row("dqi_paper_mixture_bp1", paper_bp1, "dqi_paper_mixture_bp1")
        )

    paired = results.get("stage1_paired_encoding")
    if isinstance(paired, dict):
        headline_rows.append(_build_paired_encoding_display_row(paired))

    if results.get("parameters", {}).get("include_alpha_ablation", False):
        for key, label in (
            ("dqi_uniform_mixture", "alpha_ablation_uniform"),
            ("dqi_heuristic_mixture", "alpha_ablation_heuristic"),
        ):
            report = results.get(key)
            if isinstance(report, dict):
                appendix_rows.append(_build_dqi_display_row(key, report, label))

    coherent = results.get("dqi_paper_coherent")
    if should_render_coherent_appendix(
        coherent,
        enabled=bool(results.get("parameters", {}).get("include_coherent_appendix", False)),
    ):
        appendix_rows.append(
            _build_dqi_display_row("dqi_paper_coherent", coherent, "dqi_paper_coherent_appendix")
        )

    return {
        "headline_order": [row["source_key"] for row in headline_rows],
        "headline_rows": headline_rows,
        "appendix_rows": appendix_rows,
    }


def run_benchmark(
    prob: PricingProblem,
    k: int = 15,
    ell: int = 3,
    n_dqi_samples: int = 1000,
    sa_iter: int = 10000,
    top_k_by_surrogate: int = 10,
    seed: Optional[int] = None,
    include_stage1_paired_encoding: bool = True,
    run_wht_ablation: bool = True,
    include_alpha_ablation: bool = False,
    include_coherent_appendix: bool = False,
) -> dict:
    """Run full benchmark on a pricing problem instance.

    Args:
        prob: PricingProblem instance.
        k: Number of surrogate terms.
        ell: DQI Dicke weight bound.
        n_dqi_samples: Number of DQI samples.
        sa_iter: SA iterations.
        seed: Random seed.
        include_stage1_paired_encoding: Include paired WHT vs exact-reference comparison payload.
        run_wht_ablation: Run exploratory WHT paired comparison on the default pricing path.
        include_alpha_ablation: Render alpha-ablation appendix rows in reviewer-facing summaries.
        include_coherent_appendix: Render coherent appendix rows when complete.

    Returns:
        Dictionary with all benchmark results.
    """
    results = {
        'problem': {
            'n_features': prob.n_features,
            'n_bits': prob.n_bits,
            'n_configurations': prob.n_configurations,
        },
        'parameters': {
            'k': k,
            'ell': ell,
            'n_dqi_samples': n_dqi_samples,
            'top_k_by_surrogate': top_k_by_surrogate,
            'seed': seed,
            'include_stage1_paired_encoding': include_stage1_paired_encoding,
            'run_wht_ablation': run_wht_ablation,
            'include_alpha_ablation': include_alpha_ablation,
            'include_coherent_appendix': include_coherent_appendix,
        },
    }

    # Ground truth
    x_opt, f_opt = prob.brute_force_solve()
    results['ground_truth'] = {
        'x_opt': x_opt,
        'x_opt_bits': format(x_opt, f'0{prob.n_bits}b'),
        'f_opt': f_opt,
    }

    # Surrogate faithfulness
    rho, eta = surrogate_faithfulness(prob, k=k)
    results['surrogate'] = {
        'spearman_rho': rho,
        'spearman_rho_weighted': rho,  # explicit alias: rho(F, G_weighted)
        'energy_fraction': eta,
    }

    surrogate_for_resources = build_surrogate(prob, k=k, ell=ell)
    structure = surrogate_for_resources["structure_metrics"]
    # Qiskit structural counts are decoder-agnostic (they measure
    # the implemented subset without decode). Enable for all decoders.
    resource_summary_lookup = dqi_resource_estimate(
        B=surrogate_for_resources["B"],
        v=surrogate_for_resources["v"],
        ell=ell,
        decode_uncompute_model="lookup_table_reversible",
        include_qiskit_structural_count=True,
    )
    # BP1 uses the same Qiskit structural counts as bruteforce - the transpiled
    # subset metrics are decoder-agnostic (they measure the implemented subset
    # without decode). Enable Qiskit structural counts for BP1 as well.
    resource_summary_bp1 = dqi_resource_estimate(
        B=surrogate_for_resources["B"],
        v=surrogate_for_resources["v"],
        ell=ell,
        decode_uncompute_model="bp1_classical_oracle_estimate",
        include_qiskit_structural_count=True,
    )
    encoding_artifact_id = _stable_encoding_artifact_id(
        B=surrogate_for_resources["B"],
        v=surrogate_for_resources["v"],
        term_weights=surrogate_for_resources["term_weights"],
    )
    results["structure"] = {
        "row_weight_mean": float(structure["row_weight_mean"]),
        "row_weight_max": int(structure["row_weight_max"]),
        "col_weight_mean": float(structure["col_weight_mean"]),
        "col_weight_max": int(structure["col_weight_max"]),
        "syndrome_density": float(structure["syndrome_density"]),
        "number_of_unique_syndromes_decodable_at_ell": int(
            structure["number_of_unique_syndromes_decodable_at_ell"]
        ),
        "decoder_collision_rate": float(structure["decoder_collision_rate"]),
        "estimated_lookup_table_size": int(structure["estimated_lookup_table_size"]),
        "ambiguous_syndrome_count": int(structure["ambiguous_syndrome_count"]),
        "rank_B_gf2": int(structure["rank_B_gf2"]),
        "rank": int(structure["rank"]),
        "rank_deficiency": int(structure["rank_deficiency"]),
        "decodable_syndrome_fraction": float(structure["decodable_syndrome_fraction"]),
        "syndrome_space_size": int(structure["syndrome_space_size"]),
        "m": int(structure["m"]),
        "n": int(structure["n"]),
    }
    results["resources"] = {
        "encoding_artifact_id": encoding_artifact_id,
        "decode_uncompute_model": resource_summary_lookup["decode_uncompute_model"],
        "decode_uncompute_status": resource_summary_lookup["decode_uncompute_status"],
        "decode_uncompute_confidence": resource_summary_lookup["decode_uncompute_confidence"],
        "decode_uncompute_gate_est": int(resource_summary_lookup["decode_uncompute_gate_est"]),
        "decode_uncompute_depth_est": int(resource_summary_lookup["decode_uncompute_depth_est"]),
        "total_gate_est": int(resource_summary_lookup["total_gate_est"]),
        "total_depth_est": int(resource_summary_lookup["total_depth_est"]),
        "total_gate_est_with_decode": int(resource_summary_lookup["total_gate_est_with_decode"]),
        "total_depth_est_with_decode": int(resource_summary_lookup["total_depth_est_with_decode"]),
        "total_qubits_est_with_decode": int(resource_summary_lookup["total_qubits_est_with_decode"]),
        "decode_cost_drivers": resource_summary_lookup["decode_cost_drivers"],
        "qiskit_transpiled_structural_count": resource_summary_lookup["qiskit_transpiled_structural_count"],
        "qiskit_subset_resource_report": resource_summary_lookup["qiskit_subset_resource_report"],
        "implemented_resources": resource_summary_lookup["implemented_resources"],
        "analytic_estimates": resource_summary_lookup["analytic_estimates"],
        "total_estimated_resources": resource_summary_lookup["total_estimated_resources"],
        "prefix_gate_est": int(resource_summary_lookup["prefix_gate_est"]),
        "prefix_depth_est": int(resource_summary_lookup["prefix_depth_est"]),
        "decode_model_gate_est": int(resource_summary_lookup["decode_model_gate_est"]),
        "decode_model_depth_est": int(resource_summary_lookup["decode_model_depth_est"]),
        "weight_register_qubits": int(resource_summary_lookup["weight_register_qubits"]),
        "coherent_weight_register_gate_est": int(resource_summary_lookup["coherent_weight_register_gate_est"]),
        "coherent_weight_register_depth_est": int(resource_summary_lookup["coherent_weight_register_depth_est"]),
        "weight_register_overhead_status": resource_summary_lookup["weight_register_overhead_status"],
    }
    results["resources_by_decoder_model"] = {
        "bruteforce": {
            **resource_summary_lookup,
            "encoding_artifact_id": encoding_artifact_id,
        },
        "bp1": {
            **resource_summary_bp1,
            "encoding_artifact_id": encoding_artifact_id,
        },
    }

    # Classical baselines
    results['baselines'] = run_all_baselines(prob, sa_iter=sa_iter, seed=seed)

    mixture_result_key = {
        'uniform': 'dqi_uniform_mixture',
        'paper': 'dqi_paper_mixture',
        'heuristic': 'dqi_heuristic_mixture',
    }
    mode_results: dict[str, dict] = {}
    raw_runs: dict[str, dict] = {}

    for alpha_mode in ['uniform', 'paper', 'heuristic']:
        t0 = time.perf_counter()
        pipeline = DQIPipeline(
            prob,
            k=k,
            ell=ell,
            alpha_mode=alpha_mode,
            execution_mode="mixture",
            decoder_mode="bruteforce",
        )
        dqi_run = pipeline.run_with_metrics(n_samples=n_dqi_samples, seed=seed)
        elapsed = time.perf_counter() - t0

        raw_runs[f"{alpha_mode}_mixture"] = dqi_run
        mode_results[alpha_mode] = _build_dqi_report(
            dqi_results=dqi_run,
            pipeline=pipeline,
            x_opt=x_opt,
            f_opt=f_opt,
            elapsed=elapsed,
            top_k_by_surrogate=top_k_by_surrogate,
        )
        results[mixture_result_key[alpha_mode]] = mode_results[alpha_mode]

    # Coherent execution is appendix-only for tiny supported instances.
    coherent_pipeline = DQIPipeline(
        prob,
        k=k,
        ell=ell,
        alpha_mode="paper",
        execution_mode="coherent",
        decoder_mode="bruteforce",
    )
    coherent_supported = coherent_supported_for_instance(
        m=coherent_pipeline.m,
        n=coherent_pipeline.n_bits,
        ell=ell,
        cap_limit=COHERENT_STATE_SIZE_CAP,
    )
    coherent_raw: dict | None = None
    if coherent_supported:
        t0 = time.perf_counter()
        coherent_raw = coherent_pipeline.run_with_metrics(n_samples=n_dqi_samples, seed=seed)
        elapsed = time.perf_counter() - t0
        results["dqi_paper_coherent"] = _build_dqi_report(
            dqi_results=coherent_raw,
            pipeline=coherent_pipeline,
            x_opt=x_opt,
            f_opt=f_opt,
            elapsed=elapsed,
            top_k_by_surrogate=top_k_by_surrogate,
        )
    else:
        results["dqi_paper_coherent"] = _build_coherent_unsupported_report(
            alpha_vector=coherent_pipeline.get_alpha_vector(),
            k=k,
            ell=ell,
            n_bits=coherent_pipeline.n_bits,
            m=coherent_pipeline.m,
            decoder_mode=coherent_pipeline.decoder_mode,
            cap_limit=COHERENT_STATE_SIZE_CAP,
        )

    # Decoder sensitivity study under paper alpha + mixture execution.
    results["dqi_paper_mixture_bruteforce"] = copy.deepcopy(results["dqi_paper_mixture"])
    paper_bp1_pipeline = DQIPipeline(
        prob,
        k=k,
        ell=ell,
        alpha_mode="paper",
        execution_mode="mixture",
        decoder_mode="bp1",
    )
    t0 = time.perf_counter()
    paper_bp1_raw = paper_bp1_pipeline.run_with_metrics(n_samples=n_dqi_samples, seed=seed)
    elapsed_bp1 = time.perf_counter() - t0
    results["dqi_paper_mixture_bp1"] = _build_dqi_report(
        dqi_results=paper_bp1_raw,
        pipeline=paper_bp1_pipeline,
        x_opt=x_opt,
        f_opt=f_opt,
        elapsed=elapsed_bp1,
        top_k_by_surrogate=top_k_by_surrogate,
    )
    results["decoder_sensitivity_paper_mixture"] = _build_decoder_sensitivity_summary(
        bruteforce_report=results["dqi_paper_mixture_bruteforce"],
        bp1_report=results["dqi_paper_mixture_bp1"],
    )
    results["structure"]["paper_mixture_success_prob"] = float(
        results["dqi_paper_mixture"]["success_prob"]
    )

    # Backward-compatible aliases remain mixture-mode reports.
    results['dqi_uniform'] = copy.deepcopy(results['dqi_uniform_mixture'])
    results['dqi_paper'] = copy.deepcopy(results['dqi_paper_mixture'])
    results['dqi_heuristic'] = copy.deepcopy(results['dqi_heuristic_mixture'])

    results['dqi_heuristic_optimal'] = copy.deepcopy(results['dqi_heuristic_mixture'])
    results['dqi_heuristic_optimal']['_compat_source_key'] = 'dqi_heuristic_mixture'
    results['dqi_heuristic_optimal']['_compat_note'] = (
        "Legacy key retained for compatibility; use dqi_heuristic_mixture."
    )
    results['dqi_optimal'] = copy.deepcopy(results['dqi_heuristic_mixture'])
    results['dqi_optimal']['_compat_source_key'] = 'dqi_heuristic_mixture'
    results['dqi_optimal']['_compat_note'] = (
        "Legacy key retained for compatibility; use dqi_heuristic_mixture."
    )

    alpha_uniform = np.asarray(results['dqi_uniform_mixture']['alpha_vector'], dtype=np.float64)
    alpha_paper = np.asarray(results['dqi_paper_mixture']['alpha_vector'], dtype=np.float64)
    alpha_heuristic = np.asarray(results['dqi_heuristic_mixture']['alpha_vector'], dtype=np.float64)
    results['alpha_diagnostics'] = _build_alpha_diagnostics(
        alpha_uniform,
        alpha_paper,
        alpha_heuristic,
    )
    results['dqi_pairwise_diagnostics'] = _build_pairwise_diagnostics(
        paper_mix_raw=raw_runs["paper_mixture"],
        heuristic_mix_raw=raw_runs["heuristic_mixture"],
        paper_mix_raw_prob=raw_runs["paper_mixture"],
        paper_coherent_report=results["dqi_paper_coherent"],
        paper_coherent_raw=coherent_raw,
    )
    results['dqi_key_diagnostics'] = _build_dqi_key_diagnostics(
        results['dqi_optimal'],
        results['dqi_heuristic_mixture'],
    )
    results['dqi_run_diagnostics'] = _build_dqi_run_diagnostics(
        results['dqi_uniform_mixture'],
        results['dqi_paper_mixture'],
        results['dqi_heuristic_mixture'],
        results['alpha_diagnostics'],
    )

    if include_stage1_paired_encoding and run_wht_ablation and int(prob.n_features) in {3, 4, 5}:
        results["stage1_paired_encoding"] = run_paired_encoding_comparison(
            prob=prob,
            k_wht=int(k),
            alpha_mode="paper",
            execution_mode="mixture",
            decoder_mode="bruteforce",
            n_samples=int(n_dqi_samples),
            seed=seed,
            top_k_eval=int(top_k_by_surrogate),
        )

    results["encoding_backend_recommendation"] = _build_encoding_backend_recommendation(
        results.get("stage1_paired_encoding"),
        paired_requested=bool(include_stage1_paired_encoding and run_wht_ablation),
        is_pricing_instance=bool(int(prob.n_features) in {3, 4, 5}),
    )
    results["reviewer_display"] = _build_reviewer_display(results)
    results["quality_vs_cost"] = build_quality_vs_cost_summary(results)

    return results


def run_structure_sweep_benchmark(
    *,
    instance_ids: list[str] | None = None,
    base_dir: str | Path = "data/instances/family_s",
    ell: int | None = None,
    alpha_mode: str = "paper",
    execution_mode: str = "mixture",
    decoders: tuple[str, ...] = ("bruteforce", "bp1"),
    n_dqi_samples: int = 256,
    top_k_by_surrogate: int = 10,
    seed: Optional[int] = None,
    output_dir: str | Path = "results",
    write_json: bool = True,
) -> list[dict]:
    """Run dedicated Family S synthetic structure-sweep benchmarks.

    This runner is intentionally separate from pricing-family scaling benchmarks.
    It evaluates synthetic max-XORSAT artifacts to characterize decoder behavior
    under controlled structure variation.
    """
    instances = load_family_s_family(base_dir=base_dir, instance_ids=instance_ids)
    results: list[dict] = []

    for inst in instances:
        B = np.asarray(inst["B"], dtype=np.int8)
        v = np.asarray(inst["v"], dtype=np.int8)
        n_bits = int(inst["n_bits"])
        m_rows = int(inst["m_rows"])
        ell_eff = int(inst["ell_default"] if ell is None else ell)
        alpha = _alpha_vector_for_mode(alpha_mode=alpha_mode, B=B, v=v, ell=ell_eff)
        unit_weights = np.ones(m_rows, dtype=np.float64)
        truth_table_f = np.array(
            [surrogate_objective(x, B, v, unit_weights, n_bits) for x in range(2 ** n_bits)],
            dtype=np.float64,
        )
        f_opt = float(np.max(truth_table_f))
        optimizer_set = np.flatnonzero(np.isclose(truth_table_f, f_opt)).astype(np.int64).tolist()
        structure = compute_structure_metrics(B, ell=ell_eff)
        encoding_artifact_id = _stable_encoding_artifact_id(
            B=B,
            v=v,
            term_weights=unit_weights,
        )

        record = {
            "family": "S",
            "family_label": "synthetic_structure_sweep",
            "instance_id": inst["instance_id"],
            "source_type": inst.get("source_type", "synthetic_structured_bv"),
            "objective_type": inst.get("objective_type", "max_xorsat_synthetic"),
            "description": inst.get("description"),
            "seed": int(inst["seed"]),
            "n_bits": n_bits,
            "m_rows": m_rows,
            "ell": ell_eff,
            "alpha_mode": alpha_mode,
            "execution_mode": execution_mode,
            "n_dqi_samples": int(n_dqi_samples),
            "top_k_by_surrogate": int(top_k_by_surrogate),
            "encoding_artifact_id": encoding_artifact_id,
            "generation_params": inst.get("generation_params", {}),
            "intended_structure_tags": list(inst.get("intended_structure_tags", [])),
            "structure_metrics": structure,
            "truth_summary": {
                "F_star": f_opt,
                "optimizer_set_size": len(optimizer_set),
                "optimizer_set": [int(x) for x in optimizer_set],
            },
            "dqi_by_decoder": {},
        }

        for idx, decoder_mode in enumerate(decoders):
            decoder_seed = None if seed is None else int(seed + idx)
            decode_model = _decode_model_for_decoder(decoder_mode)
            # Qiskit structural counts are decoder-agnostic (they measure
            # the implemented subset without decode). Enable for all decoders.
            resources = dqi_resource_estimate(
                B=B,
                v=v,
                ell=ell_eff,
                decode_uncompute_model=decode_model,
                include_qiskit_structural_count=True,
            )
            diagnostics = decoder_diagnostics(B=B, ell=ell_eff, decoder_mode=decoder_mode)

            # Pre-check: Skip DQI execution if structurally impossible (zero decodable syndromes)
            decoder_success_rate = diagnostics.get("decoder_success_rate", 0.0)
            if decoder_success_rate == 0.0:
                # Mark as not_applicable instead of failed - this is a structural boundary
                record["dqi_by_decoder"][decoder_mode] = {
                    "status": "not_applicable",
                    "status_reason_code": "dqi_impossible_zero_decodable_syndromes",
                    "status_reason": (
                        f"DQI execution skipped: instance has zero decodable syndromes at ell={ell_eff}. "
                        f"This is a structural property (decoder_success_rate=0.0), not a decoder failure."
                    ),
                    "decoder_mode": decoder_mode,
                    "decode_uncompute_model": decode_model,
                    "alpha_mode": alpha_mode,
                    "execution_mode": execution_mode,
                    "alpha_vector": [float(a) for a in np.asarray(alpha, dtype=np.float64)],
                    "error": None,
                    "decision_metrics": None,
                    "decoder_diagnostics": diagnostics,
                    "resources": resources,
                }
                continue

            t0 = time.perf_counter()
            try:
                samples, probabilities, success_prob = run_dqi_statevector(
                    B=B,
                    v=v,
                    alpha=alpha,
                    ell=ell_eff,
                    n_samples=n_dqi_samples,
                    seed=decoder_seed,
                    decoder_mode=decoder_mode,
                    execution_mode=execution_mode,
                )
                elapsed = float(time.perf_counter() - t0)
                samples_arr = np.asarray(samples, dtype=np.int64)
                f_values = truth_table_f[samples_arr]
                g_values = f_values.copy()
                satisfied = np.asarray(
                    [count_satisfied(int(x), B, v, n_bits) for x in samples_arr],
                    dtype=np.int64,
                )
                best_idx = int(np.argmax(f_values))

                decision_metrics = compute_decision_metrics(
                    F_values=f_values.tolist(),
                    G_values=g_values.tolist(),
                    f_opt=f_opt,
                    k_eval=int(top_k_by_surrogate),
                    sampled_x=samples_arr.tolist(),
                    optimum_x=optimizer_set,
                    decode_success=diagnostics.get("decoder_success_rate"),
                    postselection_success=float(success_prob),
                    retained_energy_eta=None,
                    candidate_source_label=(
                        f"family_s|{inst['instance_id']}|{decoder_mode}|{alpha_mode}|{execution_mode}"
                    ),
                    runtime_s=elapsed,
                )

                record["dqi_by_decoder"][decoder_mode] = {
                    "status": "ok",
                    "decoder_mode": decoder_mode,
                    "decode_uncompute_model": decode_model,
                    "alpha_mode": alpha_mode,
                    "execution_mode": execution_mode,
                    "alpha_vector": [float(a) for a in np.asarray(alpha, dtype=np.float64)],
                    "best_x": int(samples_arr[best_idx]),
                    "best_F": float(f_values[best_idx]),
                    "best_G": float(g_values[best_idx]),
                    "best_satisfied_count": int(np.max(satisfied)),
                    "approximation_ratio": (
                        float(f_values[best_idx] / f_opt) if abs(f_opt) > 1e-15 else None
                    ),
                    "success_prob": float(success_prob),
                    "sample_count": int(len(samples_arr)),
                    "unique_samples": int(np.unique(samples_arr).size),
                    "decision_metrics": decision_metrics,
                    "decoder_diagnostics": diagnostics,
                    "resources": resources,
                    "probabilities": [float(p) for p in np.asarray(probabilities, dtype=np.float64)],
                    "elapsed_s": elapsed,
                }
            except Exception as exc:  # pragma: no cover - defensive payload path
                record["dqi_by_decoder"][decoder_mode] = {
                    "status": "error",
                    "decoder_mode": decoder_mode,
                    "decode_uncompute_model": decode_model,
                    "alpha_mode": alpha_mode,
                    "execution_mode": execution_mode,
                    "alpha_vector": [float(a) for a in np.asarray(alpha, dtype=np.float64)],
                    "error": str(exc),
                    "decision_metrics": None,
                    "decoder_diagnostics": diagnostics,
                    "resources": resources,
                }

        results.append(record)

    if write_json:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "family_s_structure_sweep.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)

    return results


def run_scaling_benchmark(
    output_dir: str = "results",
    instances_dir: str | Path = "data/instances",
    seed: Optional[int] = 42,
) -> list[dict]:
    """Run benchmark on 3, 4, 5 feature instances.

    Args:
        output_dir: Directory to save results.
        instances_dir: Directory containing frozen JSON instances.
        seed: Random seed.

    Returns:
        List of result dictionaries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for n_features in [3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"Running benchmark: {n_features} features, {2*n_features} bits")
        print(f"{'='*60}")

        prob, from_frozen, instance_path = load_scaling_instance(
            n_features=n_features,
            instances_dir=instances_dir,
        )
        if not from_frozen:
            print(f"  WARNING: frozen instance missing at {instance_path}; using generated instance.")

        if n_features == 5:
            k = 15
        else:
            k = min(12, 2 ** (2 * n_features) - 1)

        # Compute safe ell to avoid syndrome collisions
        m = k  # number of parity terms
        n = prob.n_bits
        safe_ell = max_safe_ell(m, n)
        ell = min(3, safe_ell)  # Use safe value, capped at 3

        is_valid, msg = validate_dqi_parameters(m, n, ell)
        if not is_valid:
            print(f"  {msg}")

        results = run_benchmark(
            prob,
            k=k,
            ell=ell,
            n_dqi_samples=1000,
            seed=seed,
            include_stage1_paired_encoding=True,
        )

        # Print summary
        print(f"\nGround truth F*: {results['ground_truth']['f_opt']:.2f}")
        print(f"Surrogate rho(F,G_w): {results['surrogate'].get('spearman_rho_weighted', results['surrogate']['spearman_rho']):.4f}")
        print(f"Energy eta: {results['surrogate']['energy_fraction']:.4f}")
        structure = results.get("structure", {})
        if structure:
            print(
                "Structure: "
                f"row_w_mean={structure.get('row_weight_mean'):.3f}, "
                f"col_w_mean={structure.get('col_weight_mean'):.3f}, "
                f"syndrome_density={structure.get('syndrome_density'):.6f}, "
                f"collision_rate={structure.get('decoder_collision_rate'):.6f}"
            )
        print("Mainline: mixture execution with paper-derived alpha; WHT and coherent are non-headline paths.")
        reviewer_display = results.get("reviewer_display", {})
        for row in reviewer_display.get("headline_rows", []):
            if row.get("kind") == "dqi":
                dqi_report = results[row["source_key"]]
                print(
                    f"\n{row['display_label']}: "
                    f"F = {dqi_report['best_F']:.2f}, "
                    f"ratio = {dqi_report['approximation_ratio']:.4f}, "
                    f"G_w = {dqi_report['best_G_weighted']:.2f}, "
                    f"G_u = {dqi_report['best_G_unweighted']}, "
                    f"sat = {dqi_report['best_satisfied_count']}/{dqi_report['m_constraints']}, "
                    f"success = {dqi_report['success_prob']:.4f}"
                )
                top = dqi_report['top_by_weighted_surrogate']['F_summary']
                gap = dqi_report['gap_summary']
                if top['mean'] is None:
                    print("  top-G_w F stats: mean=NA, median=NA, max=NA")
                else:
                    print(
                        f"  top-G_w F stats: mean={top['mean']:.2f}, "
                        f"median={top['median']:.2f}, max={top['max']:.2f}"
                    )
                print(
                    f"  x_F_opt: F={gap['at_true_optimum']['F']:.2f}, "
                    f"G_w={gap['at_true_optimum']['G_weighted']:.2f}, "
                    f"G_u={gap['at_true_optimum']['G_unweighted']}"
                )
            elif row.get("kind") == "paired_encoding":
                paired_best_f = row.get("best_F")
                paired_ratio = (
                    None if paired_best_f is None else float(paired_best_f) / float(results['ground_truth']['f_opt'])
                )
                paired_ratio_s = "NA" if paired_ratio is None else f"{paired_ratio:.4f}"
                print(
                    f"\n{row['display_label']}: "
                    f"F = {paired_best_f if paired_best_f is not None else 'NA'}, "
                    f"ratio = {paired_ratio_s}, "
                    f"top1_regret = {row.get('top1_regret')}, "
                    f"topk_regret = {row.get('topk_regret')}"
                )
        if reviewer_display.get("appendix_rows"):
            print("\nAppendix:")
            for row in reviewer_display["appendix_rows"]:
                if row.get("kind") == "dqi":
                    print(
                        f"  {row['display_label']}: "
                        f"F = {row['best_F']:.2f}, "
                        f"ratio = {row['approximation_ratio']:.4f}, "
                        f"success = {row['success_prob']:.4f}"
                    )
        pair_diag = results.get("dqi_pairwise_diagnostics", {})
        if pair_diag and (
            results["parameters"].get("include_alpha_ablation")
            or results["parameters"].get("include_coherent_appendix")
        ):
            print(
                "  Appendix diagnostics: "
                f"ΔF(paper-heur)={pair_diag.get('delta_best_F_paper_vs_heuristic')}, "
                f"Δsucc(paper-heur)={pair_diag.get('delta_success_prob_paper_vs_heuristic')}, "
                f"TVD(paper mix vs coh)={pair_diag.get('tvd_paper_mixture_vs_coherent')}"
            )
        decoder_diag = results.get("decoder_sensitivity_paper_mixture", {})
        by_decoder = decoder_diag.get("by_decoder", {})
        if by_decoder:
            print("  Decoder sensitivity (paper+mixture):")
            for mode in ["bruteforce", "bp1"]:
                entry = by_decoder.get(mode, {})
                print(
                    f"    {mode}: F={entry.get('best_F')}, "
                    f"success={entry.get('success_prob')}, "
                    f"decode_success={entry.get('decode_success_rate')}"
                )

        print(f"BF:             F = {results['baselines']['brute_force']['f']:.2f} (exact)")
        cp_sat = results['baselines'].get('cp_sat', {})
        if cp_sat.get('status') == 'ok':
            print(f"CP-SAT:         F = {cp_sat['f']:.2f}, "
                  f"ratio = {cp_sat['approx_ratio']:.4f}, "
                  f"time = {cp_sat['time']:.4f}s, exact = {cp_sat.get('optimal', False)}")
        else:
            print(f"CP-SAT:         unavailable ({cp_sat.get('reason', 'not run')})")
        print(f"SA:             F = {results['baselines']['simulated_annealing']['f']:.2f}, "
              f"ratio = {results['baselines']['simulated_annealing']['approx_ratio']:.4f}")
        dqi_run_diag = results.get('dqi_run_diagnostics', {})
        if dqi_run_diag:
            print(f"  Run diagnostics: {dqi_run_diag.get('explanation', '')}")
        dqi_key_diag = results.get('dqi_key_diagnostics', {})
        if dqi_key_diag.get('dqi_optimal_is_backward_compat_key'):
            print("  Key note: dqi_optimal is a backward-compatible key for dqi_heuristic_mixture.")

        all_results.append(results)

        # Save individual result
        filename = f"benchmark_{n_features}feat.json"
        with open(output_path / filename, 'w') as f:
            json.dump(results, f, indent=2, default=float)

    # Save combined results
    with open(output_path / "benchmark_all.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    return all_results


def print_comparison_table(results: list[dict]) -> str:
    """Generate comparison table from benchmark results."""
    lines = [
        "",
        "=" * 136,
        "BENCHMARK COMPARISON",
        "=" * 136,
        "",
        (
            f"{'Features':<10} {'Bits':<6} {'F*':<10} "
            f"{'DQI-P':<12} {'DQI-BP1':<12} {'ILP-Exact':<12} "
            f"{'SA':<12} {'Succ-P':<10} {'Succ-BP1':<10}"
        ),
        "-" * 136,
    ]

    for r in results:
        n_feat = r['problem']['n_features']
        n_bits = r['problem']['n_bits']
        f_opt = r['ground_truth']['f_opt']
        dqi_p = r['dqi_paper_mixture']['approximation_ratio']
        dqi_bp1 = r['dqi_paper_mixture_bp1']['approximation_ratio']
        sa = r['baselines']['simulated_annealing']['approx_ratio']
        succ_p = r['dqi_paper_mixture']['success_prob']
        succ_bp1 = r['dqi_paper_mixture_bp1']['success_prob']
        paired = r.get("stage1_paired_encoding")
        paired_best_f = _paired_metric(
            paired.get("backend_b_metrics") if isinstance(paired, dict) else None,
            "core_decision",
            "best_sampled_F",
        )
        paired_ratio = None if paired_best_f is None else float(paired_best_f) / float(f_opt)
        paired_ratio_s = "NA" if paired_ratio is None else f"{paired_ratio:.4f}"

        lines.append(
            f"{n_feat:<10} {n_bits:<6} {f_opt:<10.2f} "
            f"{dqi_p:<12.4f} {dqi_bp1:<12.4f} {paired_ratio_s:<12} "
            f"{sa:<12.4f} {succ_p:<10.4f} {succ_bp1:<10.4f}"
        )

    lines.extend([
        "-" * 136,
        "DQI-P: paper-derived alpha under mixture execution (mainline reviewer-facing path)",
        "DQI-BP1: paper-derived alpha with BP1 decoder under mixture execution",
        "ILP-Exact: reviewer-facing paired encoding comparison labeled ilp_derived_exact",
        "SA: Simulated annealing (10k iterations)",
        "Succ-*: Decoding success probability for reviewer-facing mixture runs",
        "=" * 136,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("DQI Pricing Benchmark")
    print("=" * 60)

    results = run_scaling_benchmark(seed=42)
    print(print_comparison_table(results))
