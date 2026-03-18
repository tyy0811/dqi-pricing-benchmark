"""Shared decision-quality metrics for encoding and benchmark comparisons."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import spearmanr


METRICS_SCHEMA_VERSION = "1.0"


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _metric_goal_map() -> dict[str, str]:
    return {
        "best_sampled_F": "higher_is_better",
        "best_sampled_G": "higher_is_better",
        "top1_regret": "lower_is_better",
        "topk_regret": "lower_is_better",
        "optimum_recall_at_k": "higher_is_better",
        "precision_at_k_high_F": "higher_is_better",
        "best_top_G_sampled_F": "higher_is_better",
        "surrogate_best_vs_true_best_sampled_gap": "lower_abs_is_better",
        "spearman_rho_F_G": "higher_is_better",
        "retained_energy_eta": "higher_is_better",
        "decision_distortion": "lower_is_better",
        "decode_success": "higher_is_better",
        "postselection_success": "higher_is_better",
    }


def compute_decision_metrics(
    *,
    F_values: Iterable[float],
    G_values: Iterable[float],
    f_opt: float,
    k_eval: int,
    sampled_x: Iterable[int] | None = None,
    optimum_x: Iterable[int] | None = None,
    decode_success: float | None = None,
    postselection_success: float | None = None,
    retained_energy_eta: float | None = None,
    candidate_source_label: str,
    runtime_s: float | None = None,
    precision_high_f_threshold: float | None = None,
) -> dict:
    """Compute grouped decision metrics with stable key presence."""
    f_vals = _as_float_array(F_values)
    g_vals = _as_float_array(G_values)
    if f_vals.shape != g_vals.shape:
        raise ValueError("F_values and G_values must have the same shape.")
    n = int(f_vals.size)
    if n == 0:
        raise ValueError("F_values and G_values must be non-empty.")

    k_eff = int(min(max(int(k_eval), 1), n))
    g_order = np.argsort(g_vals)[::-1]
    top_idx = g_order[:k_eff]

    best_sampled_f = float(np.max(f_vals))
    best_sampled_g = float(np.max(g_vals))
    best_top_g_sampled_f = float(np.max(f_vals[top_idx]))
    f_at_surrogate_best = float(f_vals[int(g_order[0])])

    top1_regret = float(f_opt - best_sampled_f)
    topk_regret = float(f_opt - best_top_g_sampled_f)

    if optimum_x is not None and sampled_x is not None:
        optimum_set = set(int(x) for x in optimum_x)
        sampled_ordered = [int(x) for x in np.asarray(list(sampled_x), dtype=np.int64)[top_idx]]
        matched = sum(1 for x in sampled_ordered if x in optimum_set)
        optimum_recall_at_k = float(matched / max(1, len(optimum_set)))
        optimum_recall_reason = "available"
    else:
        # Fallback proxy: whether top-k by G contains an f_opt-attaining sample.
        optimum_recall_at_k = float(np.any(np.isclose(f_vals[top_idx], float(f_opt))))
        optimum_recall_reason = "available_proxy_by_f_opt"

    if precision_high_f_threshold is None:
        precision_high_f = None
        precision_reason = "not_computed"
    else:
        hits = np.sum(f_vals[top_idx] >= float(precision_high_f_threshold))
        precision_high_f = float(hits / k_eff)
        precision_reason = "available"

    if n > 1:
        rho, _ = spearmanr(f_vals, g_vals)
        spearman = float(rho)
        spearman_reason = "available"
    else:
        spearman = None
        spearman_reason = "insufficient_samples"

    if f_opt != 0:
        decision_distortion = float(abs(top1_regret) / abs(float(f_opt)))
        distortion_reason = "available"
    else:
        decision_distortion = None
        distortion_reason = "not_applicable"

    core_decision = {
        "best_sampled_F": best_sampled_f,
        "best_sampled_G": best_sampled_g,
        "top1_regret": top1_regret,
        "topk_regret": topk_regret,
        "optimum_recall_at_k": optimum_recall_at_k,
        "precision_at_k_high_F": precision_high_f,
        "best_top_G_sampled_F": best_top_g_sampled_f,
        "surrogate_best_vs_true_best_sampled_gap": float(best_sampled_f - f_at_surrogate_best),
        "decision_distortion": decision_distortion,
    }
    faithfulness = {
        "spearman_rho_F_G": spearman,
        "retained_energy_eta": (
            None if retained_energy_eta is None else float(retained_energy_eta)
        ),
    }
    pipeline = {
        "decode_success": None if decode_success is None else float(decode_success),
        "postselection_success": (
            None if postselection_success is None else float(postselection_success)
        ),
    }
    timing = {"runtime_s": None if runtime_s is None else float(runtime_s)}
    metadata = {
        "k_eval": int(k_eff),
        "sample_count": int(n),
        "unique_sample_count": (
            int(len(np.unique(np.asarray(list(sampled_x), dtype=np.int64))))
            if sampled_x is not None
            else int(n)
        ),
        "candidate_source_label": candidate_source_label,
    }
    metric_availability = {
        "best_sampled_F": "available",
        "best_sampled_G": "available",
        "top1_regret": "available",
        "topk_regret": "available",
        "optimum_recall_at_k": optimum_recall_reason,
        "precision_at_k_high_F": precision_reason,
        "best_top_G_sampled_F": "available",
        "surrogate_best_vs_true_best_sampled_gap": "available",
        "spearman_rho_F_G": spearman_reason,
        "retained_energy_eta": "available" if retained_energy_eta is not None else "not_computed",
        "decision_distortion": distortion_reason,
        "decode_success": "available" if decode_success is not None else "not_computed",
        "postselection_success": (
            "available" if postselection_success is not None else "not_computed"
        ),
    }

    return {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "core_decision": core_decision,
        "faithfulness": faithfulness,
        "pipeline": pipeline,
        "timing": timing,
        "metadata": metadata,
        "metric_goal": _metric_goal_map(),
        "metric_availability": metric_availability,
    }
