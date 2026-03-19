"""Task 3: Connect 3-period extension to max-XORSAT.

Extract single-period sub-problems from the 3-period scenario matrix,
reduce to max-XORSAT via ilp_to_xorsat_exact.py, and run DQI statevector.

Saves results to results/task3_multiperiod_xorsat_reduction.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.problem_generator import PricingProblem, decode_bitstring
from src.classical_baselines_multiperiod import make_multiperiod_scenario_problem
from src.pricing_ilp import build_toy_pricing_ilp_model
from src.ilp_to_xorsat_exact import (
    build_ilp_exact_reference_artifact,
    canonicalize_ilp_encoding,
)
from src.dqi_state import run_dqi_statevector
from src.weights import uniform_weights
from src.reduction import surrogate_objective


def run_period_dqi(
    multiperiod_prob,
    period_idx: int,
    scenario_label: str,
    ell: int = 1,
    k: int | None = None,
    n_samples: int = 512,
):
    """Extract period sub-problem, reduce to max-XORSAT, run DQI."""
    print(f"  Period {period_idx} ...", flush=True)

    # Extract single-period sub-problem
    period_prob = multiperiod_prob._period_problem(period_idx)
    n_bits = period_prob.n_bits
    n_features = period_prob.n_features

    # Brute-force ground truth for this period
    x_opt, f_star = period_prob.brute_force_solve()
    tiers_opt = decode_bitstring(x_opt, n_features)
    print(f"    Ground truth: F*={f_star:.1f}, x*={x_opt} (tiers={tiers_opt})", flush=True)

    # Build ILP model for the period sub-problem
    period_instance_id = f"{scenario_label}_period{period_idx}"
    ilp_model = build_toy_pricing_ilp_model(period_prob, instance_id=period_instance_id)

    # Build exact reference artifact
    exact_artifact = build_ilp_exact_reference_artifact(ilp_model)
    full_m = int(exact_artifact["m_terms"])

    # Canonicalize (use all terms if k not specified, since n=4 bits means only 15 terms)
    if k is None:
        k = full_m
    encoding = canonicalize_ilp_encoding(
        exact_artifact,
        instance_id=period_instance_id,
        source_metadata={"scenario": scenario_label, "period": period_idx},
        k=k,
    )

    B = np.asarray(encoding["B"], dtype=np.int8)
    v = np.asarray(encoding["v"], dtype=np.int8)
    w = np.asarray(encoding["term_weights"], dtype=np.float64)
    m = int(encoding["m"])
    n = int(encoding["n"])

    print(f"    Reduction: n={n}, m={m} (full_m={full_m}), ell={ell}", flush=True)

    # Verify the surrogate preserves the optimizer
    g_at_opt = float(exact_artifact["constant_offset"]) + surrogate_objective(
        x_opt, B, v, w, n
    )
    g_values = []
    for x in range(2**n):
        g_values.append(
            float(exact_artifact["constant_offset"]) + surrogate_objective(x, B, v, w, n)
        )
    g_values = np.array(g_values)
    g_opt_idx = int(np.argmax(g_values))
    g_preserves_optimizer = bool(g_opt_idx == x_opt)
    print(f"    Surrogate preserves optimizer: {g_preserves_optimizer} "
          f"(G argmax={g_opt_idx}, F argmax={x_opt})", flush=True)

    # Run DQI statevector
    alpha = uniform_weights(ell)
    t0 = time.perf_counter()
    samples, probs, success_prob = run_dqi_statevector(
        B=B, v=v, alpha=alpha, ell=ell,
        n_samples=n_samples, seed=42,
        decoder_mode="bp1",
    )
    dqi_time = time.perf_counter() - t0

    # Evaluate samples under true period objective
    f_values = np.array([period_prob.evaluate(int(x)) for x in samples], dtype=np.float64)
    dqi_best_f = float(np.max(f_values))
    dqi_approx_ratio = dqi_best_f / f_star if f_star > 0 else float("nan")

    # Check if DQI found the true optimum
    found_optimum = bool(dqi_best_f >= f_star - 1e-6)

    print(f"    DQI: best_F={dqi_best_f:.1f}, approx_ratio={dqi_approx_ratio:.4f}, "
          f"found_optimum={found_optimum}, time={dqi_time:.2f}s", flush=True)

    return {
        "scenario_label": scenario_label,
        "period_idx": period_idx,
        "period_instance_id": period_instance_id,
        "n_features": n_features,
        "n_bits": n_bits,
        "m_terms": m,
        "full_spectrum_m": full_m,
        "k": k,
        "ell": ell,
        "ground_truth_f_star": f_star,
        "ground_truth_x_star": x_opt,
        "ground_truth_tiers": tiers_opt,
        "surrogate_preserves_optimizer": g_preserves_optimizer,
        "dqi_decoder": "bp1",
        "dqi_best_f": dqi_best_f,
        "dqi_approx_ratio": round(dqi_approx_ratio, 6),
        "dqi_found_optimum": found_optimum,
        "dqi_success_prob": round(float(success_prob), 6),
        "dqi_time_s": round(dqi_time, 4),
        "n_samples": n_samples,
    }


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Pick the most interesting scenario: high-tight-smooth-0
    # (largest dynamic uplift = 150, package changes across periods)
    scenario_configs = [
        ("high", "tight", 0.0),
        ("medium", "tight", 0.0),
    ]

    all_results = []

    for demand_shift, inventory_regime, smoothing_penalty in scenario_configs:
        scenario_label = f"{demand_shift}-{inventory_regime}-smooth-{int(smoothing_penalty)}"
        print(f"\n=== Scenario: {scenario_label} ===", flush=True)

        prob = make_multiperiod_scenario_problem(
            demand_shift=demand_shift,
            inventory_regime=inventory_regime,
            smoothing_penalty=smoothing_penalty,
        )

        print(f"  Features: {[f.name for f in prob.features]}")
        print(f"  n_bits={prob.n_bits}, n_configs={prob.n_configurations}")

        # Run DQI on each of the 3 periods
        scenario_results = []
        for period_idx in range(3):
            result = run_period_dqi(prob, period_idx, scenario_label, ell=1)
            scenario_results.append(result)

        all_results.append({
            "scenario_label": scenario_label,
            "demand_shift": demand_shift,
            "inventory_regime": inventory_regime,
            "smoothing_penalty": smoothing_penalty,
            "n_features": prob.n_features,
            "n_bits": prob.n_bits,
            "periods": scenario_results,
            "all_periods_found_optimum": all(r["dqi_found_optimum"] for r in scenario_results),
        })

    # Summary
    output = {
        "task": "task3_multiperiod_xorsat_reduction",
        "description": (
            "Dynamic-pricing sub-problems successfully reduced to max-XORSAT "
            "and evaluated via DQI statevector"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "source": "3-period dynamic pricing extension (NB09)",
            "reduction": "ilp_to_xorsat_exact (full-spectrum WHT of period truth table)",
            "dqi": "statevector simulation with BP1 decoder",
            "scope": "single-period sub-problems extracted from multiperiod scenario matrix",
        },
        "scenarios": all_results,
        "summary": {
            "n_scenarios": len(all_results),
            "n_periods_total": sum(len(s["periods"]) for s in all_results),
            "all_optimums_found": all(s["all_periods_found_optimum"] for s in all_results),
        },
    }

    out_path = results_dir / "task3_multiperiod_xorsat_reduction.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Results saved to {out_path} ===")
    print(f"Scenarios tested: {output['summary']['n_scenarios']}")
    print(f"Total period sub-problems: {output['summary']['n_periods_total']}")
    print(f"All optimums found: {output['summary']['all_optimums_found']}")


if __name__ == "__main__":
    main()
