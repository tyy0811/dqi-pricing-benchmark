"""Task 5: Fix negative Mean F reporting with feasible-only metrics.

Re-runs the NB05 decoder comparison on P5 (pricing_5feat_10bit) and adds:
- mean_F_feasible: average F over only feasible (non-penalized) samples
- median_F_feasible: median F over feasible samples
- n_feasible / n_total: feasible sample fraction
- feasible_rate: fraction of samples mapping to feasible pricing configs

Saves results to results/task5_decoder_comparison_corrected.json and .csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.problem_generator import PricingProblem, decode_bitstring
from src.reduction import build_surrogate_artifact
from src.dqi_state import run_dqi_statevector
from src.weights import uniform_weights


def is_feasible(prob: PricingProblem, x: int) -> bool:
    """Check if configuration x is feasible (no constraint violations)."""
    tiers = decode_bitstring(x, prob.n_features)
    if prob.count_invalid(tiers) > 0:
        return False
    if prob.bundle_violations(tiers) > 0:
        return False
    if prob.count_luxury(tiers) > prob.max_luxury:
        return False
    return True


def run_decoder_comparison(prob, B, v, ell, decoder_mode, n_samples=600, seed=0):
    """Run DQI and compute feasible-only metrics."""
    alpha = uniform_weights(ell)

    t0 = time.perf_counter()
    samples, probs, success_prob = run_dqi_statevector(
        B=B, v=v, alpha=alpha, ell=ell,
        n_samples=n_samples, seed=seed,
        decoder_mode=decoder_mode,
    )
    wall_time = time.perf_counter() - t0

    # Evaluate all samples
    f_values = np.array([prob.evaluate(int(x)) for x in samples], dtype=np.float64)
    feasible_flags = np.array([is_feasible(prob, int(x)) for x in samples], dtype=bool)

    n_total = len(f_values)
    n_feasible = int(np.sum(feasible_flags))
    feasible_rate = n_feasible / n_total if n_total > 0 else 0.0

    # All-sample metrics (original)
    best_f = float(np.max(f_values))
    mean_f = float(np.mean(f_values))

    # Feasible-only metrics (corrected)
    if n_feasible > 0:
        f_feasible = f_values[feasible_flags]
        mean_f_feasible = float(np.mean(f_feasible))
        median_f_feasible = float(np.median(f_feasible))
        best_f_feasible = float(np.max(f_feasible))
        std_f_feasible = float(np.std(f_feasible))
    else:
        mean_f_feasible = float("nan")
        median_f_feasible = float("nan")
        best_f_feasible = float("nan")
        std_f_feasible = float("nan")

    return {
        "decoder": decoder_mode,
        "ell": ell,
        "n_samples": n_total,
        "success_prob": round(float(success_prob), 6),
        "best_F": best_f,
        "mean_F_all": round(mean_f, 2),
        "n_feasible": n_feasible,
        "feasible_rate": round(feasible_rate, 6),
        "best_F_feasible": best_f_feasible if n_feasible > 0 else None,
        "mean_F_feasible": round(mean_f_feasible, 2) if n_feasible > 0 else None,
        "median_F_feasible": round(median_f_feasible, 2) if n_feasible > 0 else None,
        "std_F_feasible": round(std_f_feasible, 2) if n_feasible > 0 else None,
        "wall_time_s": round(wall_time, 4),
    }


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Load P5 instance and surrogate (matching NB05)
    instance_path = ROOT / "data" / "instances" / "pricing_5feat_10bit.json"
    surrogate_path = ROOT / "data" / "instances" / "surrogate_pricing_5feat_10bit_k15.json"

    with instance_path.open("r") as f:
        prob_data = json.load(f)
    prob = PricingProblem.from_dict(prob_data)

    with surrogate_path.open("r") as f:
        surrogate_data = json.load(f)

    B = np.asarray(surrogate_data["B"], dtype=np.int8)
    v = np.asarray(surrogate_data["v"], dtype=np.int8)
    n = int(surrogate_data.get("n_bits", surrogate_data.get("n")))
    m = B.shape[0]

    print(f"Instance: pricing_5feat_10bit (n={n}, m={m})")
    print(f"Penalty magnitude: {prob.penalty_invalid}")
    print(f"Feasible configs: {int(np.sum(prob.feasible_mask()))} / {prob.n_configurations}")
    print()

    rows = []
    for decoder_mode in ["bruteforce", "bp1"]:
        for ell in [1, 2, 3]:
            print(f"Running {decoder_mode} ell={ell} ...", flush=True)
            row = run_decoder_comparison(prob, B, v, ell, decoder_mode, n_samples=600, seed=0)
            rows.append(row)
            print(f"  best_F={row['best_F']:.0f}  mean_F_all={row['mean_F_all']:.0f}  "
                  f"mean_F_feasible={row.get('mean_F_feasible', 'N/A')}  "
                  f"feasible_rate={row['feasible_rate']:.4f}")

    # Save JSON
    output = {
        "task": "task5_decoder_comparison_corrected",
        "description": "Decoder comparison with feasible-only metrics (fixes negative Mean F)",
        "instance_id": "pricing_5feat_10bit",
        "n_bits": n,
        "m_terms": m,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "encoding": "wht_truncated (k=15, matching NB05)",
            "n_samples": 600,
            "seed": 0,
            "feasibility_check": "no invalid tiers, no bundle violations, luxury <= max_luxury",
        },
        "rows": rows,
        "note": (
            "mean_F_all includes penalty-heavy infeasible configs (penalty ~50,000 >> max revenue ~3,000). "
            "mean_F_feasible and median_F_feasible report only over postselected feasible pricing configs, "
            "which is the decision-relevant metric for a pricing reviewer."
        ),
    }

    json_path = results_dir / "task5_decoder_comparison_corrected.json"
    with json_path.open("w") as f:
        json.dump(output, f, indent=2)

    # Save CSV
    csv_path = results_dir / "task5_decoder_comparison_corrected.csv"
    fieldnames = [
        "decoder", "ell", "success_prob", "best_F",
        "mean_F_all", "feasible_rate", "mean_F_feasible",
        "median_F_feasible", "std_F_feasible",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # Print corrected table
    print(f"\n{'Decoder':<12} {'ell':>3} {'success':>8} {'best_F':>7} "
          f"{'mean_F_all':>11} {'feas_rate':>10} {'mean_F_feas':>12} {'med_F_feas':>11}")
    print("-" * 82)
    for r in rows:
        mff = f"{r['mean_F_feasible']:.1f}" if r['mean_F_feasible'] is not None else "N/A"
        medf = f"{r['median_F_feasible']:.1f}" if r['median_F_feasible'] is not None else "N/A"
        print(f"{r['decoder']:<12} {r['ell']:>3} {r['success_prob']:>8.4f} {r['best_F']:>7.0f} "
              f"{r['mean_F_all']:>11.0f} {r['feasible_rate']:>10.4f} {mff:>12} {medf:>11}")

    print(f"\n=== Results saved ===")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
