"""Task 2: Classical vs DQI comparison table across P3-P6.

Produces a clean comparison table:
| Instance | n | CP-SAT F* | CP-SAT time (s) | DQI best_F (BP1) | DQI time (s) | DQI approx ratio |

Saves results to results/task2_classical_vs_dqi_comparison.json and .csv
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

from src.problem_generator import PricingProblem
from src.pricing_model_ortools import solve_pricing_with_cp_sat
from src.pricing_ilp import build_toy_pricing_ilp_model
from src.ilp_to_xorsat_exact import (
    build_ilp_exact_reference_artifact,
    canonicalize_ilp_encoding,
)
from src.dqi_state import run_dqi_statevector
from src.weights import uniform_weights


INSTANCES = [
    ("P3", "pricing_3feat_6bit", 3, 6),
    ("P4", "pricing_4feat_8bit", 4, 8),
    ("P5", "pricing_5feat_10bit", 5, 10),
    ("P6", "pricing_6feat_12bit", 6, 12),
]

# DQI parameters per instance (ell, k, n_samples)
DQI_PARAMS = {
    "P3": {"ell": 1, "k": 8, "n_samples": 1024},
    "P4": {"ell": 2, "k": 15, "n_samples": 1024},
    "P5": {"ell": 3, "k": 15, "n_samples": 1024},
    "P6": {"ell": 2, "k": 15, "n_samples": 1024},
}


def run_instance(label, instance_id, n_features, n_bits):
    print(f"\n=== {label} ({instance_id}) ===", flush=True)
    instance_path = ROOT / "data" / "instances" / f"{instance_id}.json"
    with instance_path.open("r") as f:
        data = json.load(f)
    prob = PricingProblem.from_dict(data)

    # --- CP-SAT baseline ---
    print(f"  CP-SAT solve ...", flush=True)
    cp_result = solve_pricing_with_cp_sat(prob, instance_id=instance_id)
    cp_f_star = float(cp_result["objective_value"])
    cp_time = float(cp_result["solve_time_s"])
    print(f"    F* = {cp_f_star}, time = {cp_time:.4f}s", flush=True)

    # --- DQI statevector ---
    params = DQI_PARAMS[label]
    ell = params["ell"]
    k = params["k"]
    n_samples = params["n_samples"]

    print(f"  DQI statevector (BP1, ell={ell}, k={k}) ...", flush=True)

    # Build ILP-exact encoding
    ilp_model = build_toy_pricing_ilp_model(prob, instance_id=instance_id)
    exact_artifact = build_ilp_exact_reference_artifact(ilp_model)
    encoding = canonicalize_ilp_encoding(
        exact_artifact,
        instance_id=instance_id,
        source_metadata={},
        k=k,
    )

    B = np.asarray(encoding["B"], dtype=np.int8)
    v = np.asarray(encoding["v"], dtype=np.int8)
    m = int(encoding["m"])
    alpha = uniform_weights(ell)

    t0 = time.perf_counter()
    try:
        samples, probs, success_prob = run_dqi_statevector(
            B=B, v=v, alpha=alpha, ell=ell,
            n_samples=n_samples, seed=42,
            decoder_mode="bp1",
        )
        dqi_time = time.perf_counter() - t0

        # Evaluate samples under true objective
        f_values = np.array([prob.evaluate(int(x)) for x in samples], dtype=np.float64)
        dqi_best_f = float(np.max(f_values))
        dqi_approx_ratio = dqi_best_f / cp_f_star if cp_f_star > 0 else float("nan")

        # Feasible-only stats
        feasible_mask = f_values > -1000  # feasible configs have F > 0 typically
        n_feasible = int(np.sum(feasible_mask))
        mean_f_feasible = float(np.mean(f_values[feasible_mask])) if n_feasible > 0 else float("nan")

        dqi_status = "executed"
        print(f"    best_F = {dqi_best_f}, approx_ratio = {dqi_approx_ratio:.4f}, "
              f"time = {dqi_time:.2f}s, success = {success_prob:.4f}", flush=True)

    except Exception as exc:
        dqi_time = time.perf_counter() - t0
        dqi_best_f = float("nan")
        dqi_approx_ratio = float("nan")
        success_prob = float("nan")
        n_feasible = 0
        mean_f_feasible = float("nan")
        dqi_status = f"failed: {exc}"
        print(f"    DQI FAILED: {exc}", flush=True)

    return {
        "label": label,
        "instance_id": instance_id,
        "n_features": n_features,
        "n_bits": n_bits,
        "m_terms": m,
        "cpsat_f_star": cp_f_star,
        "cpsat_time_s": round(cp_time, 6),
        "dqi_decoder": "bp1",
        "dqi_ell": ell,
        "dqi_k": k,
        "dqi_best_f": dqi_best_f,
        "dqi_time_s": round(dqi_time, 4),
        "dqi_approx_ratio": round(dqi_approx_ratio, 6) if not np.isnan(dqi_approx_ratio) else None,
        "dqi_success_prob": round(float(success_prob), 6) if not np.isnan(success_prob) else None,
        "dqi_n_feasible_samples": n_feasible,
        "dqi_mean_f_feasible": round(mean_f_feasible, 2) if not np.isnan(mean_f_feasible) else None,
        "dqi_status": dqi_status,
    }


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    rows = []
    for label, instance_id, n_features, n_bits in INSTANCES:
        rows.append(run_instance(label, instance_id, n_features, n_bits))

    # Save JSON
    output = {
        "task": "task2_classical_vs_dqi_comparison",
        "description": "Classical CP-SAT vs DQI comparison across pricing instances P3-P6",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "classical": "CP-SAT exact solver via OR-Tools",
            "dqi_decoder": "BP1 (belief propagation)",
            "dqi_encoding": "ilp_derived_exact_reference (top-k truncation)",
            "dqi_execution": "statevector simulation (mixture mode)",
            "n_samples": 1024,
        },
        "rows": rows,
        "note": (
            "DQI does not win at this scale — classical CP-SAT solves in <0.1s. "
            "The purpose is an honest baseline showing where the crossover question lives."
        ),
    }

    json_path = results_dir / "task2_classical_vs_dqi_comparison.json"
    with json_path.open("w") as f:
        json.dump(output, f, indent=2)

    # Save CSV
    csv_path = results_dir / "task2_classical_vs_dqi_comparison.csv"
    fieldnames = [
        "label", "n_bits", "cpsat_f_star", "cpsat_time_s",
        "dqi_best_f", "dqi_time_s", "dqi_approx_ratio",
        "dqi_success_prob", "dqi_status",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"\n=== Results saved ===")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # Print table
    print(f"\n{'Instance':<10} {'n':>3} {'CP-SAT F*':>10} {'CP-SAT t(s)':>12} "
          f"{'DQI best_F':>11} {'DQI t(s)':>9} {'Approx ratio':>13} {'Status':>10}")
    print("-" * 85)
    for r in rows:
        ar = f"{r['dqi_approx_ratio']:.4f}" if r['dqi_approx_ratio'] is not None else "N/A"
        bf = f"{r['dqi_best_f']:.0f}" if not np.isnan(r['dqi_best_f']) else "N/A"
        print(f"{r['label']:<10} {r['n_bits']:>3} {r['cpsat_f_star']:>10.1f} "
              f"{r['cpsat_time_s']:>12.4f} {bf:>11} {r['dqi_time_s']:>9.2f} "
              f"{ar:>13} {r['dqi_status']:>10}")


if __name__ == "__main__":
    main()
