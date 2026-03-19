"""Task 1: Decode-inclusive Qiskit parity on pricing instances P3 and P4.

Reproduces and extends the NB07 decode-inclusive validation:
- P3: ell=1, k=8 (matching NB07)
- P4: ell=2, k=8, k=12, k=15

Saves results to results/task1_qiskit_pricing_parity.json
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

from src.problem_generator import PricingProblem
from src.pricing_ilp import build_toy_pricing_ilp_model
from src.ilp_to_xorsat_exact import (
    build_ilp_exact_reference_artifact,
    canonicalize_ilp_encoding,
)
from src.decoder_bruteforce import BoundedDistanceDecoder
from src.dqi_state import run_dqi_decode_inclusive_reference
from src.dqi_circuit_qiskit import qiskit_decode_inclusive_distribution


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def run_parity_check(
    instance_path: Path,
    instance_id: str,
    k: int,
    ell: int,
) -> dict:
    """Run decode-inclusive parity check on a pricing instance."""
    print(f"  Running {instance_id} k={k} ell={ell} ...", flush=True)
    t0 = time.time()

    # Load instance
    with instance_path.open("r") as f:
        data = json.load(f)
    prob = PricingProblem.from_dict(data)

    # Build ILP-exact artifact and canonicalize with top-k
    ilp_model = build_toy_pricing_ilp_model(prob, instance_id=instance_id)
    exact_artifact = build_ilp_exact_reference_artifact(ilp_model)
    encoding = canonicalize_ilp_encoding(
        exact_artifact,
        instance_id=instance_id,
        source_metadata={"pricing_instance_path": str(instance_path)},
        k=k,
    )

    B = np.asarray(encoding["B"], dtype=np.int8)
    v = np.asarray(encoding["v"], dtype=np.int8)
    n = int(encoding["n"])
    m = int(encoding["m"])
    full_m = int(encoding["full_m_terms"])

    # Uniform alpha
    alpha = np.ones(ell + 1, dtype=np.float64)
    alpha /= np.linalg.norm(alpha)

    # Build decoder
    decoder = BoundedDistanceDecoder(B, ell=ell)

    # Hilbert space dimensions
    error_dim = 2 ** m
    syndrome_dim = 2 ** n
    total_dim = error_dim * syndrome_dim

    print(f"    m={m}, n={n}, ell={ell}, Hilbert dim={total_dim:,}", flush=True)

    # NumPy reference
    t_ref_start = time.time()
    ref_result = run_dqi_decode_inclusive_reference(B, v, alpha, ell, decoder)
    ref_time = time.time() - t_ref_start
    ref_prob = np.asarray(ref_result["prob_dist"], dtype=np.float64)
    ref_success = float(ref_result["success_prob"])

    # Qiskit decode-inclusive
    t_qk_start = time.time()
    qk_prob, qk_success, qk_meta = qiskit_decode_inclusive_distribution(
        B, v, alpha, ell, decoder
    )
    qk_time = time.time() - t_qk_start
    qk_prob = np.asarray(qk_prob, dtype=np.float64)

    # Compute metrics
    raw_tvd = tvd(ref_prob, qk_prob)
    max_abs_diff = float(np.max(np.abs(ref_prob - qk_prob)))
    elapsed = time.time() - t0

    verdict = "PASS" if raw_tvd < 1e-10 else "FAIL"
    print(f"    TVD={raw_tvd:.3e}  max_abs_diff={max_abs_diff:.3e}  [{verdict}]  ({elapsed:.1f}s)", flush=True)

    return {
        "instance_id": instance_id,
        "n_bits": n,
        "m_terms": m,
        "full_spectrum_m": full_m,
        "k": k,
        "ell": ell,
        "hilbert_space_dim": total_dim,
        "error_register_dim": error_dim,
        "syndrome_register_dim": syndrome_dim,
        "alpha": alpha.tolist(),
        "tvd": raw_tvd,
        "max_abs_diff": max_abs_diff,
        "success_prob_reference": ref_success,
        "success_prob_qiskit": float(qk_success),
        "num_decodable_syndromes": int(qk_meta["num_decodable_syndromes"]),
        "reference_time_s": round(ref_time, 4),
        "qiskit_time_s": round(qk_time, 4),
        "total_time_s": round(elapsed, 4),
        "verdict": verdict,
    }


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    runs = []

    # --- P3: ell=1, k=8 (matching NB07) ---
    p3_path = ROOT / "data" / "instances" / "pricing_3feat_6bit.json"
    print("=== P3 (3-feature, 6-bit) ===", flush=True)
    runs.append(run_parity_check(p3_path, "pricing_3feat_6bit", k=8, ell=1))

    # --- P4: ell=2, k=8, k=12 ---
    # k=15 (m=15, 8M-amplitude statevector) exceeded Qiskit initialize() limits
    # after 3.5+ hours and 7.4GB RAM — documented as scalability boundary.
    p4_path = ROOT / "data" / "instances" / "pricing_4feat_8bit.json"
    print("\n=== P4 (4-feature, 8-bit) ===", flush=True)
    for k in [8, 12]:
        runs.append(run_parity_check(p4_path, "pricing_4feat_8bit", k=k, ell=2))

    # Document the k=15 boundary
    runs.append({
        "instance_id": "pricing_4feat_8bit",
        "n_bits": 8,
        "m_terms": 15,
        "full_spectrum_m": 255,
        "k": 15,
        "ell": 2,
        "hilbert_space_dim": 2**23,
        "error_register_dim": 2**15,
        "syndrome_register_dim": 2**8,
        "alpha": None,
        "tvd": None,
        "max_abs_diff": None,
        "success_prob_reference": None,
        "success_prob_qiskit": None,
        "num_decodable_syndromes": None,
        "reference_time_s": None,
        "qiskit_time_s": None,
        "total_time_s": None,
        "verdict": "BOUNDARY",
        "boundary_note": (
            "Qiskit initialize() unitary decomposition for 2^15 error register "
            "exceeded 3.5 hours and 7.4GB RAM. The NumPy reference path completes "
            "in seconds — the bottleneck is Qiskit's state initialization, not the "
            "DQI logic. k=12 (1M amplitudes) is the practical Qiskit parity limit."
        ),
    })

    # Summary
    output = {
        "task": "task1_qiskit_pricing_parity",
        "description": "Decode-inclusive Qiskit parity verification on pricing instances",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "encoding": "ilp_derived_exact_reference",
            "decoder": "BoundedDistanceDecoder",
            "reference": "run_dqi_decode_inclusive_reference (NumPy)",
            "qiskit_path": "qiskit_decode_inclusive_distribution (postselected)",
            "metric": "total_variation_distance",
            "scope": "full_end_to_end_with_decode_postselection",
        },
        "runs": runs,
        "summary": {
            "all_executed_passed": all(
            r["verdict"] == "PASS" for r in runs if r["verdict"] != "BOUNDARY"
        ),
            "n_runs": len(runs),
            "instances_tested": sorted(set(r["instance_id"] for r in runs)),
        },
    }

    out_path = results_dir / "task1_qiskit_pricing_parity.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Results saved to {out_path} ===")
    print(f"All executed passed: {output['summary']['all_executed_passed']}")
    for r in runs:
        tvd_str = f"{r['tvd']:.3e}" if r['tvd'] is not None else "N/A"
        print(f"  {r['instance_id']} k={r['k']} ell={r['ell']}: "
              f"TVD={tvd_str}, verdict={r['verdict']}, "
              f"dim={r['hilbert_space_dim']:,}")


if __name__ == "__main__":
    main()
