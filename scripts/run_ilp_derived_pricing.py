"""Run ILP-derived exact encoding experiments for pricing family P3-P6.

This script generates ilp_derived_exact encodings and runs DQI statevector
simulation to produce quality_cost_master-compatible rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.benchmark import load_scaling_instance
from src.dqi_state import decoder_diagnostics, run_dqi_statevector
from src.ilp_to_xorsat_exact import build_ilp_exact_reference_artifact, canonicalize_ilp_encoding
from src.metrics_decision import compute_decision_metrics
from src.pricing_ilp import build_toy_pricing_ilp_model
from src.resources import dqi_resource_estimate
from src.structure_metrics import compute_structure_metrics
from src.weights import (
    heuristic_alpha_weights,
    max_safe_ell,
    paper_alpha_weights,
    uniform_weights,
    validate_dqi_parameters,
)


SCHEMA_VERSION = "1.0"
STAGE_NAME = "benchmark_matrix"
MATRIX_NAME = "matrix_c"
ENCODING_NAME = "ilp_derived_exact"


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_slug(raw: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in raw)


def _alpha_vector_for_mode(*, alpha_mode: str, B: np.ndarray, v: np.ndarray, ell: int) -> np.ndarray:
    if alpha_mode == "uniform":
        return uniform_weights(ell)
    if alpha_mode == "paper":
        return paper_alpha_weights(ell)
    if alpha_mode == "heuristic":
        return heuristic_alpha_weights(B, v, ell)
    raise ValueError(f"Unsupported alpha_mode={alpha_mode!r}")


def _metric_availability_completed() -> dict[str, str]:
    return {
        "best_sampled_F": "available",
        "best_sampled_G": "available",
        "top1_regret": "available",
        "topk_regret": "available",
        "optimum_recall_at_k": "available",
        "best_top_G_sampled_F": "available",
        "spearman_rho_F_G": "available",
        "retained_energy_eta": "available",
        "decision_distortion": "available",
        "decode_success": "available",
        "postselection_success": "available",
        "runtime_sec": "available",
    }


def _run_single_ilp_experiment(
    *,
    instance_id: str,
    n_features: int,
    alpha_mode: str,
    decoder: str,
    n_dqi_samples: int,
    top_k_by_surrogate: int,
    trial_seed: int,
    k_terms: int,
    output_root: Path,
) -> dict:
    """Run a single ILP-derived exact experiment."""

    # Load problem
    prob, from_file, instance_path = load_scaling_instance(n_features)
    n_bits = 2 * n_features

    # Build ILP model and exact encoding
    ilp_model = build_toy_pricing_ilp_model(prob, instance_id=f"pricing_{n_features}feat_{n_bits}bit")
    exact_artifact = build_ilp_exact_reference_artifact(ilp_model)

    encoding = canonicalize_ilp_encoding(
        exact_artifact,
        instance_id=f"pricing_{n_features}feat_{n_bits}bit",
        source_metadata={"pricing_instance_path": str(instance_path)},
        k=k_terms,
    )

    B = np.asarray(encoding["B"], dtype=np.int8)
    v = np.asarray(encoding["v"], dtype=np.int8)
    term_weights = np.asarray(encoding["term_weights"], dtype=np.float64)
    m = B.shape[0]

    # Compute ell
    ell = max_safe_ell(m, n_bits)
    ell = min(ell, 3)  # Cap at 3 for tractability

    # Get alpha
    alpha = _alpha_vector_for_mode(alpha_mode=alpha_mode, B=B, v=v, ell=ell)

    # Validate parameters
    validate_dqi_parameters(m=m, n=n_bits, ell=ell)

    # Build truth table for decision metrics
    f_table = np.asarray(prob.build_truth_table(), dtype=np.float64)
    f_opt = float(np.max(f_table))
    optimum_x = np.flatnonzero(np.isclose(f_table, f_opt)).astype(np.int64)

    # Check decoder diagnostics
    diag = decoder_diagnostics(B=B, ell=ell, decoder_mode=decoder)

    run_id = (
        f"{MATRIX_NAME}_P_{instance_id}_{ENCODING_NAME}_{decoder}_{alpha_mode}_mixture_seed{trial_seed}_{uuid.uuid4().hex[:8]}"
    )

    row = {
        "schema_version": SCHEMA_VERSION,
        "matrix": MATRIX_NAME,
        "stage": STAGE_NAME,
        "run_id": run_id,
        "run_timestamp_utc": _now_iso_utc(),
        "family": "P",
        "instance_id": instance_id,
        "encoding": ENCODING_NAME,
        "decoder": decoder,
        "alpha_mode": alpha_mode,
        "execution_mode": "mixture",
        "trial_seed": trial_seed,
        "seed": trial_seed,
        "ell": ell,
        "m_active": m,
        "n_bits": n_bits,
        "in_experiment_matrix": True,
    }

    # Pre-check: Skip DQI execution if structurally impossible (zero decodable syndromes)
    decoder_success_rate = diag.get("decoder_success_rate", 0.0)
    if decoder_success_rate == 0.0:
        row["run_status"] = "not_applicable"
        row["status_reason_code"] = "dqi_impossible_zero_decodable_syndromes"
        row["status_reason"] = f"DQI execution skipped: instance has zero decodable syndromes at ell={ell}"
        row["metric_availability"] = {k: "not_applicable" for k in _metric_availability_completed()}
        return row

    # Run DQI
    start_time = time.perf_counter()
    try:
        samples, probabilities, success_prob = run_dqi_statevector(
            B=B,
            v=v,
            alpha=alpha,
            ell=ell,
            n_samples=n_dqi_samples,
            seed=trial_seed,
            decoder_mode=decoder,
            execution_mode="mixture",
        )
    except Exception as exc:
        row["run_status"] = "failed"
        row["status_reason_code"] = "dqi_runtime_error"
        row["status_reason"] = str(exc)
        row["run_error"] = str(exc)
        row["metric_availability"] = {k: "failed_upstream" for k in _metric_availability_completed()}
        return row

    runtime_sec = time.perf_counter() - start_time

    # Process results
    sample_arr = np.asarray(samples, dtype=np.int64)
    if len(sample_arr) > 0:
        F_values = f_table[sample_arr]
        # Use F values as G values for ILP-derived (exact encoding)
        G_values = F_values.copy()

        decision_metrics = compute_decision_metrics(
            F_values=F_values.tolist(),
            G_values=G_values.tolist(),
            f_opt=f_opt,
            k_eval=min(top_k_by_surrogate, len(sample_arr)),
            sampled_x=sample_arr.tolist(),
            optimum_x=optimum_x.tolist(),
            decode_success=decoder_success_rate,
            postselection_success=float(success_prob),
            retained_energy_eta=1.0,  # ILP-derived is exact
            candidate_source_label=ENCODING_NAME,
        )

        core = decision_metrics.get("core_decision", {})
        faith = decision_metrics.get("faithfulness", {})

        best_f = float(np.max(F_values))
        approx = float(best_f / f_opt) if abs(f_opt) > 1e-15 else None

        row["run_status"] = "completed"
        row["status_reason_code"] = "completed"
        row["status_reason"] = "completed"
        row["best_F"] = best_f
        row["approximation_ratio"] = approx
        row["best_sampled_F"] = core.get("best_sampled_F")
        row["best_sampled_G"] = float(np.max(probabilities)) if len(probabilities) else None
        row["top1_regret"] = core.get("top1_regret")
        row["topk_regret"] = core.get("topk_regret")
        row["optimum_recall_at_k"] = core.get("optimum_recall_at_k")
        row["best_top_G_sampled_F"] = core.get("best_top_G_sampled_F")
        row["spearman_rho_F_G"] = faith.get("spearman_rho_F_G")
        row["retained_energy_eta"] = 1.0  # ILP-derived is exact
        row["decision_distortion"] = core.get("decision_distortion")
        row["decode_success"] = decoder_success_rate
        row["postselection_success"] = float(success_prob)
        row["runtime_sec"] = runtime_sec
        row["metric_availability"] = _metric_availability_completed()
    else:
        row["run_status"] = "completed"
        row["status_reason_code"] = "completed_no_samples"
        row["status_reason"] = "DQI completed but no samples returned"
        row["metric_availability"] = {k: "no_samples" for k in _metric_availability_completed()}

    # Add resource estimates
    resource = dqi_resource_estimate(B, v, ell=ell, instance_id=instance_id, k_selected=m)
    row["resource_total_qubits_est_with_decode"] = resource.get("total_qubits_est_with_decode")
    row["resource_total_depth_est_with_decode"] = resource.get("total_depth_est_with_decode")
    row["resource_total_gate_est_with_decode"] = resource.get("total_gate_est_with_decode")

    # Add structure metrics
    structure = compute_structure_metrics(B, ell)
    for key in [
        "structure_collision_rate",
        "structure_ambiguous_syndrome_count",
        "structure_rank",
        "structure_rank_deficiency",
        "structure_decodable_syndrome_fraction",
    ]:
        row[key] = structure.get(key)

    return row


def run_ilp_derived_pricing(
    *,
    output_root: str = "results",
    p_instances: tuple[str, ...] = ("P3", "P4", "P5", "P6"),
    alpha_modes: tuple[str, ...] = ("paper", "uniform"),
    decoders: tuple[str, ...] = ("bp1", "bruteforce"),
    n_dqi_samples: int = 128,
    top_k_by_surrogate: int = 10,
    trial_seed: int = 42,
    k_terms: int = 15,
) -> dict:
    """Run ILP-derived exact experiments for pricing instances."""

    output_path = Path(output_root) / MATRIX_NAME / "runs"
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for instance_label in p_instances:
        # Parse n_features from label (P3 -> 3, P4 -> 4, etc.)
        n_features = int(instance_label[1:])
        instance_id = f"P{n_features}"

        print(f"Processing {instance_id} (n_features={n_features})...")

        for decoder in decoders:
            for alpha_mode in alpha_modes:
                print(f"  {decoder}/{alpha_mode}...")

                row = _run_single_ilp_experiment(
                    instance_id=instance_id,
                    n_features=n_features,
                    alpha_mode=alpha_mode,
                    decoder=decoder,
                    n_dqi_samples=n_dqi_samples,
                    top_k_by_surrogate=top_k_by_surrogate,
                    trial_seed=trial_seed,
                    k_terms=k_terms,
                    output_root=output_path,
                )

                results.append(row)

                # Save individual run
                run_path = output_path / f"{row['run_id']}.json"
                with open(run_path, "w") as f:
                    json.dump(row, f, indent=2, default=str)

                status = row.get("run_status", "unknown")
                print(f"    -> {status}")

    summary = {
        "stage": STAGE_NAME,
        "matrix": MATRIX_NAME,
        "encoding": ENCODING_NAME,
        "total_runs": len(results),
        "completed": sum(1 for r in results if r.get("run_status") == "completed"),
        "failed": sum(1 for r in results if r.get("run_status") == "failed"),
        "not_applicable": sum(1 for r in results if r.get("run_status") == "not_applicable"),
        "output_path": str(output_path),
    }

    return summary


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ILP-derived exact encoding experiments for P family.")
    p.add_argument("--output-root", default="results")
    p.add_argument("--p-instances", default="P3,P4,P5,P6")
    p.add_argument("--alpha-modes", default="paper,uniform")
    p.add_argument("--decoders", default="bp1,bruteforce")
    p.add_argument("--n-dqi-samples", type=int, default=128)
    p.add_argument("--top-k-by-surrogate", type=int, default=10)
    p.add_argument("--trial-seed", type=int, default=42)
    p.add_argument("--k-terms", type=int, default=15)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = run_ilp_derived_pricing(
        output_root=args.output_root,
        p_instances=_split_csv(args.p_instances),
        alpha_modes=_split_csv(args.alpha_modes),
        decoders=_split_csv(args.decoders),
        n_dqi_samples=int(args.n_dqi_samples),
        top_k_by_surrogate=int(args.top_k_by_surrogate),
        trial_seed=int(args.trial_seed),
        k_terms=int(args.k_terms),
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
