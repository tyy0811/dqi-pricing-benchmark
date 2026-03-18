"""
resources.py — DQI resource estimation by stage composition.

Computes gate counts, qubit requirements, and depth estimates for the
implemented DQI circuit stages, including explicit decode/uncompute estimates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.dicke_circuit import dicke_superposition_upper_bound
from src.decoder_bruteforce import lookup_table_reversible_resource_estimate
from src.decoder_bp1 import bp1_classical_oracle_resource_estimate
from src.dqi_circuit_qiskit import (
    qiskit_subset_validation_report,
    qiskit_transpiled_structural_count,
    subset_scope_metadata,
)
from src.dqi_state import weight_register_bits
from src.structure_metrics import compute_structure_metrics

DECODE_UNCOMPUTE_MODELS = {
    "lookup_table_reversible",
    "bp1_classical_oracle_estimate",
}
CONFIDENCE_LABELS = {
    "exact_formula",
    "analytic_upper_bound",
    "implementation_proxy",
    "not_included",
}
RESOURCE_VALIDATION_SCOPE = "analytic_full_pipeline_plus_qiskit_structural_subset"


def summarize_resource_scope(report: dict[str, Any]) -> dict[str, str]:
    """Return a compact reviewer-facing summary of resource validation scope."""
    qiskit_scope = (
        report.get("qiskit_transpiled_structural_count", {})
        .get("scope", {})
        .get("scope", "structural_subset_only")
    )
    return {
        "analytic_scope": "full_pipeline_estimate",
        "qiskit_scope": str(qiskit_scope),
        "note": (
            "Decode-inclusive analytic estimate and Qiskit structural subset counts "
            "are distinct validation scopes."
        ),
    }


def qiskit_subset_resource_report(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
    *,
    include_qiskit_structural_count: bool = True,
    basis_gates: list[str] | None = None,
    optimization_level: int = 1,
) -> dict[str, Any]:
    """Return a structural-subset-only Qiskit resource report.

    The scope block is the authoritative statement of what this report
    includes and excludes. Decode/uncompute is excluded, so this report does
    not make a full end-to-end parity claim.
    """
    scope = subset_scope_metadata()
    if not include_qiskit_structural_count:
        return {
            "scope": scope,
            "includes": list(scope["includes"]),
            "excludes": list(scope["excludes"]),
            "decode_uncompute_included": scope["decode_uncompute_included"],
            "subset_path": "implemented_structural_only",
            "distribution_basis": scope["distribution_basis"],
            "postselection_stage_present": False,
            "parity_claim": scope["parity_claim"],
            "status": "not_included",
            "reason": "disabled_by_configuration",
            "transpiled_depth": None,
            "cx_count": None,
            "qubit_count": None,
            "gate_counts": {},
        }

    report = qiskit_subset_validation_report(
        B=B,
        v=v,
        alpha=alpha,
        ell=ell,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )
    return {
        "scope": scope,
        "includes": list(scope["includes"]),
        "excludes": list(scope["excludes"]),
        "decode_uncompute_included": scope["decode_uncompute_included"],
        "subset_path": "implemented_structural_only",
        "distribution_basis": report["distribution_basis"],
        "postselection_stage_present": False,
        "parity_claim": scope["parity_claim"],
        "status": report["status"],
        "reason": report.get("reason"),
        "transpiled_depth": report["transpiled_depth"],
        "cx_count": report["cx_count"],
        "qubit_count": report["qubit_count"],
        "gate_counts": report["gate_counts"],
    }


def _validate_binary(v: np.ndarray, name: str = "v") -> None:
    """Raise ValueError if array is not binary (0/1 only)."""
    v_arr = np.asarray(v)
    if not np.all((v_arr == 0) | (v_arr == 1)):
        raise ValueError(f"{name} must be binary (0/1 only), got values: {np.unique(v_arr)}")


def _validate_shapes(B: np.ndarray, v: np.ndarray) -> None:
    """Raise ValueError if B and v have incompatible shapes."""
    if B.ndim != 2:
        raise ValueError(f"B must be 2D, got shape {B.shape}")
    if v.ndim != 1:
        raise ValueError(f"v must be 1D, got shape {v.shape}")
    if B.shape[0] != v.shape[0]:
        raise ValueError(f"B has {B.shape[0]} rows but v has {v.shape[0]} elements")


def _coherent_weight_register_overhead_estimate(ell: int) -> dict[str, Any]:
    """Estimate coherent weight-register preparation overhead.

    Uses a conservative generic state-preparation upper bound for q qubits:
    O(2^q) two-level rotations and entanglers.
    """
    q = weight_register_bits(int(ell))
    if q == 0:
        return {
            "weight_register_qubits": 0,
            "coherent_weight_register_gate_est": 0,
            "coherent_weight_register_depth_est": 0,
            "coherent_weight_register_ancilla_est": 0,
            "weight_register_overhead_status": "not_included",
            "confidence": "not_included",
            "assumptions": [
                "ell=0 requires no explicit weight-register qubits",
            ],
        }

    # Conservative upper bound proxy for generic q-qubit state preparation.
    entangler_upper = max(0, 2 ** (q + 1) - 2)
    single_qubit_upper = max(0, 2 ** (q + 1) - 2)
    gate_est = 4 * (entangler_upper + single_qubit_upper)
    depth_est = 2 * (2 ** q) + q

    return {
        "weight_register_qubits": int(q),
        "coherent_weight_register_gate_est": int(gate_est),
        "coherent_weight_register_depth_est": int(depth_est),
        "coherent_weight_register_ancilla_est": 0,
        "weight_register_overhead_status": "analytic_upper_bound",
        "confidence": "analytic_upper_bound",
        "assumptions": [
            "Generic q-qubit amplitude preparation upper bound (not transpiled)",
            "No explicit ancilla assumed for the coarse upper-bound model",
        ],
    }


def dqi_resource_estimate(
    B: np.ndarray,
    v: np.ndarray,
    ell: int,
    decode_uncompute_model: str = "lookup_table_reversible",
    bp1_max_iter: int = 16,
    include_qiskit_structural_count: bool = True,
    qiskit_basis_gates: list[str] | None = None,
    qiskit_optimization_level: int = 1,
    instance_id: str | None = None,
    k_selected: int | None = None,
    artifact_version: str | None = None,
    repo_commit: str | None = None,
) -> dict[str, Any]:
    """Full DQI resource estimate for a (B, v) surrogate artifact.

    Composes: dicke_prep + phase_kick + syndrome_compute + decode_uncompute + final_hadamard

    Args:
        B: (m, n) binary parity matrix.
        v: (m,) binary target parity vector.
        ell: Dicke weight bound.
        decode_uncompute_model: Decode/uncompute resource model.
            Supported: "lookup_table_reversible", "bp1_classical_oracle_estimate".
        bp1_max_iter: Iteration budget used by BP1 resource proxy.
        include_qiskit_structural_count: Whether to compute transpiled structural
            counts for the implemented subset.
        qiskit_basis_gates: Optional basis gate set for transpilation.
        qiskit_optimization_level: Qiskit transpiler optimization level.
        instance_id: Optional identifier for the instance.
        k_selected: Number of surrogate terms (from artifact if available).
        artifact_version: Version string from artifact.
        repo_commit: Commit hash from artifact.

    Returns:
        Resource estimate dictionary with full schema.

    Raises:
        ValueError: If v is not binary or shapes are incompatible.
    """
    B = np.asarray(B)
    v = np.asarray(v)

    _validate_shapes(B, v)
    _validate_binary(v, "v")
    if decode_uncompute_model not in DECODE_UNCOMPUTE_MODELS:
        raise ValueError(
            "decode_uncompute_model must be one of "
            f"{sorted(DECODE_UNCOMPUTE_MODELS)}, got {decode_uncompute_model!r}"
        )

    m, n = B.shape
    structure_metrics = compute_structure_metrics(B, ell=ell)

    alpha_struct = np.ones(ell + 1, dtype=np.float64)
    alpha_struct /= np.linalg.norm(alpha_struct)

    # Dicke superposition estimate
    dicke = dicke_superposition_upper_bound(m, ell)

    # Syndrome: one CNOT per nonzero entry in B
    syndrome_cnot_count = int(np.sum(B))

    # Phase kick: one Z per nonzero entry in v
    phase_z_count = int(np.sum(v))

    # Final Hadamard: one H per problem qubit
    final_h_count = n

    # Aggregates (decode-exclusive baseline)
    total_gate_est = (
        dicke["cnot_est"] +
        dicke["single_qubit_rot_est"] +
        syndrome_cnot_count +
        phase_z_count +
        final_h_count
    )

    # Depth estimate: Dicke + syndrome (1 layer) + phase (1 layer) + Hadamard (1 layer)
    total_depth_est = dicke["depth_est"] + 3

    if decode_uncompute_model == "lookup_table_reversible":
        decode_est = lookup_table_reversible_resource_estimate(B=B, ell=ell)
    else:
        decode_est = bp1_classical_oracle_resource_estimate(
            B=B,
            ell=ell,
            max_iter=bp1_max_iter,
        )

    decode_uncompute_gate_est = int(decode_est["decode_uncompute_gate_est"])
    decode_uncompute_depth_est = int(decode_est["decode_uncompute_depth_est"])
    decode_uncompute_ancilla_est = int(decode_est["decode_uncompute_ancilla_est"])

    # Collision-rate guard logic overhead: conservative penalty for ambiguous
    # syndromes that still require reversible reject-path logic.
    collision_guard_gate_est = 0
    collision_guard_depth_est = 0
    if decode_uncompute_model == "lookup_table_reversible":
        collision_rate = float(structure_metrics["decoder_collision_rate"])
        lookup_size = int(structure_metrics["estimated_lookup_table_size"])
        collision_guard_gate_est = int(np.ceil(collision_rate * lookup_size * max(2, 6 * n)))
        collision_guard_depth_est = int(np.ceil(collision_rate * max(1, lookup_size // 2)))
        decode_uncompute_gate_est += collision_guard_gate_est
        decode_uncompute_depth_est += collision_guard_depth_est

    decode_confidence = (
        "analytic_upper_bound"
        if decode_uncompute_model == "lookup_table_reversible"
        else "implementation_proxy"
    )
    weight_reg_est = _coherent_weight_register_overhead_estimate(ell=ell)
    weight_register_qubits = int(weight_reg_est["weight_register_qubits"])
    coherent_weight_register_gate_est = int(weight_reg_est["coherent_weight_register_gate_est"])
    coherent_weight_register_depth_est = int(weight_reg_est["coherent_weight_register_depth_est"])
    coherent_weight_register_ancilla_est = int(weight_reg_est["coherent_weight_register_ancilla_est"])
    weight_register_overhead_status = str(weight_reg_est["weight_register_overhead_status"])

    total_gate_est_with_decode = total_gate_est + decode_uncompute_gate_est
    total_depth_est_with_decode = total_depth_est + decode_uncompute_depth_est
    total_gate_est_with_decode_and_weight_register_est = (
        total_gate_est_with_decode + coherent_weight_register_gate_est
    )
    total_depth_est_with_decode_and_weight_register_est = (
        total_depth_est_with_decode + coherent_weight_register_depth_est
    )

    # Qubit breakdown
    n_problem_qubits = n
    m_error_qubits = m
    n_syndrome_qubits = n
    ancilla_qubits_est = 0
    total_qubits_est = m_error_qubits + n_syndrome_qubits + ancilla_qubits_est
    total_qubits_est_with_decode = (
        m_error_qubits + n_syndrome_qubits + max(ancilla_qubits_est, decode_uncompute_ancilla_est)
    )
    total_qubits_est_with_decode_and_weight_register_est = (
        m_error_qubits
        + n_syndrome_qubits
        + weight_register_qubits
        + max(
            ancilla_qubits_est,
            decode_uncompute_ancilla_est,
            coherent_weight_register_ancilla_est,
        )
    )

    implemented_resources = {
        "dicke_prep": {
            "gate_est": int(dicke["cnot_est"] + dicke["single_qubit_rot_est"]),
            "depth_est": int(dicke["depth_est"]),
            "ancilla_est": int(ancilla_qubits_est),
            "confidence": "analytic_upper_bound",
        },
        "phase_kick": {
            "gate_est": int(phase_z_count),
            "depth_est": 1,
            "ancilla_est": 0,
            "confidence": "exact_formula",
        },
        "syndrome_compute": {
            "gate_est": int(syndrome_cnot_count),
            "depth_est": 1,
            "ancilla_est": 0,
            "confidence": "exact_formula",
        },
        "hadamards": {
            "gate_est": int(final_h_count),
            "depth_est": 1,
            "ancilla_est": 0,
            "confidence": "exact_formula",
        },
    }

    estimated_resources = {
        "decode_uncompute": {
            "model": decode_est["decode_uncompute_model"],
            "status": decode_est["decode_uncompute_status"],
            "gate_est": decode_uncompute_gate_est,
            "depth_est": decode_uncompute_depth_est,
            "ancilla_est": decode_uncompute_ancilla_est,
            "confidence": decode_confidence,
            "details": {
                k: v
                for k, v in decode_est.items()
                if k not in {
                    "decode_uncompute_gate_est",
                    "decode_uncompute_depth_est",
                    "decode_uncompute_ancilla_est",
                    "decode_uncompute_model",
                    "decode_uncompute_status",
                }
            },
            "cost_drivers": {
                "estimated_lookup_table_size": int(structure_metrics["estimated_lookup_table_size"]),
                "number_of_unique_syndromes_decodable_at_ell": int(
                    structure_metrics["number_of_unique_syndromes_decodable_at_ell"]
                ),
                "decoder_collision_rate": float(structure_metrics["decoder_collision_rate"]),
                "collision_guard_gate_est": int(collision_guard_gate_est),
                "collision_guard_depth_est": int(collision_guard_depth_est),
            },
        },
        "coherent_weight_register_overhead": {
            "weight_register_qubits": weight_register_qubits,
            "gate_est": coherent_weight_register_gate_est,
            "depth_est": coherent_weight_register_depth_est,
            "ancilla_est": coherent_weight_register_ancilla_est,
            "status": weight_register_overhead_status,
            "confidence": weight_reg_est["confidence"],
            "details": {
                "assumptions": weight_reg_est["assumptions"],
            },
        },
    }
    analytic_estimates = estimated_resources

    total_estimated_resources = {
        "total_gate_est_without_decode": int(total_gate_est),
        "total_depth_est_without_decode": int(total_depth_est),
        "total_qubits_est_without_decode": int(total_qubits_est),
        "total_gate_est_with_decode": int(total_gate_est_with_decode),
        "total_depth_est_with_decode": int(total_depth_est_with_decode),
        "total_qubits_est_with_decode": int(total_qubits_est_with_decode),
        "total_gate_est_with_decode_and_weight_register_est": int(
            total_gate_est_with_decode_and_weight_register_est
        ),
        "total_depth_est_with_decode_and_weight_register_est": int(
            total_depth_est_with_decode_and_weight_register_est
        ),
        "total_qubits_est_with_decode_and_weight_register_est": int(
            total_qubits_est_with_decode_and_weight_register_est
        ),
        # Backward-compatible nested totals.
        "total_without_decode": {
            "gate_est": int(total_gate_est),
            "depth_est": int(total_depth_est),
            "qubits_est": int(total_qubits_est),
        },
        "total_with_decode": {
            "gate_est": int(total_gate_est_with_decode),
            "depth_est": int(total_depth_est_with_decode),
            "qubits_est": int(total_qubits_est_with_decode),
        },
        "total_with_decode_and_weight_register_est": {
            "gate_est": int(total_gate_est_with_decode_and_weight_register_est),
            "depth_est": int(total_depth_est_with_decode_and_weight_register_est),
            "qubits_est": int(total_qubits_est_with_decode_and_weight_register_est),
        },
    }

    total_resources = {
        "total_without_decode": {
            "gate_est": int(total_gate_est),
            "depth_est": int(total_depth_est),
            "qubits_est": int(total_qubits_est),
            "confidence": "mixed",
        },
        "total_with_decode": {
            "gate_est": int(total_gate_est_with_decode),
            "depth_est": int(total_depth_est_with_decode),
            "qubits_est": int(total_qubits_est_with_decode),
            "confidence": "mixed",
        },
        "total_with_decode_and_weight_register_est": {
            "gate_est": int(total_gate_est_with_decode_and_weight_register_est),
            "depth_est": int(total_depth_est_with_decode_and_weight_register_est),
            "qubits_est": int(total_qubits_est_with_decode_and_weight_register_est),
            "confidence": "mixed",
        },
    }

    if include_qiskit_structural_count:
        qiskit_structural = qiskit_transpiled_structural_count(
            B=B,
            v=v,
            alpha=alpha_struct,
            ell=ell,
            basis_gates=qiskit_basis_gates,
            optimization_level=qiskit_optimization_level,
        )
    else:
        qiskit_structural = {
            "category": "qiskit_transpiled_structural_count",
            "status": "not_included",
            "reason": "disabled_by_configuration",
            "scope": {
                **subset_scope_metadata(),
                "includes": [],
                "dicke_prep_status": "unknown",
                "category": "implemented_subset_only",
            },
            "qubit_count": None,
            "transpiled_depth": None,
            "cx_count": None,
            "gate_counts": {},
            "basis_gates": qiskit_basis_gates or ["cx", "rz", "sx", "x"],
            "optimization_level": int(qiskit_optimization_level),
        }
    qiskit_subset_resources = qiskit_subset_resource_report(
        B=B,
        v=v,
        alpha=alpha_struct,
        ell=ell,
        include_qiskit_structural_count=include_qiskit_structural_count,
        basis_gates=qiskit_basis_gates,
        optimization_level=qiskit_optimization_level,
    )

    return {
        "instance_id": instance_id,
        "n": n,
        "m": m,
        "k_selected": k_selected,
        "ell": ell,
        "dicke_method": dicke["method"],
        "dicke_cnot_est": dicke["cnot_est"],
        "dicke_single_qubit_rot_est": dicke["single_qubit_rot_est"],
        "dicke_depth_est": dicke["depth_est"],
        "syndrome_cnot_count": syndrome_cnot_count,
        "phase_z_count": phase_z_count,
        "final_h_count": final_h_count,
        "total_gate_est": total_gate_est,
        "total_depth_est": total_depth_est,
        "decode_uncompute_gate_est": decode_uncompute_gate_est,
        "decode_uncompute_depth_est": decode_uncompute_depth_est,
        "decode_uncompute_ancilla_est": decode_uncompute_ancilla_est,
        "decode_uncompute_model": decode_est["decode_uncompute_model"],
        "decode_uncompute_status": decode_est["decode_uncompute_status"],
        "decode_uncompute_confidence": decode_confidence,
        "total_gate_est_with_decode": int(total_gate_est_with_decode),
        "total_depth_est_with_decode": int(total_depth_est_with_decode),
        "total_gate_est_with_decode_and_weight_register_est": int(
            total_gate_est_with_decode_and_weight_register_est
        ),
        "total_depth_est_with_decode_and_weight_register_est": int(
            total_depth_est_with_decode_and_weight_register_est
        ),
        "n_problem_qubits": n_problem_qubits,
        "m_error_qubits": m_error_qubits,
        "n_syndrome_qubits": n_syndrome_qubits,
        "ancilla_qubits_est": ancilla_qubits_est,
        "total_qubits_est": total_qubits_est,
        "total_qubits_est_with_decode": int(total_qubits_est_with_decode),
        "total_qubits_est_with_decode_and_weight_register_est": int(
            total_qubits_est_with_decode_and_weight_register_est
        ),
        "weight_register_qubits": weight_register_qubits,
        "weight_register_overhead_status": weight_register_overhead_status,
        "coherent_weight_register_gate_est": coherent_weight_register_gate_est,
        "coherent_weight_register_depth_est": coherent_weight_register_depth_est,
        "coherent_weight_register_ancilla_est": coherent_weight_register_ancilla_est,
        "row_weight_mean": float(structure_metrics["row_weight_mean"]),
        "row_weight_max": int(structure_metrics["row_weight_max"]),
        "col_weight_mean": float(structure_metrics["col_weight_mean"]),
        "col_weight_max": int(structure_metrics["col_weight_max"]),
        "syndrome_density": float(structure_metrics["syndrome_density"]),
        "number_of_unique_syndromes_decodable_at_ell": int(
            structure_metrics["number_of_unique_syndromes_decodable_at_ell"]
        ),
        "decoder_collision_rate": float(structure_metrics["decoder_collision_rate"]),
        "estimated_lookup_table_size": int(structure_metrics["estimated_lookup_table_size"]),
        "structure_metrics": structure_metrics,
        "decode_cost_drivers": estimated_resources["decode_uncompute"]["cost_drivers"],
        "implemented_resources": implemented_resources,
        "estimated_resources": estimated_resources,
        "analytic_estimates": analytic_estimates,
        "total_resources": total_resources,
        "total_estimated_resources": total_estimated_resources,
        "analytic_estimate": {
            "category": "analytic_estimate",
            "status": "ok",
            "scope": "full_pipeline_estimate_with_decode_model",
            "total_resources": total_resources,
        },
        "resource_validation_scope": RESOURCE_VALIDATION_SCOPE,
        "qiskit_transpiled_structural_count": qiskit_structural,
        "qiskit_subset_resource_report": qiskit_subset_resources,
        "resource_category_note": (
            "analytic_estimate and qiskit_transpiled_structural_count are "
            "different categories and are not direct substitutes."
        ),
        "decode_uncompute_details": estimated_resources["decode_uncompute"]["details"],
        "prefix_gate_est": int(total_gate_est),
        "prefix_depth_est": int(total_depth_est),
        "decode_model_gate_est": int(decode_uncompute_gate_est),
        "decode_model_depth_est": int(decode_uncompute_depth_est),
        "rank": int(structure_metrics["rank"]),
        "rank_deficiency": int(structure_metrics["rank_deficiency"]),
        "decodable_syndrome_fraction": float(structure_metrics["decodable_syndrome_fraction"]),
        "ambiguous_syndrome_count": int(structure_metrics["ambiguous_syndrome_count"]),
        "register_assumptions": {
            "m_error_qubits": "error register holding weight-t error patterns",
            "n_syndrome_qubits": "syndrome register, output of B^T @ y mod 2",
            "ancilla_qubits_est": "additional ancilla for Dicke prep if needed (0 for SCS)",
            "decode_uncompute_ancilla_est": (
                "additional workspace for decode/uncompute model "
                f"({decode_uncompute_model})"
            ),
            "weight_register_qubits": "explicit coherent weight-register size for t in [0, ell]",
        },
        "included_stages": [
            "dicke_prep",
            "phase_kick",
            "syndrome_compute",
            "decode_uncompute",
            "final_hadamard",
        ],
        "excluded_stages": [],
        "artifact_version": artifact_version,
        "repo_commit": repo_commit,
        "assumptions": [
            "Dicke prep: deterministic SCS construction (Bärtschi-Eidenbenz 2019)",
            "Superposition: naive sum over t=0..ell, not optimized weighted prep",
            "Syndrome: one CNOT per nonzero entry in B",
            "Phase kick: one Z per nonzero entry in v",
            (
                "Decode/uncompute: explicit "
                f"{decode_uncompute_model} estimate "
                f"({decode_est['decode_uncompute_status']})"
            ),
            (
                "Coherent weight register: "
                f"{weight_register_overhead_status} overhead model"
            ),
            (
                "Qiskit structural counts cover only implemented subset "
                "(decode/uncompute excluded)."
            ),
        ],
        "reference": "arXiv:1904.07358",
    }


def resource_estimate_from_artifact(
    artifact: dict,
    ell: int,
    decode_uncompute_model: str = "lookup_table_reversible",
    bp1_max_iter: int = 16,
    include_qiskit_structural_count: bool = True,
    qiskit_basis_gates: list[str] | None = None,
    qiskit_optimization_level: int = 1,
) -> dict[str, Any]:
    """Compute DQI resource estimate from a loaded surrogate artifact.

    Extracts B, v, and metadata from artifact and delegates to dqi_resource_estimate.

    Args:
        artifact: Surrogate artifact dictionary (as loaded from JSON or build_surrogate_artifact).
        ell: Dicke weight bound.
        decode_uncompute_model: Decode/uncompute model name.
        bp1_max_iter: BP1 iteration budget for analytic proxy.
        include_qiskit_structural_count: Whether to include Qiskit structural counts.
        qiskit_basis_gates: Optional basis gate set for transpilation.
        qiskit_optimization_level: Qiskit transpiler optimization level.

    Returns:
        Resource estimate dictionary.

    Raises:
        ValueError: If artifact is missing required fields.
    """
    if "B" not in artifact:
        raise ValueError("Artifact missing required field 'B'")
    if "v" not in artifact:
        raise ValueError("Artifact missing required field 'v'")

    B = np.asarray(artifact["B"], dtype=np.int8)
    v = np.asarray(artifact["v"], dtype=np.int8)

    source_meta = artifact.get("source_metadata", {})
    instance_id = source_meta.get("instance_id", artifact.get("instance_id"))
    repo_commit = source_meta.get("repo_commit", artifact.get("repo_commit"))
    artifact_version = artifact.get("artifact_version")
    k_selected = artifact.get("k_selected")

    return dqi_resource_estimate(
        B=B,
        v=v,
        ell=ell,
        decode_uncompute_model=decode_uncompute_model,
        bp1_max_iter=bp1_max_iter,
        include_qiskit_structural_count=include_qiskit_structural_count,
        qiskit_basis_gates=qiskit_basis_gates,
        qiskit_optimization_level=qiskit_optimization_level,
        instance_id=instance_id,
        k_selected=k_selected,
        artifact_version=artifact_version,
        repo_commit=repo_commit,
    )


def resource_report(
    artifact_path: str,
    ell: int,
    decode_uncompute_model: str = "lookup_table_reversible",
    bp1_max_iter: int = 16,
    include_qiskit_structural_count: bool = True,
    qiskit_basis_gates: list[str] | None = None,
    qiskit_optimization_level: int = 1,
) -> dict[str, Any]:
    """Load surrogate artifact from path, compute resources, return full report.

    Args:
        artifact_path: Path to surrogate artifact JSON file.
        ell: Dicke weight bound.
        decode_uncompute_model: Decode/uncompute model name.
        bp1_max_iter: BP1 iteration budget for analytic proxy.
        include_qiskit_structural_count: Whether to include Qiskit structural counts.
        qiskit_basis_gates: Optional basis gate set for transpilation.
        qiskit_optimization_level: Qiskit transpiler optimization level.

    Returns:
        Resource estimate dictionary.

    Raises:
        FileNotFoundError: If artifact file does not exist.
        ValueError: If artifact is malformed.
    """
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    with open(path) as f:
        artifact = json.load(f)

    return resource_estimate_from_artifact(
        artifact,
        ell,
        decode_uncompute_model=decode_uncompute_model,
        bp1_max_iter=bp1_max_iter,
        include_qiskit_structural_count=include_qiskit_structural_count,
        qiskit_basis_gates=qiskit_basis_gates,
        qiskit_optimization_level=qiskit_optimization_level,
    )


def export_resources_json(
    reports: list[dict],
    output_path: str,
) -> None:
    """Export resource reports to aggregated JSON file.

    Args:
        reports: List of resource estimate dictionaries.
        output_path: Path to output JSON file.
    """
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reports": reports,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
