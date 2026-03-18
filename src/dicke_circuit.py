"""
dicke_circuit.py — Analytical resource estimates for Dicke state preparation.

Implements resource formulas based on Bärtschi-Eidenbenz (2019) deterministic
SCS (Split-Cyclic-Shift) construction for Dicke states.

Reference: Bärtschi & Eidenbenz, "Deterministic Preparation of Dicke States,"
           arXiv:1904.07358 (2019).
"""

from __future__ import annotations


def dicke_resources(m: int, t: int) -> dict:
    """Analytic resource estimate for preparing |D_t^m⟩ via deterministic SCS construction.

    For the trivial cases:
    - t=0: |D_0^m⟩ = |00...0⟩, no gates needed
    - t=m: |D_m^m⟩ = |11...1⟩, just X gates

    For general t in (0, m):
    - CNOTs ≈ t(m-t) from the SCS structure
    - Single-qubit rotations ≈ (m-1) controlled-Ry gates
    - Depth ≈ m (linear in qubit count)

    Args:
        m: Number of qubits.
        t: Target Hamming weight.

    Returns:
        dict with keys: cnot_est, single_qubit_rot_est, depth_est, method, reference
    """
    if t < 0 or t > m:
        raise ValueError(f"t={t} must be in [0, m={m}]")
    if m < 0:
        raise ValueError(f"m={m} must be non-negative")

    # Trivial cases
    if t == 0:
        return {
            "cnot_est": 0,
            "single_qubit_rot_est": 0,
            "depth_est": 0,
            "method": "bartschi_eidenbenz_scs",
            "reference": "arXiv:1904.07358",
        }

    if t == m:
        return {
            "cnot_est": 0,
            "single_qubit_rot_est": m,
            "depth_est": 1,
            "method": "bartschi_eidenbenz_scs",
            "reference": "arXiv:1904.07358",
        }

    # General case: SCS construction
    cnot_est = t * (m - t)
    single_qubit_rot_est = m - 1
    depth_est = m

    return {
        "cnot_est": cnot_est,
        "single_qubit_rot_est": single_qubit_rot_est,
        "depth_est": depth_est,
        "method": "bartschi_eidenbenz_scs",
        "reference": "arXiv:1904.07358",
    }


def dicke_superposition_upper_bound(m: int, ell: int) -> dict:
    """Naive upper-bound estimate for Σ_{t=0}^ell α_t |D_t^m⟩ by aggregating per-t Dicke costs.

    NOT a compiled circuit count for a specific weighted-superposition construction.
    This is an upper bound assuming independent preparation of each Dicke state.

    Args:
        m: Number of qubits in error register.
        ell: Maximum Hamming weight (inclusive).

    Returns:
        dict with keys: cnot_est, single_qubit_rot_est, depth_est, method,
                        per_weight_breakdown
    """
    if ell < 0:
        raise ValueError(f"ell={ell} must be non-negative")
    if ell > m:
        raise ValueError(f"ell={ell} cannot exceed m={m}")

    per_weight = []
    total_cnot = 0
    total_single_qubit = 0
    max_depth = 0

    for t in range(ell + 1):
        res_t = dicke_resources(m, t)
        per_weight.append({
            "t": t,
            "cnot_est": res_t["cnot_est"],
            "single_qubit_rot_est": res_t["single_qubit_rot_est"],
            "depth_est": res_t["depth_est"],
        })
        total_cnot += res_t["cnot_est"]
        total_single_qubit += res_t["single_qubit_rot_est"]
        max_depth = max(max_depth, res_t["depth_est"])

    return {
        "cnot_est": total_cnot,
        "single_qubit_rot_est": total_single_qubit,
        "depth_est": max_depth,
        "method": "bartschi_eidenbenz_scs_upper_bound",
        "reference": "arXiv:1904.07358",
        "per_weight_breakdown": per_weight,
    }
