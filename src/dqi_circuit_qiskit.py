"""
dqi_circuit_qiskit.py — Qiskit port of DQI circuit.

Implements a structural thin-port/demo of the DQI circuit in Qiskit.

Important limitations:
- Dicke preparation is approximate.
- Decode/uncompute is not implemented as a reversible circuit.
- Therefore this module is useful for circuit structure demos only, and is
  not distribution-comparable to the main NumPy DQI statevector simulator.

Reference: Jordan et al. (2024) arXiv:2408.08292, Section 8.1.2
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def check_qiskit():
    """Check if Qiskit is available."""
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is not installed. Install with: pip install qiskit"
        )


def build_dicke_prep_circuit(
    qr: "QuantumRegister",
    m: int,
    ell: int,
    alpha: np.ndarray,
) -> "QuantumCircuit":
    """Build approximate Dicke state preparation circuit.

    This is a simplified version for small m and ell.
    For exact Dicke preparation, see Bartschi & Eidenbenz (2019).

    Args:
        qr: Quantum register of m qubits.
        m: Number of qubits.
        ell: Maximum weight.
        alpha: Weight coefficients.

    Returns:
        QuantumCircuit preparing approximate Dicke superposition.
    """
    check_qiskit()

    qc = QuantumCircuit(qr, name='dicke_prep')

    # Simplified: prepare uniform superposition over low-weight states
    # This is NOT the correct Dicke preparation but demonstrates structure
    for i in range(min(ell + 1, m)):
        # Apply Hadamard to first ell+1 qubits
        qc.h(qr[i])

    return qc


def build_phase_kick_circuit(
    qr: "QuantumRegister",
    v: np.ndarray,
) -> "QuantumCircuit":
    """Build phase kick circuit: apply Z where v_i = 1.

    Args:
        qr: Quantum register.
        v: Target parity vector.

    Returns:
        QuantumCircuit applying phase kick.
    """
    check_qiskit()

    qc = QuantumCircuit(qr, name='phase_kick')

    for i, vi in enumerate(v):
        if vi == 1:
            qc.z(qr[i])

    return qc


def build_syndrome_circuit(
    error_qr: "QuantumRegister",
    syndrome_qr: "QuantumRegister",
    B: np.ndarray,
) -> "QuantumCircuit":
    """Build syndrome computation circuit: CNOT network for B^T @ y.

    Args:
        error_qr: Error register (m qubits).
        syndrome_qr: Syndrome register (n qubits).
        B: (m, n) parity matrix.

    Returns:
        QuantumCircuit computing syndrome.
    """
    check_qiskit()

    m, n = B.shape
    qc = QuantumCircuit(error_qr, syndrome_qr, name='syndrome')

    # For each entry B[i,j] = 1, add CNOT from error_qr[i] to syndrome_qr[j]
    for i in range(m):
        for j in range(n):
            if B[i, j] == 1:
                qc.cx(error_qr[i], syndrome_qr[j])

    return qc


def build_dqi_circuit(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
) -> "QuantumCircuit":
    """Build the implemented-subset structural DQI circuit.

    This circuit is intentionally limited to the structural subset:
    approximate Dicke preparation, phase kick, syndrome compute, and final
    Hadamards. Decode/uncompute is excluded, so this is not a full end-to-end
    parity implementation.

    Args:
        B: (m, n) parity matrix.
        v: (m,) target parity vector.
        alpha: Dicke weight coefficients.
        ell: Maximum weight.

    Returns:
        QuantumCircuit for the implemented structural subset only.
    """
    check_qiskit()

    m, n = B.shape

    # Registers
    error_qr = QuantumRegister(m, 'error')
    syndrome_qr = QuantumRegister(n, 'syndrome')
    cr = ClassicalRegister(n, 'measure')

    qc = QuantumCircuit(error_qr, syndrome_qr, cr)

    # Step 1: Dicke preparation (simplified)
    dicke_circuit = build_dicke_prep_circuit(error_qr, m, ell, alpha)
    qc.compose(dicke_circuit, qubits=error_qr, inplace=True)

    qc.barrier()

    # Step 2: Phase kick
    phase_circuit = build_phase_kick_circuit(error_qr, v)
    qc.compose(phase_circuit, qubits=error_qr, inplace=True)

    qc.barrier()

    # Step 3: Syndrome computation
    syndrome_circuit = build_syndrome_circuit(error_qr, syndrome_qr, B)
    qc.compose(syndrome_circuit, qubits=list(error_qr) + list(syndrome_qr), inplace=True)

    qc.barrier()

    # Step 4: Decode and uncompute (skipped in this structural demo)
    # In full implementation, would apply reversible decoder circuit

    # Step 5: Hadamard on syndrome register
    for i in range(n):
        qc.h(syndrome_qr[i])

    # Measure syndrome register
    qc.measure(syndrome_qr, cr)

    return qc


def subset_scope_metadata() -> dict[str, Any]:
    """Return explicit scope metadata for the implemented Qiskit subset path."""
    return {
        "scope": "structural_subset_only",
        "includes": [
            "approximate_low_weight_superposition",
            "phase_kick",
            "syndrome_compute",
            "final_hadamards",
        ],
        "excludes": ["decode_uncompute"],
        "decode_uncompute_included": False,
        "postselection_stage_present": False,
        "distribution_basis": "syndrome_register_after_final_hadamards",
        "subset_path": "implemented_structural_only",
        "parity_claim": "no_full_end_to_end_parity_claim",
    }


def build_dqi_structural_resource_circuit(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
) -> dict[str, Any]:
    """Build a simplified DQI circuit scoped for structural resource extraction.

    Scope:
    - Includes implemented subset only: Dicke prep (approximate), phase kick,
      syndrome compute, final Hadamards.
    - Excludes decode/uncompute unless implemented (currently excluded).
    """
    check_qiskit()
    qc = build_dqi_circuit(B=B, v=v, alpha=alpha, ell=ell)
    return {
        "circuit": qc,
        "scope": {
            **subset_scope_metadata(),
            "dicke_prep_status": "approximate_structural",
            "category": "implemented_subset_only",
        },
    }


def qiskit_transpiled_structural_count(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
    *,
    basis_gates: Optional[list[str]] = None,
    optimization_level: int = 1,
) -> dict[str, Any]:
    """Return transpiled structural resource counts for the implemented subset.

    This function is intentionally scoped for Stage-3 structural comparisons and
    must not be interpreted as full-pipeline resource validation.
    """
    if not QISKIT_AVAILABLE:
        return {
            "category": "qiskit_transpiled_structural_count",
            "status": "not_available",
            "reason": "qiskit_not_installed",
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
            "basis_gates": basis_gates or ["cx", "rz", "sx", "x"],
            "optimization_level": int(optimization_level),
        }

    if basis_gates is None:
        basis_gates = ["cx", "rz", "sx", "x"]

    try:
        built = build_dqi_structural_resource_circuit(B=B, v=v, alpha=alpha, ell=ell)
        circuit = built["circuit"]
        scope = built["scope"]
        tqc = transpile(
            circuit,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )
        op_counts = {str(k): int(v) for k, v in tqc.count_ops().items()}
        cx_count = int(op_counts.get("cx", 0))
        return {
            "category": "qiskit_transpiled_structural_count",
            "status": "ok",
            "reason": None,
            "scope": scope,
            "qubit_count": int(circuit.num_qubits),
            "transpiled_depth": int(tqc.depth()),
            "cx_count": cx_count,
            "gate_counts": op_counts,
            "basis_gates": list(basis_gates),
            "optimization_level": int(optimization_level),
        }
    except Exception as exc:
        return {
            "category": "qiskit_transpiled_structural_count",
            "status": "error",
            "reason": str(exc),
            "scope": {
                **subset_scope_metadata(),
                "dicke_prep_status": "approximate_structural",
                "category": "implemented_subset_only",
            },
            "qubit_count": None,
            "transpiled_depth": None,
            "cx_count": None,
            "gate_counts": {},
            "basis_gates": list(basis_gates),
            "optimization_level": int(optimization_level),
        }


def run_qiskit_statevector(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
) -> np.ndarray:
    """Legacy alias for the implemented Qiskit structural-subset distribution.

    This wrapper preserves backward compatibility. It exposes only the subset
    path implemented in this module: approximate low-weight superposition,
    phase kick, syndrome compute, and final Hadamards. Decode/uncompute is
    excluded, so this helper does not make a full end-to-end parity claim.
    """
    return qiskit_subset_syndrome_distribution(B=B, v=v, alpha=alpha, ell=ell)


def qiskit_subset_syndrome_distribution(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
) -> np.ndarray:
    """Return the structural-subset syndrome distribution after final Hadamards.

    Decode/uncompute is excluded. This helper does not make a full end-to-end
    parity claim.
    """
    probs_syndrome, _ = qiskit_subset_syndrome_distribution_with_metadata(
        B=B,
        v=v,
        alpha=alpha,
        ell=ell,
    )

    return probs_syndrome


def qiskit_subset_syndrome_distribution_with_metadata(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
) -> tuple[np.ndarray, list[int]]:
    """Return subset syndrome probabilities plus the qargs ordering used."""
    check_qiskit()

    m, n = B.shape

    error_qr = QuantumRegister(m, "error")
    syndrome_qr = QuantumRegister(n, "syndrome")
    qc = QuantumCircuit(error_qr, syndrome_qr)

    for i in range(min(ell + 1, m)):
        qc.h(error_qr[i])

    for i, vi in enumerate(v):
        if vi == 1:
            qc.z(error_qr[i])

    for i in range(m):
        for j in range(n):
            if B[i, j] == 1:
                qc.cx(error_qr[i], syndrome_qr[j])

    for i in range(n):
        qc.h(syndrome_qr[i])

    sv = Statevector(qc)
    syndrome_qargs = [qc.find_bit(qubit).index for qubit in syndrome_qr]
    probs_syndrome = np.asarray(sv.probabilities(qargs=syndrome_qargs), dtype=np.float64)

    total = float(np.sum(probs_syndrome))
    if total > 1e-15:
        probs_syndrome /= total

    return probs_syndrome, syndrome_qargs


def qiskit_subset_validation_report(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
    *,
    basis_gates: Optional[list[str]] = None,
    optimization_level: int = 1,
) -> dict[str, Any]:
    """Return a structural-subset-only Qiskit validation payload.

    Decode/uncompute is excluded. This report is not a full end-to-end parity
    validation claim.
    """
    scope = subset_scope_metadata()
    if not QISKIT_AVAILABLE:
        return {
            "status": "not_available",
            "reason": "qiskit_not_installed",
            "scope": scope,
            "prob_dist": None,
            "syndrome_qargs": None,
            "distribution_basis": scope["distribution_basis"],
            "decode_uncompute_included": scope["decode_uncompute_included"],
            "includes": list(scope["includes"]),
            "excludes": list(scope["excludes"]),
            "parity_claim": scope["parity_claim"],
            "postselection_stage_present": False,
            "transpiled_depth": None,
            "cx_count": None,
            "qubit_count": None,
            "gate_counts": {},
        }
    prob_dist = None
    syndrome_qargs = None
    status = "ok"
    reason = None

    try:
        prob_dist, syndrome_qargs = qiskit_subset_syndrome_distribution_with_metadata(
            B=B,
            v=v,
            alpha=alpha,
            ell=ell,
        )
    except Exception as exc:
        status = "runtime_error"
        reason = f"subset_statevector_failed: {exc}"

    try:
        structural_counts = qiskit_transpiled_structural_count(
            B=B,
            v=v,
            alpha=alpha,
            ell=ell,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )
    except Exception as exc:
        structural_counts = {
            "status": "runtime_error",
            "reason": f"subset_transpile_failed: {exc}",
            "transpiled_depth": None,
            "cx_count": None,
            "qubit_count": None,
            "gate_counts": {},
        }

    if status == "ok" and structural_counts.get("status") != "ok":
        status = str(structural_counts.get("status"))
        reason = structural_counts.get("reason")

    return {
        "status": status,
        "reason": reason,
        "scope": scope,
        "prob_dist": prob_dist,
        "syndrome_qargs": syndrome_qargs,
        "distribution_basis": scope["distribution_basis"],
        "decode_uncompute_included": scope["decode_uncompute_included"],
        "includes": list(scope["includes"]),
        "excludes": list(scope["excludes"]),
        "parity_claim": scope["parity_claim"],
        "postselection_stage_present": False,
        "transpiled_depth": structural_counts.get("transpiled_depth"),
        "cx_count": structural_counts.get("cx_count"),
        "qubit_count": structural_counts.get("qubit_count"),
        "gate_counts": structural_counts.get("gate_counts", {}),
    }


def compare_frameworks(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
) -> dict:
    """Return an explicit non-validation status for framework comparison.

    Distributional comparison is intentionally disabled because this Qiskit
    path is a structural demo (approximate Dicke prep, no decode/uncompute),
    while src.dqi_state.run_dqi_statevector uses a different simulation model.
    """
    return {
        'status': 'disabled',
        'reason': (
            "Distribution comparison disabled: Qiskit path is a structural demo "
            "with approximate Dicke preparation and decode/uncompute excluded, so "
            "it does not support a full end-to-end parity claim against "
            "src.dqi_state.run_dqi_statevector."
        ),
        'comparable': False,
    }


def decode_inclusive_scope_metadata() -> dict[str, Any]:
    """Return explicit scope metadata for decode-inclusive postselected validation."""
    return {
        "scope": "decode_inclusive_postselected",
        "includes": [
            "dicke_superposition",
            "phase_kick",
            "syndrome_compute",
            "decode_postselection",
            "final_hadamards",
        ],
        "excludes": [],
        "decode_uncompute_included": True,
        "postselection_stage_present": True,
        "distribution_basis": "syndrome_register_after_decode_postselection_and_hadamards",
        "subset_path": "full_pipeline_postselected",
        "parity_claim": "full_end_to_end_parity_with_postselection",
    }


def build_dqi_circuit_for_postselection(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
) -> "QuantumCircuit":
    """Build DQI circuit for postselected decode-inclusive comparison.

    This circuit implements steps 1-3 of DQI:
    1. Dicke superposition preparation
    2. Phase kick
    3. Syndrome computation

    Decode/uncompute (step 4) is handled via classical postselection on the
    statevector, and Hadamards (step 5) are applied after postselection.

    Convention: We use direct qubit-index mapping (error_qr[i] = error bit i,
    syndrome_qr[j] = syndrome bit j) and handle bit ordering in state extraction.

    Args:
        B: (m, n) parity matrix.
        v: (m,) target parity vector.
        alpha: Dicke weight coefficients (ell+1,).
        ell: Maximum error weight.

    Returns:
        QuantumCircuit for statevector extraction (no measurement).
    """
    check_qiskit()
    from math import comb
    from itertools import combinations

    m, n = B.shape

    error_qr = QuantumRegister(m, "error")
    syndrome_qr = QuantumRegister(n, "syndrome")
    qc = QuantumCircuit(error_qr, syndrome_qr)

    # Step 1: Prepare weighted Dicke superposition
    # Use direct index mapping: error_qr[i] corresponds to error bit i (position i from left in MSB)
    # In Qiskit statevector, error_qr[i] is bit i of the index
    dim = 2 ** m
    dicke_state = np.zeros(dim, dtype=np.complex128)

    for t in range(ell + 1):
        if t > m:
            continue
        norm_t = 1.0 / np.sqrt(comb(m, t))
        for positions in combinations(range(m), t):
            # positions are the indices of active error bits
            # In Qiskit, position i sets bit i of the statevector index
            y = sum(1 << pos for pos in positions)
            dicke_state[y] += alpha[t] * norm_t

    # Normalize
    dicke_norm = np.linalg.norm(dicke_state)
    if dicke_norm > 1e-15:
        dicke_state /= dicke_norm

    # Initialize error register to Dicke superposition
    qc.initialize(dicke_state, error_qr)

    qc.barrier()

    # Step 2: Phase kick - apply Z where v_i = 1
    # v[i] corresponds to error bit i → error_qr[i]
    for i, vi in enumerate(v):
        if vi == 1:
            qc.z(error_qr[i])

    qc.barrier()

    # Step 3: Syndrome computation - CNOT network for B^T @ y
    # B[i, j] = 1 means error bit i contributes to syndrome bit j
    # error_qr[i] controls syndrome_qr[j]
    for i in range(m):
        for j in range(n):
            if B[i, j] == 1:
                qc.cx(error_qr[i], syndrome_qr[j])

    # No measurement - circuit is used for statevector extraction
    return qc


def _reverse_bits(idx: int, nbits: int) -> int:
    """Reverse the bits of an integer."""
    result = 0
    for i in range(nbits):
        if (idx >> i) & 1:
            result |= 1 << (nbits - 1 - i)
    return result


def qiskit_decode_inclusive_distribution(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
    decoder: object,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Compute decode-inclusive syndrome distribution using postselection.

    This implements full DQI pipeline validation:
    1. Build Qiskit circuit for steps 1-3
    2. Extract full statevector |y>|s>
    3. Apply decode postselection (keep amplitude where y == decode(s))
    4. Apply Hadamard transform on syndrome register
    5. Return probability distribution

    Note: Qiskit uses direct qubit indexing (qubit i = bit i of index).
    NumPy uses MSB-first (position i from left = bit m-1-i of index).
    We convert between these conventions.

    Args:
        B: (m, n) parity matrix.
        v: (m,) target parity vector.
        alpha: Dicke weight coefficients.
        ell: Maximum error weight.
        decoder: Decoder object with decode(syndrome_bits) method.

    Returns:
        Tuple of (probability distribution, success probability, metadata).
    """
    check_qiskit()

    m, n = B.shape
    error_dim = 2 ** m
    syndrome_dim = 2 ** n

    # Build and run Qiskit circuit
    qc = build_dqi_circuit_for_postselection(B=B, v=v, alpha=alpha, ell=ell)
    sv = Statevector(qc)

    # Extract full statevector (Qiskit direct indexing)
    # Qiskit ordering: index = y_qk + s_qk * 2^m
    # where y_qk uses direct indexing (error_qr[i] = bit i)
    full_state_qk = np.asarray(sv.data, dtype=np.complex128)

    # Convert to NumPy MSB-first ordering
    # NumPy ordering: index = y_np * 2^n + s_np
    # where y_np uses MSB-first (position i from left = bit m-1-i)
    full_state_np = np.zeros_like(full_state_qk)
    for idx_qk in range(len(full_state_qk)):
        if np.abs(full_state_qk[idx_qk]) < 1e-15:
            continue
        # Extract Qiskit components
        y_qk = idx_qk % error_dim
        s_qk = idx_qk // error_dim
        # Convert to NumPy MSB-first by reversing bits
        y_np = _reverse_bits(y_qk, m)
        s_np = _reverse_bits(s_qk, n)
        # Compute NumPy combined index
        idx_np = y_np * syndrome_dim + s_np
        full_state_np[idx_np] = full_state_qk[idx_qk]

    # Reshape into (error, syndrome) matrix (MSB-first ordering)
    state_matrix = full_state_np.reshape(error_dim, syndrome_dim)

    # Decode postselection: for each syndrome, find decoded error index
    # Decoder expects MSB-first syndrome bits
    decoded_y = np.full(syndrome_dim, -1, dtype=np.int64)
    for s in range(syndrome_dim):
        s_bits = np.array([(s >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int8)
        e_recovered = decoder.decode(s_bits)
        if e_recovered is not None:
            decoded_y[s] = sum(int(e_recovered[j]) << (m - 1 - j) for j in range(m))

    # Project onto successfully decoded states
    syndrome_state = np.zeros(syndrome_dim, dtype=np.complex128)
    valid_s = np.where(decoded_y >= 0)[0]
    if valid_s.size > 0:
        selected = state_matrix[decoded_y[valid_s], valid_s]
        selected = np.where(np.abs(selected) < 1e-15, 0.0 + 0.0j, selected)
        syndrome_state[valid_s] = selected

    # Compute success probability before normalization
    success_prob = float(np.sum(np.abs(syndrome_state) ** 2))

    # Normalize
    norm = np.linalg.norm(syndrome_state)
    if norm > 1e-15:
        syndrome_state /= norm

    # Apply Hadamard transform (fast Walsh-Hadamard)
    transformed = syndrome_state.copy()
    h = 1
    while h < syndrome_dim:
        for i in range(0, syndrome_dim, h * 2):
            for j in range(i, i + h):
                x = transformed[j]
                y = transformed[j + h]
                transformed[j] = x + y
                transformed[j + h] = x - y
        h *= 2
    transformed /= np.sqrt(syndrome_dim)

    # Convert to probability distribution
    prob_dist = np.abs(transformed) ** 2
    total = float(np.sum(prob_dist))
    if total > 1e-15:
        prob_dist /= total

    metadata = {
        "m": m,
        "n": n,
        "ell": ell,
        "error_dim": error_dim,
        "syndrome_dim": syndrome_dim,
        "num_decodable_syndromes": int(np.sum(decoded_y >= 0)),
        "success_prob": success_prob,
    }

    return prob_dist, success_prob, metadata


def qiskit_decode_inclusive_validation_report(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
    decoder: object,
) -> dict[str, Any]:
    """Return a decode-inclusive postselected validation payload.

    This provides full end-to-end parity validation using postselection
    to simulate decode/uncompute without implementing a reversible decoder circuit.

    Args:
        B: (m, n) parity matrix.
        v: (m,) target parity vector.
        alpha: Dicke weight coefficients.
        ell: Maximum error weight.
        decoder: Decoder object with decode(syndrome_bits) method.

    Returns:
        Validation report dict with status, distribution, and metadata.
    """
    scope = decode_inclusive_scope_metadata()

    if not QISKIT_AVAILABLE:
        return {
            "status": "not_available",
            "reason": "qiskit_not_installed",
            "scope": scope,
            "prob_dist": None,
            "success_prob": None,
            "metadata": None,
        }

    try:
        prob_dist, success_prob, metadata = qiskit_decode_inclusive_distribution(
            B=B, v=v, alpha=alpha, ell=ell, decoder=decoder
        )
        return {
            "status": "ok",
            "reason": None,
            "scope": scope,
            "prob_dist": prob_dist,
            "success_prob": success_prob,
            "metadata": metadata,
        }
    except Exception as exc:
        return {
            "status": "runtime_error",
            "reason": str(exc),
            "scope": scope,
            "prob_dist": None,
            "success_prob": None,
            "metadata": None,
        }
