"""
dqi_state.py — DQI five-step state preparation.

Implements the paper-faithful DQI algorithm from Jordan et al. (2024):
1. Prepare Dicke superposition
2. Apply phase kick from v
3. Compute syndrome B^T @ y
4. Decode and uncompute
5. Hadamard and measure

This module provides statevector simulation of the DQI state.

Reference: Jordan et al. (2024) arXiv:2408.08292, Section 8.1.2
"""

from __future__ import annotations

import numpy as np
from math import comb
from itertools import combinations
from typing import Optional

from src.decoder_bruteforce import BoundedDistanceDecoder
from src.decoder_bp1 import BP1Decoder
from src.decoder_bp_osd import BPOSDLiteDecoder
from src.decoder_oracle import TinyOracleDecoder

COHERENT_STATE_SIZE_CAP = 4096
DECODER_MODES = {"bruteforce", "bp1", "bp_osd_lite", "oracle"}


def dicke_state_dimension(m: int, t: int) -> int:
    """Number of basis states in Dicke state |D_t^m>."""
    return comb(m, t)


def prepare_dicke_state(m: int, t: int) -> np.ndarray:
    """Prepare the Dicke state |D_t^m> as a statevector.

    |D_t^m> = (1/sqrt(C(m,t))) * sum_{|y|=t} |y>

    Uses itertools.combinations for efficiency: generates only C(m,t)
    basis states instead of iterating over all 2^m.

    Args:
        m: Number of qubits.
        t: Target Hamming weight.

    Returns:
        state: Complex statevector of dimension 2^m.
    """
    dim = 2 ** m
    state = np.zeros(dim, dtype=np.complex128)

    if t > m:
        return state  # No weight-t states possible

    norm = 1.0 / np.sqrt(comb(m, t))

    # Generate only the C(m,t) basis indices with Hamming weight t
    for positions in combinations(range(m), t):
        # Convert bit positions to integer index (MSB-first)
        y = sum(1 << (m - 1 - pos) for pos in positions)
        state[y] = norm

    return state


def prepare_dicke_superposition(
    m: int,
    ell: int,
    alpha: np.ndarray,
) -> np.ndarray:
    """Prepare weighted superposition of Dicke states.

    |Ψ_1> = sum_{t=0}^{ell} α_t |D_t^m>

    Note: This helper omits the explicit weight register and creates one coherent
    superposition over different t sectors. It is kept for compatibility and
    experimentation, but run_dqi_statevector() uses a fixed-t mixture path
    (without cross-sector interference) for paper-faithful |t>|D_t^m> semantics.

    Args:
        m: Number of qubits in error register.
        ell: Maximum weight.
        alpha: (ell+1,) weight coefficients.

    Returns:
        state: Complex statevector of dimension 2^m.
    """
    dim = 2 ** m
    state = np.zeros(dim, dtype=np.complex128)

    for t in range(ell + 1):
        dicke_t = prepare_dicke_state(m, t)
        state += alpha[t] * dicke_t

    # Normalize (alpha should already be normalized, but ensure)
    norm = np.linalg.norm(state)
    if norm > 0:
        state /= norm

    return state


def apply_phase_kick(
    state: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Apply phase kick: multiply each |y> by (-1)^(v·y).

    This implements Step 2 of the DQI algorithm.

    Args:
        state: Input statevector of dimension 2^m.
        v: (m,) binary target parity vector.

    Returns:
        kicked_state: Phase-kicked statevector.
    """
    m = len(v)
    dim = len(state)
    assert dim == 2 ** m

    kicked = state.copy()

    for y in range(dim):
        # Compute v · y mod 2
        dot = 0
        for j in range(m):
            y_j = (y >> (m - 1 - j)) & 1
            dot += v[j] * y_j
        dot %= 2

        # Apply phase
        kicked[y] *= (-1) ** dot

    return kicked


def compute_syndrome_state(
    error_state: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Compute syndrome s = B^T @ y into ancilla register.

    This implements Step 3 of the DQI algorithm.
    Transforms |y>|0> -> |y>|B^T y mod 2>

    Args:
        error_state: Statevector of dimension 2^m (error register).
        B: (m, n) binary parity matrix.

    Returns:
        full_state: Statevector of dimension 2^(m+n) (error + syndrome).
    """
    m, n = B.shape
    error_dim = 2 ** m
    syndrome_dim = 2 ** n
    full_dim = error_dim * syndrome_dim

    assert len(error_state) == error_dim

    full_state = np.zeros(full_dim, dtype=np.complex128)

    for y in range(error_dim):
        if np.abs(error_state[y]) < 1e-15:
            continue

        # Compute syndrome
        y_bits = np.array([(y >> (m - 1 - j)) & 1 for j in range(m)], dtype=np.int8)
        s_bits = (B.T @ y_bits) % 2
        s = sum(int(s_bits[j]) << (n - 1 - j) for j in range(n))

        # Index in full state: y * 2^n + s
        full_idx = y * syndrome_dim + s
        full_state[full_idx] = error_state[y]

    return full_state


def decode_and_uncompute(
    full_state: np.ndarray,
    B: np.ndarray,
    decoder: object,
) -> tuple[np.ndarray, float]:
    """Decode syndrome and uncompute error register.

    This implements Step 4 of the DQI algorithm.
    Transforms |y>|s> -> |0>|s> when decoding succeeds.

    Note: This is a statevector-faithful simplification. In a full circuit
    implementation, decoding would be done coherently with reversible logic.

    Args:
        full_state: Statevector of dimension 2^(m+n).
        B: (m, n) parity matrix.
        decoder: Decoder object exposing ``decode(syndrome_bits)``.

    Returns:
        syndrome_state: Statevector of dimension 2^n (syndrome register only).
        success_prob: Probability of successful decoding (pre-normalization norm^2).
    """
    m, n = B.shape
    error_dim = 2 ** m
    syndrome_dim = 2 ** n

    # Reshape into (error basis, syndrome basis) for direct indexed access
    state_matrix = full_state.reshape(error_dim, syndrome_dim)

    # Precompute decoded error index for each syndrome (-1 if ambiguous/missing)
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
        # Preserve prior behavior that drops tiny amplitudes
        selected = np.where(np.abs(selected) < 1e-15, 0.0 + 0.0j, selected)
        syndrome_state[valid_s] = selected

    # Compute success probability before normalization
    success_prob = float(np.sum(np.abs(syndrome_state) ** 2))

    # Normalize
    norm = np.linalg.norm(syndrome_state)
    if norm > 0:
        syndrome_state /= norm

    return syndrome_state, success_prob


def hadamard_transform_state(state: np.ndarray) -> np.ndarray:
    """Apply Hadamard transform to statevector.

    This implements Step 5 of the DQI algorithm.

    Args:
        state: Input statevector.

    Returns:
        transformed: Hadamard-transformed statevector.
    """
    n = int(np.log2(len(state)))
    dim = len(state)

    # Fast Walsh-Hadamard transform
    transformed = state.copy()
    h = 1
    while h < dim:
        for i in range(0, dim, h * 2):
            for j in range(i, i + h):
                x = transformed[j]
                y = transformed[j + h]
                transformed[j] = x + y
                transformed[j + h] = x - y
        h *= 2

    transformed /= np.sqrt(dim)
    return transformed


def run_dqi_subset_reference(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
) -> dict:
    """Return the NumPy structural-subset reference distribution.

    This helper simulates only approximate low-weight superposition, phase
    kick, syndrome compute, and final Hadamards. Decode/uncompute is excluded,
    so this does not make a full end-to-end parity claim.
    """
    m, n = B.shape
    k = min(ell + 1, m)
    support_size = 2 ** k
    state = np.zeros(2 ** m, dtype=np.complex128)
    for prefix in range(support_size):
        y = prefix << (m - k)
        state[y] = 1.0 / np.sqrt(support_size)

    state = apply_phase_kick(state, v)
    full_state = compute_syndrome_state(state, B)
    state_matrix = full_state.reshape(2 ** m, 2 ** n)
    transformed_rows = np.zeros_like(state_matrix)
    for y in range(2 ** m):
        transformed_rows[y] = hadamard_transform_state(state_matrix[y])
    prob_dist = np.sum(np.abs(transformed_rows) ** 2, axis=0).real
    total = float(np.sum(prob_dist))
    if total > 1e-15:
        prob_dist /= total

    scope = {
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
        "parity_claim": "no_full_end_to_end_parity_claim",
    }

    return {
        "status": "ok",
        "scope": scope["scope"],
        "scope_metadata": scope,
        "prob_dist": prob_dist,
        "distribution_basis": scope["distribution_basis"],
        "decode_uncompute_included": scope["decode_uncompute_included"],
        "postselection_stage_present": scope["postselection_stage_present"],
        "includes": list(scope["includes"]),
        "excludes": list(scope["excludes"]),
        "parity_claim": scope["parity_claim"],
    }


def run_dqi_decode_inclusive_reference(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
    decoder: object,
) -> dict:
    """Return the NumPy decode-inclusive full-pipeline reference distribution.

    This helper simulates the full DQI pipeline with coherent Dicke superposition:
    1. Prepare weighted Dicke superposition (sum_t alpha_t |D_t^m>)
    2. Apply phase kick
    3. Compute syndrome
    4. Decode and uncompute (via projection/postselection)
    5. Apply Hadamard transform

    This is the reference for validating Qiskit's postselected decode-inclusive path.

    Args:
        B: (m, n) parity matrix.
        v: (m,) target parity vector.
        alpha: (ell+1,) weight coefficients.
        ell: Maximum error weight.
        decoder: Decoder object with decode(syndrome_bits) method.

    Returns:
        Dict with status, probability distribution, success probability, and metadata.
    """
    m, n = B.shape

    # Step 1: Prepare weighted Dicke superposition
    state = prepare_dicke_superposition(m, ell, alpha)

    # Step 2: Apply phase kick
    state = apply_phase_kick(state, v)

    # Step 3: Compute syndrome
    full_state = compute_syndrome_state(state, B)

    # Step 4: Decode and uncompute
    syndrome_state, success_prob = decode_and_uncompute(full_state, B, decoder)

    # Step 5: Hadamard transform
    final_state = hadamard_transform_state(syndrome_state)

    # Convert to probability distribution
    prob_dist = np.abs(final_state) ** 2
    total = float(np.sum(prob_dist))
    if total > 1e-15:
        prob_dist /= total

    scope = {
        "scope": "decode_inclusive_reference",
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
        "parity_claim": "full_end_to_end_reference",
    }

    return {
        "status": "ok",
        "scope": scope["scope"],
        "scope_metadata": scope,
        "prob_dist": prob_dist,
        "success_prob": success_prob,
        "distribution_basis": scope["distribution_basis"],
        "decode_uncompute_included": scope["decode_uncompute_included"],
        "postselection_stage_present": scope["postselection_stage_present"],
        "includes": list(scope["includes"]),
        "excludes": list(scope["excludes"]),
        "parity_claim": scope["parity_claim"],
        "metadata": {
            "m": m,
            "n": n,
            "ell": ell,
        },
    }


def sample_from_state(
    state: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample from statevector probability distribution.

    Args:
        state: Complex statevector.
        n_samples: Number of samples to draw.
        seed: Random seed for reproducibility.

    Returns:
        samples: Array of sampled integers.
    """
    probabilities = np.abs(state) ** 2
    return sample_from_probabilities(probabilities, n_samples, seed)


def sample_from_probabilities(
    probabilities: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample from a (possibly unnormalized) probability vector."""
    probs = np.asarray(probabilities, dtype=np.float64).copy()

    # Explicit failure on empty postselected state (no surviving amplitude)
    total = probs.sum()
    if total < 1e-15:
        raise RuntimeError(
            "DQI sampling failed: zero surviving amplitude after decode/uncompute "
            "(zero postselection mass)."
        )

    rng = np.random.default_rng(seed)
    probs /= total
    return rng.choice(len(probs), size=n_samples, p=probs)


def weight_register_bits(ell: int) -> int:
    """Number of qubits needed to represent weight labels 0..ell."""
    if ell < 0:
        raise ValueError(f"ell must be non-negative, got {ell}")
    if ell == 0:
        return 0
    return int(np.ceil(np.log2(ell + 1)))


def joint_num_bits_coherent(m: int, n: int, ell: int) -> int:
    """Total number of bits in coherent |t>|e>|s> state."""
    return weight_register_bits(ell) + int(m) + int(n)


def joint_state_size_coherent(m: int, n: int, ell: int) -> int:
    """Hilbert-space dimension of coherent |t>|e>|s> state."""
    return 2 ** joint_num_bits_coherent(m, n, ell)


def coherent_supported_for_instance(
    m: int,
    n: int,
    ell: int,
    cap_limit: int = COHERENT_STATE_SIZE_CAP,
) -> bool:
    """Check if coherent path is supported under configured Hilbert-size cap."""
    return joint_state_size_coherent(m, n, ell) <= cap_limit


def _resolve_decoder_class(
    *,
    decoder_mode: str,
    decoder_cls: type | None,
) -> tuple[type, str]:
    """Resolve decoder class and canonical mode label."""
    if decoder_cls is not None:
        name = getattr(decoder_cls, "__name__", "").lower()
        if "osd" in name:
            return decoder_cls, "bp_osd_lite"
        if "oracle" in name:
            return decoder_cls, "oracle"
        if "bp1" in name:
            return decoder_cls, "bp1"
        return decoder_cls, "bruteforce"

    if decoder_mode == "bruteforce":
        return BoundedDistanceDecoder, "bruteforce"
    if decoder_mode == "bp1":
        return BP1Decoder, "bp1"
    if decoder_mode == "bp_osd_lite":
        return BPOSDLiteDecoder, "bp_osd_lite"
    if decoder_mode == "oracle":
        return TinyOracleDecoder, "oracle"

    raise ValueError(
        f"Invalid decoder_mode={decoder_mode!r}. Must be one of {sorted(DECODER_MODES)}."
    )


def _build_decoder(
    *,
    B: np.ndarray,
    ell: int,
    decoder_mode: str,
    decoder_cls: type | None,
) -> tuple[object, str]:
    """Construct decoder instance from mode/class selector."""
    resolved_cls, resolved_mode = _resolve_decoder_class(
        decoder_mode=decoder_mode,
        decoder_cls=decoder_cls,
    )
    return resolved_cls(B, ell), resolved_mode


def _decode_result_payload(decoder: object, syndrome: np.ndarray) -> dict:
    """Return canonical decode payload for one syndrome."""
    if hasattr(decoder, "decode_result"):
        result = decoder.decode_result(syndrome)
        if isinstance(result, dict):
            return result

    decoded = decoder.decode(syndrome)  # type: ignore[attr-defined]
    success = decoded is not None
    distance = int(np.sum(decoded)) if success else None
    return {
        "decoder_name": "custom",
        "decoded_error": None if decoded is None else np.asarray(decoded, dtype=np.int8),
        "success": bool(success),
        "within_radius": bool(success),
        "distance": distance,
        "num_flips": int(distance or 0),
        "metadata": {},
    }


def _numeric_stats(values: list[float]) -> dict | None:
    """Compute compact summary stats for numeric values."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def decoder_diagnostics(
    *,
    B: np.ndarray,
    ell: int,
    decoder_mode: str = "bruteforce",
    decoder_cls: type | None = None,
) -> dict:
    """Compute decoder diagnostics used by pipeline/benchmark payloads."""
    m, n = B.shape
    syndrome_dim = 2 ** n
    decoder, resolved_mode = _build_decoder(
        B=B,
        ell=ell,
        decoder_mode=decoder_mode,
        decoder_cls=decoder_cls,
    )

    success_count = 0
    bp1_flip_counts: list[int] = []
    confidence_values: list[float] = []
    for s in range(syndrome_dim):
        s_bits = np.array([(s >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int8)
        result = _decode_result_payload(decoder, s_bits)
        if bool(result.get("success", False)):
            success_count += 1
        if resolved_mode == "bp1":
            bp1_flip_counts.append(int(result.get("num_flips", 0)))
            meta = result.get("metadata", {}) or {}
            if "confidence" in meta and meta["confidence"] is not None:
                confidence_values.append(float(meta["confidence"]))

    # Exact-recovery rate over true low-weight error patterns.
    total_patterns = 0
    exact_recoveries = 0
    for weight in range(ell + 1):
        for positions in combinations(range(m), weight):
            total_patterns += 1
            e = np.zeros(m, dtype=np.int8)
            for pos in positions:
                e[pos] = 1
            s_bits = (B.T @ e) % 2
            result = _decode_result_payload(decoder, s_bits.astype(np.int8))
            decoded = result.get("decoded_error")
            if decoded is not None and np.array_equal(np.asarray(decoded, dtype=np.int8), e):
                exact_recoveries += 1

    decode_exact_recovery_rate = (
        float(exact_recoveries / total_patterns) if total_patterns > 0 else None
    )
    ambiguous_rate = None
    if resolved_mode == "bruteforce" and hasattr(decoder, "lookup"):
        ambiguous = sum(1 for v in decoder.lookup.values() if v is None)
        ambiguous_rate = float(ambiguous / syndrome_dim)

    return {
        "decoder_mode": resolved_mode,
        "decoder_success_rate": float(success_count / syndrome_dim),
        "decode_exact_recovery_rate": decode_exact_recovery_rate,
        "ambiguous_syndrome_rate": ambiguous_rate,
        "flip_count_stats": _numeric_stats([float(v) for v in bp1_flip_counts]) if resolved_mode == "bp1" else None,
        "confidence_stats": _numeric_stats(confidence_values),
    }


def _decoded_error_indices(
    m: int,
    n: int,
    decoder: object,
) -> np.ndarray:
    """Return decoded error index per syndrome, or -1 when unavailable."""
    syndrome_dim = 2 ** n
    decoded = np.full(syndrome_dim, -1, dtype=np.int64)
    for s in range(syndrome_dim):
        s_bits = np.array([(s >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int8)
        e_rec = decoder.decode(s_bits)
        if e_rec is not None:
            decoded[s] = sum(int(e_rec[j]) << (m - 1 - j) for j in range(m))
    return decoded


def run_dqi_statevector_mixture(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    decoder_mode: str = "bruteforce",
    decoder_cls: type | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run statevector DQI via fixed-weight branch mixture."""
    m, n = B.shape
    decoder, _ = _build_decoder(
        B=B,
        ell=ell,
        decoder_mode=decoder_mode,
        decoder_cls=decoder_cls,
    )
    alpha_sq = np.abs(alpha[:ell + 1]) ** 2
    alpha_total = float(np.sum(alpha_sq))

    probabilities = np.zeros(2 ** n, dtype=np.float64)
    success_prob = 0.0

    if alpha_total < 1e-15:
        samples = sample_from_probabilities(probabilities, n_samples, seed)
        return samples, probabilities, success_prob

    for t in range(ell + 1):
        branch_weight = float(alpha_sq[t] / alpha_total)
        if branch_weight < 1e-15:
            continue

        # Step 1: Prepare fixed-weight Dicke state |D_t^m>
        state_t = prepare_dicke_state(m, t)

        # Step 2: Apply phase kick
        state_t = apply_phase_kick(state_t, v)

        # Step 3: Compute syndrome
        full_state_t = compute_syndrome_state(state_t, B)

        # Step 4: Decode and uncompute
        syndrome_state_t, success_t = decode_and_uncompute(full_state_t, B, decoder)

        # Step 5: Hadamard transform
        final_state_t = hadamard_transform_state(syndrome_state_t)

        probabilities += branch_weight * (np.abs(final_state_t) ** 2)
        success_prob += branch_weight * success_t

    total_prob = float(np.sum(probabilities))
    if total_prob > 1e-15:
        probabilities /= total_prob

    samples = sample_from_probabilities(probabilities, n_samples, seed)
    return samples, probabilities, success_prob


def run_dqi_statevector_coherent(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    decoder_mode: str = "bruteforce",
    decoder_cls: type | None = None,
    cap_limit: int = COHERENT_STATE_SIZE_CAP,
    return_joint_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Run explicit coherent tiny-instance state on |t>|e>|s>.

    This path simulates coherent amplitudes in the projective/postselected
    surrogate model used by this repository (non-unitary decode projection).
    """
    m, n = B.shape
    q = weight_register_bits(ell)
    t_dim = 2 ** q
    e_dim = 2 ** m
    s_dim = 2 ** n

    joint_num_bits = joint_num_bits_coherent(m, n, ell)
    joint_state_size = 2 ** joint_num_bits
    if joint_state_size > cap_limit:
        raise ValueError(
            "coherent_hilbert_cap_exceeded: "
            f"joint_state_size={joint_state_size} exceeds cap_limit={cap_limit} "
            f"(joint_num_bits={joint_num_bits}, q={q}, m={m}, n={n}, ell={ell})."
        )

    decoder, _ = _build_decoder(
        B=B,
        ell=ell,
        decoder_mode=decoder_mode,
        decoder_cls=decoder_cls,
    )

    alpha_vec = np.asarray(alpha[:ell + 1], dtype=np.complex128)
    alpha_norm = float(np.linalg.norm(alpha_vec))
    if alpha_norm < 1e-15:
        probabilities = np.zeros(s_dim, dtype=np.float64)
        samples = sample_from_probabilities(probabilities, n_samples, seed)
        if return_joint_state:
            return samples, probabilities, 0.0, np.zeros((t_dim, s_dim), dtype=np.complex128)
        return samples, probabilities, 0.0

    # Step 1: Prepare coherent |t>|e>|s=0> state.
    state_tes = np.zeros((t_dim, e_dim, s_dim), dtype=np.complex128)
    for t in range(ell + 1):
        dicke_t = prepare_dicke_state(m, t)
        state_tes[t, :, 0] = alpha_vec[t] * dicke_t

    # Step 2: Apply phase kick on error register.
    phase = np.ones(e_dim, dtype=np.complex128)
    for e in range(e_dim):
        dot = 0
        for j in range(m):
            e_j = (e >> (m - 1 - j)) & 1
            dot += int(v[j]) * e_j
        phase[e] = (-1) ** (dot % 2)
    state_tes[:, :, 0] *= phase[np.newaxis, :]

    # Step 3: Coherently compute syndrome into s register.
    state_tes_after_syn = np.zeros_like(state_tes)
    syndrome_index = np.zeros(e_dim, dtype=np.int64)
    for e in range(e_dim):
        e_bits = np.array([(e >> (m - 1 - j)) & 1 for j in range(m)], dtype=np.int8)
        s_bits = (B.T @ e_bits) % 2
        syndrome_index[e] = sum(int(s_bits[j]) << (n - 1 - j) for j in range(n))
        s = syndrome_index[e]
        state_tes_after_syn[:, e, s] = state_tes[:, e, 0]

    # Step 4: Projective decode/uncompute surrogate on joint state.
    decoded_e = _decoded_error_indices(m=m, n=n, decoder=decoder)
    state_ts = np.zeros((t_dim, s_dim), dtype=np.complex128)
    valid_s = np.where(decoded_e >= 0)[0]
    if valid_s.size > 0:
        selected = state_tes_after_syn[:, decoded_e[valid_s], valid_s]
        selected = np.where(np.abs(selected) < 1e-15, 0.0 + 0.0j, selected)
        state_ts[:, valid_s] = selected

    success_prob = float(np.sum(np.abs(state_ts) ** 2))
    norm = float(np.linalg.norm(state_ts))
    if norm > 0:
        state_ts /= norm

    # Step 5: Hadamard on syndrome register for each t basis state.
    final_joint_state = np.zeros_like(state_ts)
    for t in range(t_dim):
        final_joint_state[t] = hadamard_transform_state(state_ts[t])

    probabilities = np.sum(np.abs(final_joint_state) ** 2, axis=0).real
    total_prob = float(np.sum(probabilities))
    if total_prob > 1e-15:
        probabilities /= total_prob

    samples = sample_from_probabilities(probabilities, n_samples, seed)
    if return_joint_state:
        return samples, probabilities, success_prob, final_joint_state
    return samples, probabilities, success_prob


def run_dqi_statevector(
    B: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ell: int,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    decoder_mode: str = "bruteforce",
    decoder_cls: type | None = None,
    execution_mode: str = "mixture",
    cap_limit: int = COHERENT_STATE_SIZE_CAP,
    return_joint_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Run full DQI pipeline in statevector simulation.

    Note: This is a statevector-faithful simplification of the DQI algorithm.
    The default execution path is a weighted fixed-branch mixture for speed.
    Coherent mode explicitly simulates |t>|e>|s> under a tiny-instance cap.

    Args:
        B: (m, n) parity matrix.
        v: (m,) target parity vector.
        alpha: (ell+1,) Dicke weight coefficients.
        ell: Maximum Dicke weight.
        n_samples: Number of samples to draw.
        seed: Random seed.
        decoder_cls: Optional decoder class with signature ``decoder_cls(B, ell)``
            and ``decode(syndrome_bits)`` method. Defaults to
            ``BoundedDistanceDecoder``.
        decoder_mode: "bruteforce" (default) or "bp1". Ignored when
            ``decoder_cls`` is provided.
        execution_mode: "mixture" (default) or "coherent".
        cap_limit: Maximum coherent state size (coherent mode only).
        return_joint_state: When True in coherent mode, return the final
            (t, x) joint state after Hadamard for diagnostics.

    Returns:
        samples: Array of sampled x values.
        probabilities: Full probability distribution over x.
        success_prob: Decoding success probability (DQI paper diagnostic).
    """
    if execution_mode == "mixture":
        if return_joint_state:
            raise ValueError("return_joint_state is only supported in execution_mode='coherent'")
        return run_dqi_statevector_mixture(
            B=B,
            v=v,
            alpha=alpha,
            ell=ell,
            n_samples=n_samples,
            seed=seed,
            decoder_mode=decoder_mode,
            decoder_cls=decoder_cls,
        )

    if execution_mode == "coherent":
        return run_dqi_statevector_coherent(
            B=B,
            v=v,
            alpha=alpha,
            ell=ell,
            n_samples=n_samples,
            seed=seed,
            decoder_mode=decoder_mode,
            decoder_cls=decoder_cls,
            cap_limit=cap_limit,
            return_joint_state=return_joint_state,
        )

    raise ValueError(
        f"Invalid execution_mode={execution_mode!r}. Must be one of ['coherent', 'mixture']."
    )
