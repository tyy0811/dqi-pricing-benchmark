"""
test_dqi_state.py — Tests for DQI five-step state preparation.

Run: python -m pytest tests/test_dqi_state.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

from src.dqi_state import (
    prepare_dicke_state,
    dicke_state_dimension,
    apply_phase_kick,
    compute_syndrome_state,
    decode_and_uncompute,
    run_dqi_statevector,
)
from src.decoder_bruteforce import BoundedDistanceDecoder


def _decode_and_uncompute_reference(
    full_state: np.ndarray,
    B: np.ndarray,
    decoder: BoundedDistanceDecoder,
) -> tuple[np.ndarray, float]:
    """Reference implementation with the original nested-loop semantics."""
    m, n = B.shape
    error_dim = 2 ** m
    syndrome_dim = 2 ** n

    decoded_y = {}
    for s in range(syndrome_dim):
        s_bits = np.array([(s >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int8)
        e_recovered = decoder.decode(s_bits)
        if e_recovered is not None:
            decoded_y[s] = sum(int(e_recovered[j]) << (m - 1 - j) for j in range(m))
        else:
            decoded_y[s] = None

    syndrome_state = np.zeros(syndrome_dim, dtype=np.complex128)
    for y in range(error_dim):
        for s in range(syndrome_dim):
            full_idx = y * syndrome_dim + s
            amp = full_state[full_idx]
            if np.abs(amp) < 1e-15:
                continue
            if decoded_y[s] is not None and decoded_y[s] == y:
                syndrome_state[s] += amp

    success_prob = float(np.sum(np.abs(syndrome_state) ** 2))
    norm = np.linalg.norm(syndrome_state)
    if norm > 0:
        syndrome_state /= norm
    return syndrome_state, success_prob


def test_dicke_state_dimension():
    """Check Dicke state dimension formula."""
    from math import comb
    for m in [3, 5, 8]:
        for t in range(m + 1):
            expected = comb(m, t)
            assert dicke_state_dimension(m, t) == expected


def test_dicke_state_normalized():
    """Dicke state should be normalized."""
    for m in [3, 4, 5]:
        for t in range(m + 1):
            state = prepare_dicke_state(m, t)
            norm = np.sum(np.abs(state) ** 2)
            np.testing.assert_allclose(norm, 1.0, rtol=1e-10)


def test_dicke_state_support():
    """Dicke state should only have support on weight-t strings."""
    m = 5
    t = 2
    state = prepare_dicke_state(m, t)

    for x in range(2 ** m):
        weight = bin(x).count('1')
        if weight == t:
            # Should have amplitude 1/sqrt(C(m,t))
            from math import comb
            expected_amp = 1 / np.sqrt(comb(m, t))
            np.testing.assert_allclose(np.abs(state[x]), expected_amp, rtol=1e-10)
        else:
            # Should be zero
            np.testing.assert_allclose(state[x], 0, atol=1e-15)


def test_phase_kick():
    """Phase kick should apply (-1)^(v·y) to each basis state."""
    m = 3
    v = np.array([1, 0, 1], dtype=np.int8)  # v = 101

    # Start with uniform superposition
    state = np.ones(2 ** m, dtype=np.complex128) / np.sqrt(2 ** m)

    kicked = apply_phase_kick(state, v)

    # Check phases
    for y in range(2 ** m):
        y_bits = [(y >> (m - 1 - j)) & 1 for j in range(m)]
        dot = sum(v[j] * y_bits[j] for j in range(m)) % 2
        expected_phase = (-1) ** dot
        np.testing.assert_allclose(
            kicked[y] / state[y], expected_phase,
            rtol=1e-10
        )


def test_syndrome_computation():
    """Syndrome state should encode B^T @ y in ancilla register."""
    # Simple test: identity-like B
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)  # m=3, n=2
    m, n = B.shape

    # Start with |y=101> (y=5)
    y_state = np.zeros(2 ** m, dtype=np.complex128)
    y_state[5] = 1.0  # |101>

    # Compute syndrome
    full_state = compute_syndrome_state(y_state, B)

    # Expected: |101>|s> where s = B^T @ [1,0,1] mod 2
    # s = [1*1 + 0*0 + 1*1, 0*1 + 1*0 + 1*1] = [0, 1]
    expected_syndrome = 1  # binary 01

    # Full state dimension is 2^m * 2^n = 8 * 4 = 32
    # Index is y * 2^n + s
    expected_idx = 5 * (2 ** n) + expected_syndrome
    assert full_state.shape[0] == 2 ** (m + n)
    np.testing.assert_allclose(np.abs(full_state[expected_idx]), 1.0, rtol=1e-10)


def test_run_dqi_statevector_convex_mixture_over_t():
    """Main execution path should be a |alpha_t|^2 mixture over fixed-t branches."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)  # m=2, n=2
    v = np.array([0, 0], dtype=np.int8)
    ell = 1

    alpha = np.array([np.sqrt(0.3), np.sqrt(0.7)], dtype=np.float64)
    _, probs_mix, success_mix = run_dqi_statevector(
        B=B,
        v=v,
        alpha=alpha,
        ell=ell,
        n_samples=32,
        seed=123,
    )

    probs_by_t = []
    success_by_t = []
    for t in range(ell + 1):
        alpha_t = np.zeros(ell + 1, dtype=np.float64)
        alpha_t[t] = 1.0
        _, probs_t, success_t = run_dqi_statevector(
            B=B,
            v=v,
            alpha=alpha_t,
            ell=ell,
            n_samples=32,
            seed=123,
        )
        probs_by_t.append(probs_t)
        success_by_t.append(success_t)

    weights = np.abs(alpha) ** 2
    weights /= np.sum(weights)
    expected_probs = weights[0] * probs_by_t[0] + weights[1] * probs_by_t[1]
    expected_success = weights[0] * success_by_t[0] + weights[1] * success_by_t[1]

    np.testing.assert_allclose(np.sum(probs_mix), 1.0, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(probs_mix, expected_probs, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(success_mix, expected_success, rtol=1e-10, atol=1e-12)


def test_run_dqi_statevector_raises_on_zero_postselection_mass():
    """Sampling should fail explicitly when decode/postselection leaves zero mass."""
    # m=2, n=1. Weight-1 errors both map to syndrome 1, causing ambiguity.
    B = np.array([[1], [1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    ell = 1
    alpha = np.array([0.0, 1.0], dtype=np.float64)  # t=1 branch only

    with pytest.raises(RuntimeError, match="zero surviving amplitude|zero postselection mass"):
        run_dqi_statevector(
            B=B,
            v=v,
            alpha=alpha,
            ell=ell,
            n_samples=8,
            seed=7,
        )


def test_decode_and_uncompute_matches_reference_semantics():
    """Vectorized decode/uncompute should match original nested-loop semantics."""
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)  # m=3, n=2
    decoder = BoundedDistanceDecoder(B, ell=1)

    rng = np.random.default_rng(123)
    full_state = (
        rng.normal(size=2 ** (B.shape[0] + B.shape[1]))
        + 1j * rng.normal(size=2 ** (B.shape[0] + B.shape[1]))
    ).astype(np.complex128)

    syndrome_new, success_new = decode_and_uncompute(full_state, B, decoder)
    syndrome_ref, success_ref = _decode_and_uncompute_reference(full_state, B, decoder)

    np.testing.assert_allclose(syndrome_new, syndrome_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(success_new, success_ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
