"""
test_decoder.py — Tests for bounded-distance brute-force decoder.

Run: python -m pytest tests/test_decoder.py -v
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from src.decoder_bruteforce import (
    BoundedDistanceDecoder,
    compute_syndrome,
    hamming_weight,
)


def test_compute_syndrome():
    """Syndrome computation: s = B^T @ e mod 2."""
    # Simple 3x2 matrix
    B = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.int8)
    e = np.array([1, 0, 1], dtype=np.int8)  # weight 2

    s = compute_syndrome(B, e)
    # B^T @ e = [1*1 + 0*0 + 1*1, 1*1 + 1*0 + 0*1] = [0, 1] mod 2
    expected = np.array([0, 1], dtype=np.int8)
    np.testing.assert_array_equal(s, expected)


def test_hamming_weight():
    assert hamming_weight(np.array([0, 0, 0])) == 0
    assert hamming_weight(np.array([1, 0, 1])) == 2
    assert hamming_weight(np.array([1, 1, 1, 1])) == 4


def test_decoder_exact_recovery():
    """Decoder should recover e exactly when |e| ≤ ell."""
    # Use a matrix with good syndrome distinguishing properties
    # 4x4 matrix (m=4 constraints, n=4 bits) - like identity for unique syndromes
    B = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.int8)

    decoder = BoundedDistanceDecoder(B, ell=2)

    # Test several error patterns of weight ≤ 2
    # With identity-like B, each single-bit error has unique syndrome
    for e_int in [0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0101]:
        e = np.array([(e_int >> (3 - j)) & 1 for j in range(4)], dtype=np.int8)
        if hamming_weight(e) <= 2:
            s = compute_syndrome(B, e)
            e_recovered = decoder.decode(s)
            np.testing.assert_array_equal(e_recovered, e,
                f"Failed for e={e}, s={s}, got {e_recovered}")


def test_decoder_all_weight_up_to_ell():
    """Decoder should handle all patterns up to weight ell."""
    B = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.int8)

    decoder = BoundedDistanceDecoder(B, ell=1)

    # All weight-0 and weight-1 patterns should decode correctly
    for i in range(2 ** 4):
        e = np.array([(i >> (3 - j)) & 1 for j in range(4)], dtype=np.int8)
        if hamming_weight(e) <= 1:
            s = compute_syndrome(B, e)
            e_recovered = decoder.decode(s)
            np.testing.assert_array_equal(e_recovered, e)


def test_decoder_lookup_table_size():
    """Lookup table should have entries for all weight ≤ ell patterns."""
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)  # m=3, n=2
    decoder = BoundedDistanceDecoder(B, ell=2)

    # Number of patterns: C(3,0) + C(3,1) + C(3,2) = 1 + 3 + 3 = 7
    # But lookup is keyed by syndrome, so may have fewer entries if collisions
    assert len(decoder.lookup) <= 7


def test_decoder_failure_on_ambiguous():
    """Decoder returns None if syndrome doesn't match unique pattern."""
    # Design a matrix where some syndromes are ambiguous for weight > ell
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)  # Identity-ish
    decoder = BoundedDistanceDecoder(B, ell=0)

    # Syndrome [1, 0] requires weight-1 error, but ell=0, so should fail
    s = np.array([1, 0], dtype=np.int8)
    result = decoder.decode(s)
    assert result is None or hamming_weight(result) > 0  # Either None or wrong


def test_bruteforce_decoder_result_contract():
    """Structured decode_result payload should follow canonical schema."""
    B = np.eye(4, dtype=np.int8)
    decoder = BoundedDistanceDecoder(B, ell=2)

    syndrome = np.array([1, 0, 0, 0], dtype=np.int8)
    result = decoder.decode_result(syndrome)

    required = {
        "decoder_name",
        "decoded_error",
        "success",
        "within_radius",
        "distance",
        "num_flips",
        "metadata",
    }
    assert required.issubset(result.keys())
    assert result["decoder_name"] == "bruteforce"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
