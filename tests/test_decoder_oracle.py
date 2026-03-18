"""Tests for tiny-only oracle decoder."""

from __future__ import annotations

import numpy as np
import sys

sys.path.insert(0, ".")

from src.decoder_oracle import TinyOracleDecoder


def test_tiny_oracle_exact_on_tiny_case() -> None:
    B = np.eye(3, dtype=np.int8)
    decoder = TinyOracleDecoder(B, ell=2, max_m_for_oracle=8)

    e_true = np.array([1, 0, 1], dtype=np.int8)
    syndrome = (B.T @ e_true) % 2
    result = decoder.decode_result(syndrome.astype(np.int8))

    assert result["decoder_name"] == "oracle"
    assert result["success"] is True
    np.testing.assert_array_equal(result["decoded_error"], e_true)


def test_tiny_oracle_marks_non_applicable_over_cap() -> None:
    B = np.eye(10, dtype=np.int8)
    decoder = TinyOracleDecoder(B, ell=1, max_m_for_oracle=6)
    syndrome = np.zeros(B.shape[1], dtype=np.int8)
    result = decoder.decode_result(syndrome)

    assert result["success"] is False
    assert result["decoded_error"] is None
    meta = result["metadata"]
    assert meta["decoder_status"] == "skipped_tiny_only"
    assert meta["status_reason_code"] == "oracle_tiny_cap_exceeded"
