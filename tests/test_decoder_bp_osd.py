"""Tests for deterministic BP+OSD-lite decoder."""

from __future__ import annotations

import numpy as np
import sys

sys.path.insert(0, ".")

from src.decoder_bp_osd import BPOSDLiteDecoder


def test_bp_osd_result_contract_keys() -> None:
    B = np.eye(4, dtype=np.int8)
    decoder = BPOSDLiteDecoder(B, ell=2, bp_max_iter=8, osd_order_cap=2, osd_candidate_budget=12)
    syndrome = np.array([1, 0, 0, 1], dtype=np.int8)
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
    assert result["decoder_name"] == "bp_osd_lite"
    meta = result["metadata"]
    for key in ["osd_order_cap", "osd_candidate_budget", "tie_break", "fallback_flag", "iterations_used"]:
        assert key in meta


def test_bp_osd_deterministic_for_same_input() -> None:
    B = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    syndrome = np.array([1, 0, 1, 1], dtype=np.int8)
    decoder = BPOSDLiteDecoder(B, ell=2, bp_max_iter=10, osd_order_cap=2, osd_candidate_budget=16)

    r1 = decoder.decode_result(syndrome)
    r2 = decoder.decode_result(syndrome)
    assert r1["success"] == r2["success"]
    if r1["decoded_error"] is None:
        assert r2["decoded_error"] is None
    else:
        np.testing.assert_array_equal(r1["decoded_error"], r2["decoded_error"])


def test_bp_osd_never_crashes_on_small_cases() -> None:
    B = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    decoder = BPOSDLiteDecoder(B, ell=2, bp_max_iter=6, osd_order_cap=2, osd_candidate_budget=10)
    for syndrome_int in range(2 ** B.shape[1]):
        syndrome = np.array(
            [(syndrome_int >> (B.shape[1] - 1 - j)) & 1 for j in range(B.shape[1])],
            dtype=np.int8,
        )
        result = decoder.decode_result(syndrome)
        assert isinstance(result, dict)
        assert "success" in result
