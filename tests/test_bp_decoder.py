"""Tests for deterministic BP1-style decoder."""

from __future__ import annotations

import numpy as np

from src.decoder_bp1 import BP1Decoder
from src.dqi_state import run_dqi_statevector
from src.weights import uniform_weights


def test_bp1_decoder_is_deterministic():
    B = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )
    syndrome = np.array([1, 0, 1], dtype=np.int8)
    decoder = BP1Decoder(B, ell=2, max_iter=8)

    r1 = decoder.decode_result(syndrome)
    r2 = decoder.decode_result(syndrome)

    assert r1["success"] == r2["success"]
    if r1["decoded_error"] is None:
        assert r2["decoded_error"] is None
    else:
        np.testing.assert_array_equal(r1["decoded_error"], r2["decoded_error"])


def test_bp1_result_contract():
    B = np.eye(3, dtype=np.int8)
    decoder = BP1Decoder(B, ell=1, max_iter=5)
    result = decoder.decode_result(np.array([1, 0, 0], dtype=np.int8))

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
    assert result["decoder_name"] == "bp1"


def test_bp1_can_run_dqi_statevector_path():
    """BP1 should satisfy run_dqi_statevector decoder interface."""
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    v = np.array([0, 1, 0], dtype=np.int8)
    alpha = uniform_weights(ell=1)
    samples, probs, success_prob = run_dqi_statevector(
        B=B,
        v=v,
        alpha=alpha,
        ell=1,
        n_samples=16,
        seed=5,
        decoder_cls=BP1Decoder,
    )
    assert len(samples) == 16
    assert probs.shape == (2 ** B.shape[1],)
    assert 0.0 <= success_prob <= 1.0
