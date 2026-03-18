"""
test_reduction.py — Tests for Walsh-Hadamard transform and (B,v) construction.

Run: python -m pytest tests/test_reduction.py -v
"""

import json
import numpy as np
import sys
sys.path.insert(0, ".")

from src.reduction import (
    walsh_hadamard_transform,
    top_k_coefficients,
    coefficients_to_max_xorsat,
    surrogate_objective,
    surrogate_faithfulness,
    surrogate_structure_metrics,
    build_surrogate,
    build_surrogate_artifact,
    surrogate_artifact_to_dict,
    surrogate_artifact_from_dict,
)
from src.problem_generator import default_instance


def test_wht_parseval():
    """Walsh-Hadamard transform preserves energy (Parseval)."""
    prob = default_instance()
    f = prob.build_truth_table()
    f_hat = walsh_hadamard_transform(f)

    # Parseval: sum(f^2) / 2^n = sum(f_hat^2)
    n = prob.n_bits
    energy_time = np.sum(f ** 2) / (2 ** n)
    energy_freq = np.sum(f_hat ** 2)
    np.testing.assert_allclose(energy_time, energy_freq, rtol=1e-10)


def test_wht_inverse():
    """WHT is its own inverse (up to scaling)."""
    prob = default_instance()
    f = prob.build_truth_table()
    f_hat = walsh_hadamard_transform(f)
    f_recovered = walsh_hadamard_transform(f_hat) * (2 ** prob.n_bits)
    np.testing.assert_allclose(f, f_recovered, rtol=1e-10)


def test_top_k_excludes_constant():
    """Top-K should exclude the constant term (index 0)."""
    prob = default_instance()
    f = prob.build_truth_table()
    f_hat = walsh_hadamard_transform(f)
    indices, coeffs = top_k_coefficients(f_hat, k=15)

    assert 0 not in indices, "Constant term should be excluded"
    assert len(indices) == 15


def test_top_k_sorted_by_magnitude():
    """Top-K coefficients should be sorted by |f_hat|."""
    prob = default_instance()
    f = prob.build_truth_table()
    f_hat = walsh_hadamard_transform(f)
    indices, coeffs = top_k_coefficients(f_hat, k=10)

    magnitudes = np.abs(coeffs)
    assert np.all(magnitudes[:-1] >= magnitudes[1:]), "Should be sorted descending"


def test_max_xorsat_construction():
    """B matrix and v vector have correct shapes and values."""
    prob = default_instance()
    f = prob.build_truth_table()
    f_hat = walsh_hadamard_transform(f)
    indices, coeffs = top_k_coefficients(f_hat, k=15)
    B, v, weights = coefficients_to_max_xorsat(indices, coeffs, n_bits=prob.n_bits)

    assert B.shape == (15, 10), f"B shape should be (m, n) = (15, 10), got {B.shape}"
    assert v.shape == (15,), f"v shape should be (m,) = (15,), got {v.shape}"
    assert weights.shape == (15,), f"weights shape should be (15,)"
    assert set(v).issubset({0, 1}), "v should be binary"
    assert np.all(weights > 0), "weights should be positive"


def test_surrogate_objective():
    """Surrogate G(x) matches the truncated Fourier expansion."""
    prob = default_instance()
    f = prob.build_truth_table()
    f_hat = walsh_hadamard_transform(f)
    indices, coeffs = top_k_coefficients(f_hat, k=15)
    B, v, weights = coefficients_to_max_xorsat(indices, coeffs, n_bits=prob.n_bits)

    # Pick a random x
    x = 42
    g_computed = surrogate_objective(x, B, v, weights, prob.n_bits)

    # Manual computation
    g_expected = 0.0
    for i, (idx, coeff) in enumerate(zip(indices, coeffs)):
        bits = [(x >> (prob.n_bits - 1 - j)) & 1 for j in range(prob.n_bits)]
        mask = [(idx >> (prob.n_bits - 1 - j)) & 1 for j in range(prob.n_bits)]
        parity = sum(b * m for b, m in zip(bits, mask)) % 2
        g_expected += coeff * ((-1) ** parity)

    np.testing.assert_allclose(g_computed, g_expected, rtol=1e-10)


def test_surrogate_faithfulness():
    """Spearman correlation should be reasonable for top-15."""
    prob = default_instance()
    rho, eta = surrogate_faithfulness(prob, k=15)

    assert 0 < rho <= 1, f"Spearman correlation should be positive, got {rho}"
    assert 0 < eta <= 1, f"Energy fraction should be in (0, 1], got {eta}"


def test_surrogate_structure_metrics_fields():
    """Surrogate structure metrics should expose decoder/structure diagnostics."""
    prob = default_instance()
    metrics = surrogate_structure_metrics(prob, k=8, ell=2)
    required = {
        "row_weight_mean",
        "row_weight_max",
        "col_weight_mean",
        "col_weight_max",
        "syndrome_density",
        "number_of_unique_syndromes_decodable_at_ell",
        "decoder_collision_rate",
        "estimated_lookup_table_size",
    }
    assert required.issubset(metrics.keys())


def test_build_surrogate_can_attach_structure_metrics():
    """build_surrogate should optionally include structure metrics when ell is provided."""
    prob = default_instance()
    surrogate = build_surrogate(prob, k=8, ell=2)
    assert "structure_metrics" in surrogate
    assert surrogate["structure_metrics"]["estimated_lookup_table_size"] >= 0


def test_surrogate_artifact_contract_fields():
    """Canonical surrogate artifact should expose required contract fields."""
    prob = default_instance()
    artifact = build_surrogate_artifact(prob, k=10)
    required = {
        "artifact_version",
        "mode",
        "bit_order",
        "n",
        "m",
        "k_requested",
        "k_selected",
        "exclude_constant",
        "B",
        "v",
        "term_weights",
        "indices",
        "coefficients",
        "truncation_metadata",
        "diagnostics",
        "source_metadata",
    }
    assert required.issubset(artifact.keys())
    assert artifact["term_weights"].shape == (artifact["m"],)


def test_surrogate_artifact_roundtrip_dict():
    """Artifact dict serialization round-trip should preserve core fields."""
    prob = default_instance()
    artifact = build_surrogate_artifact(prob, k=8)
    payload = surrogate_artifact_to_dict(artifact)
    restored = surrogate_artifact_from_dict(payload)

    np.testing.assert_array_equal(artifact["B"], restored["B"])
    np.testing.assert_array_equal(artifact["v"], restored["v"])
    np.testing.assert_allclose(artifact["term_weights"], restored["term_weights"])
    assert artifact["mode"] == restored["mode"]


def test_surrogate_artifact_json_roundtrip(tmp_path):
    """Export/import through JSON should preserve schema and core shapes."""
    prob = default_instance()
    artifact = build_surrogate_artifact(prob, k=6)
    payload = surrogate_artifact_to_dict(artifact)

    out_path = tmp_path / "artifact.json"
    out_path.write_text(json.dumps(payload, indent=2))
    loaded_payload = json.loads(out_path.read_text())
    restored = surrogate_artifact_from_dict(loaded_payload)

    assert restored["source_metadata"]["instance_id"].startswith("pricing_")
    assert restored["B"].shape[1] == prob.n_bits
    assert restored["term_weights"].shape == (restored["m"],)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
