"""Tests for coherent-vs-mixture DQI execution paths."""

from __future__ import annotations

import numpy as np
import pytest
import sys

sys.path.insert(0, ".")

import src.dqi_state as dqi_state
from src.dqi_pipeline import DQIPipeline, compare_candidate_distributions_tvd
from src.dqi_state import (
    coherent_supported_for_instance,
    joint_num_bits_coherent,
    joint_state_size_coherent,
    run_dqi_statevector,
    run_dqi_statevector_coherent,
    weight_register_bits,
)
from src.problem_generator import small_instance
from src.weights import paper_alpha_weights


def test_coherent_support_helpers_report_joint_size() -> None:
    """Coherent helper functions should report joint state dimensions consistently."""
    bits = joint_num_bits_coherent(m=3, n=2, ell=2)
    size = joint_state_size_coherent(m=3, n=2, ell=2)

    assert weight_register_bits(2) == 2
    assert bits == 7
    assert size == 2 ** bits
    assert coherent_supported_for_instance(3, 2, 2, cap_limit=size)


def test_execution_mode_selector_uses_coherent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_dqi_statevector should dispatch to coherent path when requested."""
    B = np.array([[1]], dtype=np.int8)
    v = np.array([0], dtype=np.int8)
    alpha = np.array([1.0], dtype=np.float64)
    called = {"coherent": False}

    def _fake_coherent(*args, **kwargs):
        called["coherent"] = True
        probs = np.array([1.0, 0.0], dtype=np.float64)
        samples = np.zeros(kwargs.get("n_samples", 4), dtype=np.int64)
        return samples, probs, 1.0

    monkeypatch.setattr(dqi_state, "run_dqi_statevector_coherent", _fake_coherent)

    samples, probs, success_prob = run_dqi_statevector(
        B=B,
        v=v,
        alpha=alpha,
        ell=0,
        n_samples=4,
        seed=1,
        execution_mode="coherent",
    )

    assert called["coherent"] is True
    assert len(samples) == 4
    np.testing.assert_allclose(probs, np.array([1.0, 0.0]))
    assert success_prob == 1.0


def test_coherent_mode_tiny_instance_distribution_is_normalized() -> None:
    """Coherent tiny-instance path should return a valid normalized candidate distribution."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)  # m=2, n=2
    v = np.array([0, 1], dtype=np.int8)
    alpha = paper_alpha_weights(ell=1)

    _, probs, success_prob = run_dqi_statevector_coherent(
        B=B,
        v=v,
        alpha=alpha,
        ell=1,
        n_samples=32,
        seed=9,
    )

    assert probs.shape == (2 ** B.shape[1],)
    np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10, atol=1e-12)
    assert 0.0 <= success_prob <= 1.0


def test_coherent_joint_state_is_phase_sensitive_for_alpha() -> None:
    """Coherent path should reflect relative alpha phase in joint final state."""
    B = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int8)  # m=3, n=2
    v = np.array([1, 0, 1], dtype=np.int8)
    ell = 1
    norm = np.sqrt(2.0)
    alpha_plus = np.array([1.0 / norm, 1.0 / norm], dtype=np.complex128)
    alpha_minus = np.array([1.0 / norm, -1.0 / norm], dtype=np.complex128)

    _, probs_plus, _, joint_plus = run_dqi_statevector_coherent(
        B=B,
        v=v,
        alpha=alpha_plus,
        ell=ell,
        n_samples=16,
        seed=3,
        return_joint_state=True,
    )
    _, probs_minus, _, joint_minus = run_dqi_statevector_coherent(
        B=B,
        v=v,
        alpha=alpha_minus,
        ell=ell,
        n_samples=16,
        seed=3,
        return_joint_state=True,
    )

    assert not np.allclose(joint_plus, joint_minus, atol=1e-12, rtol=1e-10)
    # Candidate distributions may coincide for some instances; the coherent
    # state check ensures the path is amplitude/phase-aware (not pure mixture).
    assert probs_plus.shape == probs_minus.shape


def test_run_dqi_statevector_coherent_raises_when_cap_exceeded() -> None:
    """Oversized coherent runs should raise a clear bounded-size error."""
    B = np.ones((9, 9), dtype=np.int8)
    v = np.zeros(9, dtype=np.int8)
    alpha = np.ones(4, dtype=np.float64)
    alpha /= np.linalg.norm(alpha)

    with pytest.raises(ValueError, match="coherent_hilbert_cap_exceeded"):
        run_dqi_statevector_coherent(
            B=B,
            v=v,
            alpha=alpha,
            ell=3,
            n_samples=8,
            seed=5,
        )


def test_compare_candidate_distributions_tvd_identical_is_zero() -> None:
    """TVD helper should be zero for identical probability vectors."""
    run_a = {"probabilities": np.array([0.2, 0.8], dtype=np.float64)}
    run_b = {"probabilities": np.array([0.2, 0.8], dtype=np.float64)}

    out = compare_candidate_distributions_tvd(run_a, run_b)
    assert out["ok"] is True
    assert out["tvd"] == 0.0


def test_compare_candidate_distributions_tvd_bounds() -> None:
    """TVD helper should return values in [0, 1] for normalized inputs."""
    run_a = {"probabilities": np.array([1.0, 0.0], dtype=np.float64)}
    run_b = {"probabilities": np.array([0.0, 1.0], dtype=np.float64)}

    out = compare_candidate_distributions_tvd(run_a, run_b)
    assert out["ok"] is True
    assert 0.0 <= out["tvd"] <= 1.0
    np.testing.assert_allclose(out["tvd"], 1.0, rtol=1e-12, atol=1e-12)


def test_pipeline_persists_execution_mode_and_coherent_flags() -> None:
    """Pipeline run should persist execution/coherent metadata in results."""
    prob = small_instance(3)
    pipeline = DQIPipeline(prob, k=8, ell=2, alpha_mode="paper", execution_mode="mixture")
    results = pipeline.run(n_samples=24, seed=11)

    assert results["execution_mode"] == "mixture"
    assert "coherent_supported" in results
    assert "coherent_exact_small_instance" in results
    assert "weight_register_bits" in results
    assert "decode_semantics" in results
    assert "state_model" in results
    assert "postselection_kind" in results
