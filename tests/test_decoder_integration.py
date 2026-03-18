"""Integration tests for decoder-mode wiring in DQI pipeline and benchmark."""

from __future__ import annotations

import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

from src.benchmark import run_benchmark
from src.dqi_pipeline import DQIPipeline
from src.dqi_state import run_dqi_statevector
from src.problem_generator import small_instance
from src.weights import uniform_weights


def test_statevector_decoder_mode_selector_supports_bruteforce_and_bp1():
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    v = np.array([0, 1, 0], dtype=np.int8)
    alpha = uniform_weights(ell=1)

    for decoder_mode in ["bruteforce", "bp1"]:
        samples, probs, success_prob = run_dqi_statevector(
            B=B,
            v=v,
            alpha=alpha,
            ell=1,
            n_samples=24,
            seed=11,
            decoder_mode=decoder_mode,
            execution_mode="mixture",
        )
        assert len(samples) == 24
        assert probs.shape == (2 ** B.shape[1],)
        np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10, atol=1e-12)
        assert 0.0 <= success_prob <= 1.0

    with pytest.raises(ValueError, match="Invalid decoder_mode"):
        run_dqi_statevector(
            B=B,
            v=v,
            alpha=alpha,
            ell=1,
            n_samples=8,
            seed=11,
            decoder_mode="not-a-decoder",
            execution_mode="mixture",
        )


def test_statevector_decoder_mode_selector_supports_bp_osd_lite_and_oracle():
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    v = np.array([0, 1, 0], dtype=np.int8)
    alpha = uniform_weights(ell=1)

    for decoder_mode in ["bp_osd_lite", "oracle"]:
        samples, probs, success_prob = run_dqi_statevector(
            B=B,
            v=v,
            alpha=alpha,
            ell=1,
            n_samples=20,
            seed=12,
            decoder_mode=decoder_mode,
            execution_mode="mixture",
        )
        assert len(samples) == 20
        assert probs.shape == (2 ** B.shape[1],)
        np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10, atol=1e-12)
        assert 0.0 <= success_prob <= 1.0


def test_pipeline_accepts_bp_osd_lite_mode():
    prob = small_instance(3)
    pipeline = DQIPipeline(
        prob,
        k=8,
        ell=2,
        alpha_mode="paper",
        execution_mode="mixture",
        decoder_mode="bp_osd_lite",
    )
    results = pipeline.run(n_samples=28, seed=44)
    assert results["decoder_mode"] == "bp_osd_lite"
    assert 0.0 <= results["decoder_success_rate"] <= 1.0


def test_pipeline_persists_decoder_metadata():
    prob = small_instance(3)

    for decoder_mode in ["bruteforce", "bp1"]:
        pipeline = DQIPipeline(
            prob,
            k=8,
            ell=2,
            alpha_mode="paper",
            execution_mode="mixture",
            decoder_mode=decoder_mode,
        )
        results = pipeline.run(n_samples=40, seed=21)

        assert results["decoder_mode"] == decoder_mode
        assert 0.0 <= results["decoder_success_rate"] <= 1.0

        exact = results["decode_exact_recovery_rate"]
        assert exact is None or 0.0 <= exact <= 1.0

        if decoder_mode == "bruteforce":
            assert results["ambiguous_syndrome_rate"] is not None
            assert 0.0 <= results["ambiguous_syndrome_rate"] <= 1.0
            assert results["flip_count_stats"] is None
        else:
            assert results["ambiguous_syndrome_rate"] is None
            flip_stats = results["flip_count_stats"]
            assert flip_stats is not None
            for key in ["mean", "median", "min", "max"]:
                assert key in flip_stats


def test_benchmark_reports_decoder_sensitivity_for_paper_mixture():
    prob = small_instance(3)
    results = run_benchmark(
        prob,
        k=8,
        ell=2,
        n_dqi_samples=36,
        sa_iter=200,
        top_k_by_surrogate=6,
        seed=17,
    )

    assert "dqi_paper_mixture_bruteforce" in results
    assert "dqi_paper_mixture_bp1" in results
    assert "decoder_sensitivity_paper_mixture" in results

    paper_bf = results["dqi_paper_mixture_bruteforce"]
    paper_bp1 = results["dqi_paper_mixture_bp1"]
    assert paper_bf["decoder_mode"] == "bruteforce"
    assert paper_bp1["decoder_mode"] == "bp1"
    assert results["dqi_paper_mixture"]["decoder_mode"] == "bruteforce"

    summary = results["decoder_sensitivity_paper_mixture"]
    assert summary["alpha_mode"] == "paper"
    assert summary["execution_mode"] == "mixture"
    for decoder_mode in ["bruteforce", "bp1"]:
        entry = summary["by_decoder"][decoder_mode]
        for key in ["best_F", "success_prob", "decode_success_rate"]:
            assert key in entry
