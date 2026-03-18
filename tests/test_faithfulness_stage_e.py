"""Stage E faithfulness utility tests."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


def test_faithfulness_uses_existing_values_when_present() -> None:
    from src.faithfulness import normalize_faithfulness_fields

    row = {
        "best_sampled_F": 10.0,
        "best_top_G_sampled_F": 7.0,
        "decision_distortion": 0.1,
        "spearman_rho_F_G": 0.95,
        "retained_energy_eta": 0.8,
        "surrogate_best_vs_true_best_sampled_gap": -3.0,
        "has_verified_provenance": False,
    }

    out = normalize_faithfulness_fields(row)

    assert out["decision_distortion"] == 0.1
    assert out["spearman_rho_F_G"] == 0.95
    assert out["retained_energy_eta"] == 0.8
    assert out["best_top_G_sampled_F"] == 7.0
    assert out["surrogate_best_vs_true_best_sampled_gap"] == -3.0


def test_faithfulness_does_not_compute_without_verified_provenance() -> None:
    from src.faithfulness import can_compute_faithfulness, normalize_faithfulness_fields

    row = {
        "best_sampled_F": 11.0,
        "best_top_G_sampled_F": 8.0,
        "has_verified_provenance": False,
    }

    ok, reason = can_compute_faithfulness(row)
    assert ok is False
    assert reason == "insufficient_provenance"

    out = normalize_faithfulness_fields(row)
    assert out["surrogate_best_vs_true_best_sampled_gap"] is None


def test_surrogate_gap_computation_when_inputs_available() -> None:
    from src.faithfulness import compute_surrogate_best_gap, normalize_faithfulness_fields

    row = {
        "best_sampled_F": 12.0,
        "best_top_G_sampled_F": 9.0,
        "has_verified_provenance": True,
    }

    assert compute_surrogate_best_gap(row) == -3.0

    out = normalize_faithfulness_fields(row)
    assert out["surrogate_best_vs_true_best_sampled_gap"] == -3.0
