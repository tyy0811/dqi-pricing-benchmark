"""Stage E duplicate-key handling tests."""

from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, ".")

from scripts.make_report import _build_quality_cost_tables


def _base_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "run_b",
                "manifest_slot": "slot-2",
                "matrix": "matrix_c",
                "stage": "decoder_study",
                "family": "S",
                "instance_id": "S1",
                "encoding": "synthetic_structured_bv",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "canonical_trial_seed": 7,
                "trial_seed": 7,
                "seed": 7,
                "run_status": "completed",
                "status_reason_code": "completed",
                "status_reason": "completed",
                "observed_run": True,
                "best_sampled_F": 10.0,
                "best_top_G_sampled_F": 11.0,
                "topk_regret": 0.1,
                "decision_distortion": 0.2,
                "spearman_rho_F_G": 0.7,
                "retained_energy_eta": 0.6,
                "resource_total_qubits_est_with_decode": 5,
                "resource_total_gate_est_with_decode": 120,
                "resource_total_depth_est_with_decode": 35,
                "resource_decode_uncompute_confidence": "implementation_proxy",
                "has_verified_provenance": True,
            },
            {
                "run_id": "run_a",
                "manifest_slot": "slot-1",
                "matrix": "matrix_c",
                "stage": "decoder_study",
                "family": "S",
                "instance_id": "S1",
                "encoding": "synthetic_structured_bv",
                "decoder": "bp1",
                "alpha_mode": "paper",
                "execution_mode": "mixture",
                "canonical_trial_seed": 7,
                "trial_seed": 7,
                "seed": 7,
                "run_status": "completed",
                "status_reason_code": "completed",
                "status_reason": "completed",
                "observed_run": True,
                "best_sampled_F": 10.5,
                "best_top_G_sampled_F": 11.0,
                "topk_regret": 0.1,
                "decision_distortion": 0.2,
                "spearman_rho_F_G": 0.8,
                "retained_energy_eta": 0.6,
                "resource_total_qubits_est_with_decode": 5,
                "resource_total_gate_est_with_decode": 120,
                "resource_total_depth_est_with_decode": 35,
                "resource_decode_uncompute_confidence": "implementation_proxy",
                "has_verified_provenance": True,
            },
        ]
    )


def test_duplicate_key_definition_is_locked() -> None:
    master, unavailable = _build_quality_cost_tables(_base_rows(), results_root=None)
    assert len(master) == 1
    assert len(unavailable) == 1
    assert unavailable.iloc[0]["unavailable_class"] == "duplicate_key"


def test_duplicate_winner_tiebreak_order() -> None:
    master, unavailable = _build_quality_cost_tables(_base_rows(), results_root=None)
    assert master.iloc[0]["run_id"] == "run_a"
    assert unavailable.iloc[0]["run_id"] == "run_b"


def test_duplicate_losers_go_to_unavailable_with_duplicate_key_class() -> None:
    master, unavailable = _build_quality_cost_tables(_base_rows(), results_root=None)
    assert set(unavailable["unavailable_class"]) == {"duplicate_key"}
    assert set(unavailable["unavailable_reason_code"]) == {"duplicate_key"}
