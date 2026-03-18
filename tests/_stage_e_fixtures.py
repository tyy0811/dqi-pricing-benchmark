"""Small synthetic fixtures for Stage E policy tests."""

from __future__ import annotations


def stage_e_base_row() -> dict:
    return {
        "run_id": "run_1",
        "manifest_slot": "slot-1",
        "matrix": "matrix_c",
        "stage": "decoder_study",
        "family": "S",
        "instance_id": "S1",
        "encoding": "synthetic_structured_bv",
        "decoder": "bp1",
        "alpha_mode": "paper",
        "execution_mode": "mixture",
        "trial_seed": 7,
        "canonical_trial_seed": 7,
        "seed": 7,
        "run_status": "completed",
        "status_reason_code": "completed",
        "status_reason": "completed",
        "has_verified_provenance": True,
        "best_F": 10.0,
        "best_sampled_F": 10.0,
        "best_top_G_sampled_F": 10.5,
        "topk_regret": 0.1,
        "decision_distortion": 0.2,
        "spearman_rho_F_G": 0.8,
        "retained_energy_eta": 0.6,
        "resource_total_qubits_est_with_decode": 5,
        "resource_total_gate_est_with_decode": 120,
        "resource_total_depth_est_with_decode": 35,
        "resource_decode_uncompute_confidence": "implementation_proxy",
    }


def invalid_paired_artifact() -> dict:
    # Missing required "encodings" block.
    return {"instance_id": "S1"}
