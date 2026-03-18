"""Stage D row-schema contract tests."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")


def _valid_stage_d_row() -> dict:
    return {
        "run_id": "matrix_b_alpha_validation_C1_wht_exact_oracle_paper_coherent_ell1_m4_seed42",
        "matrix": "matrix_b",
        "stage": "alpha_validation",
        "family": "C",
        "instance_id": "C1",
        "encoding": "wht_exact",
        "decoder": "oracle",
        "alpha_mode": "paper",
        "execution_mode": "coherent",
        "ell": 1,
        "m_active": 4,
        "trial_seed": 42,
        "canonical_trial_seed": 42,
        "seed": 42,
        "run_status": "completed",
        "status_reason_code": "completed",
        "status_reason": "completed",
        "in_experiment_matrix": True,
        "run_error": None,
        "coherent_supported": True,
        "oracle_exact": True,
        "success_probability": 1.0,
        "best_F": 4.0,
        "best_sampled_F": 4.0,
        "best_top_G_sampled_F": 4.0,
        "F_star": 4.0,
        "top1_regret": 0.0,
        "postselection_success": 1.0,
    }


def test_stage_d_row_contract_accepts_valid_row() -> None:
    from src.report_schema import validate_alpha_validation_row

    row = _valid_stage_d_row()
    validate_alpha_validation_row(row)


def test_stage_d_requires_oracle_decoder() -> None:
    from src.report_schema import validate_alpha_validation_row

    row = _valid_stage_d_row()
    row["decoder"] = "bp1"
    with pytest.raises(ValueError, match="decoder"):
        validate_alpha_validation_row(row)


def test_stage_d_requires_encoding_and_canonical_trial_seed() -> None:
    from src.report_schema import validate_alpha_validation_row

    row = _valid_stage_d_row()
    row.pop("encoding")
    with pytest.raises(ValueError, match="encoding"):
        validate_alpha_validation_row(row)

    row = _valid_stage_d_row()
    row.pop("canonical_trial_seed")
    with pytest.raises(ValueError, match="canonical_trial_seed"):
        validate_alpha_validation_row(row)


def test_stage_d_requires_nullable_string_run_error() -> None:
    from src.report_schema import validate_alpha_validation_row

    row = _valid_stage_d_row()
    row["run_error"] = 123
    with pytest.raises(ValueError, match="run_error"):
        validate_alpha_validation_row(row)


def test_stage_d_requires_status_reason_code() -> None:
    from src.report_schema import validate_alpha_validation_row

    row = _valid_stage_d_row()
    row.pop("status_reason_code")
    with pytest.raises(ValueError, match="status_reason_code"):
        validate_alpha_validation_row(row)
