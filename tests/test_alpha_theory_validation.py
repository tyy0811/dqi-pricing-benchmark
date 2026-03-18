"""Stage D runner and exact-reference tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, ".")


def test_supports_tiny_coherent_returns_machine_reason() -> None:
    from src.coherent_reference import load_coherent_instance, supports_tiny_coherent

    rec = load_coherent_instance("C1")
    ok, reason = supports_tiny_coherent(rec, ell=9, m_active=4)
    assert ok is False
    assert reason in {
        "ell_out_of_bounds",
        "coherent_exact_limit_exceeded",
    }


def test_runner_emits_one_row_per_planned_slot(tmp_path: Path) -> None:
    from src.alpha_theory_validation import run_alpha_validation

    out = run_alpha_validation(
        output_root=tmp_path,
        instance_ids=["C1"],
        alpha_modes=["paper"],
        execution_modes=["coherent", "mixture"],
        ell_values=[1],
        m_active_values=[4],
        trial_seed=42,
    )
    assert out["planned_count"] == 2
    assert out["written_count"] == 2
    assert len(out["run_files"]) == 2


def test_stage_d_rows_are_tagged_and_oracle_only(tmp_path: Path) -> None:
    from src.alpha_theory_validation import run_alpha_validation

    out = run_alpha_validation(
        output_root=tmp_path,
        instance_ids=["C1"],
        alpha_modes=["uniform"],
        execution_modes=["mixture"],
        ell_values=[1],
        m_active_values=[4],
        trial_seed=42,
    )
    row = json.loads(Path(out["run_files"][0]).read_text())
    assert row["matrix"] == "matrix_b"
    assert row["stage"] == "alpha_validation"
    assert row["family"] == "C"
    assert row["decoder"] == "oracle"
    assert "encoding" in row
    assert "canonical_trial_seed" in row
    assert "run_error" in row
    assert "status_reason_code" in row


def test_unsupported_coherent_slots_are_still_emitted(tmp_path: Path) -> None:
    from src.alpha_theory_validation import run_alpha_validation

    out = run_alpha_validation(
        output_root=tmp_path,
        instance_ids=["C1"],
        alpha_modes=["paper"],
        execution_modes=["coherent"],
        ell_values=[3],  # outside tiny coherent contract (ell <= 2)
        m_active_values=[4],
        trial_seed=42,
    )
    assert out["planned_count"] == 1
    row = json.loads(Path(out["run_files"][0]).read_text())
    assert row["run_status"] == "not_applicable"
    assert row["status_reason"] in {"ell_out_of_bounds", "coherent_exact_limit_exceeded"}
    assert row["in_experiment_matrix"] is True


def test_append_only_json_serializes_failed_and_not_applicable_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.alpha_theory_validation as mod

    def _raise(*args, **kwargs):  # pragma: no cover - test hook
        raise RuntimeError("forced-failure")

    monkeypatch.setattr(mod, "evaluate_mixture_exact", _raise)

    out = mod.run_alpha_validation(
        output_root=tmp_path,
        instance_ids=["C1"],
        alpha_modes=["paper"],
        execution_modes=["coherent", "mixture"],
        ell_values=[3],  # coherent => not_applicable
        m_active_values=[4],
        trial_seed=42,
    )
    rows = [json.loads(Path(p).read_text()) for p in out["run_files"]]
    statuses = {row["run_status"] for row in rows}
    assert "not_applicable" in statuses
    assert "failed" in statuses
    failed = [row for row in rows if row["run_status"] == "failed"][0]
    assert isinstance(failed["run_error"], str)
    assert failed["run_error"] == "forced-failure"


def test_default_family_c_grid_is_revised_discriminative_slice(tmp_path: Path) -> None:
    from src.alpha_theory_validation import run_alpha_validation

    out = run_alpha_validation(output_root=tmp_path, instance_ids=["C1", "C2", "C3"])
    rows = [json.loads(Path(p).read_text()) for p in out["run_files"]]

    expected_slots = {
        ("C1", 1, 4),
        ("C1", 2, 4),
        ("C2", 1, 6),
        ("C2", 2, 6),
        ("C3", 2, 4),
    }
    observed_slots = {(row["instance_id"], row["ell"], row["m_active"]) for row in rows}
    assert expected_slots.issubset(observed_slots)
    assert len(observed_slots.intersection(expected_slots)) == 5
    assert out["planned_count"] == 30
    assert out["written_count"] == 30
    for slot in expected_slots:
        slot_rows = [row for row in rows if (row["instance_id"], row["ell"], row["m_active"]) == slot]
        assert len(slot_rows) == 6


def test_default_grid_excludes_cap_unsafe_family_c_cores(tmp_path: Path) -> None:
    from src.alpha_theory_validation import run_alpha_validation

    out = run_alpha_validation(output_root=tmp_path, instance_ids=["C3"])
    rows = [json.loads(Path(p).read_text()) for p in out["run_files"]]
    assert all(not (r["ell"] == 2 and r["m_active"] >= 6 and r["instance_id"] == "C3") for r in rows)
    assert all(r["status_reason_code"] != "coherent_exact_limit_exceeded" for r in rows)
    assert {r["run_status"] for r in rows}.issubset({"completed"})


def test_default_family_c_grid_emits_coherent_and_mixture_for_supported_slots(tmp_path: Path) -> None:
    from src.alpha_theory_validation import run_alpha_validation

    out = run_alpha_validation(output_root=tmp_path, instance_ids=["C1", "C2", "C3"])
    rows = [json.loads(Path(p).read_text()) for p in out["run_files"]]
    expected_slots = [
        ("C1", 1, 4),
        ("C1", 2, 4),
        ("C2", 1, 6),
        ("C2", 2, 6),
        ("C3", 2, 4),
    ]
    counter = Counter((row["instance_id"], row["ell"], row["m_active"], row["alpha_mode"], row["execution_mode"]) for row in rows)
    for instance_id, ell, m_active in expected_slots:
        for alpha_mode in ("uniform", "paper", "heuristic"):
            assert counter[(instance_id, ell, m_active, alpha_mode, "coherent")] == 1
            assert counter[(instance_id, ell, m_active, alpha_mode, "mixture")] == 1


def test_default_stage_d_rows_still_use_oracle_decoder_and_stage_d_contract(tmp_path: Path) -> None:
    from src.alpha_theory_validation import run_alpha_validation

    out = run_alpha_validation(output_root=tmp_path, instance_ids=["C1", "C2", "C3"])
    rows = [json.loads(Path(p).read_text()) for p in out["run_files"]]
    for row in rows:
        assert row["decoder"] == "oracle"
        assert row["matrix"] == "matrix_b"
        assert row["stage"] == "alpha_validation"
        assert row["family"] == "C"
        assert row["encoding"]


def test_candidate_distribution_summary_is_emitted_and_alpha_sensitive(tmp_path: Path) -> None:
    from src.alpha_theory_validation import run_alpha_validation

    out = run_alpha_validation(
        output_root=tmp_path,
        instance_ids=["C1"],
        alpha_modes=["uniform", "paper", "heuristic"],
        execution_modes=["coherent"],
        ell_values=[1],
        m_active_values=[4],
        trial_seed=42,
    )
    rows = [json.loads(Path(p).read_text()) for p in out["run_files"]]
    assert rows and len(rows) == 3
    for row in rows:
        assert "candidate_entropy" in row
        assert "candidate_top1_probability" in row
        assert "candidate_top2_probability" in row
        assert row["run_status"] == "completed"
        assert row["candidate_entropy"] is not None
        assert row["candidate_top1_probability"] is not None
        assert row["candidate_top2_probability"] is not None

    entropy_by_alpha = {row["alpha_mode"]: float(row["candidate_entropy"]) for row in rows}
    assert len(set(entropy_by_alpha.values())) > 1
