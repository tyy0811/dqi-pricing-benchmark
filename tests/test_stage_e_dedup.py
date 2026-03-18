"""Stage E regression tests for deterministic duplicate tie-break behavior."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from tests._stage_e_fixtures import stage_e_base_row


def test_completed_beats_failed_in_winner_key_order() -> None:
    from src.resources_compare import stage_e_duplicate_winner_key

    completed = stage_e_base_row()
    completed["run_status"] = "completed"
    failed = stage_e_base_row()
    failed["run_status"] = "failed"

    assert stage_e_duplicate_winner_key(completed) < stage_e_duplicate_winner_key(failed)


def test_verified_beats_unverified_in_winner_key_order() -> None:
    from src.resources_compare import stage_e_duplicate_winner_key

    verified = stage_e_base_row()
    verified["has_verified_provenance"] = True
    unverified = stage_e_base_row()
    unverified["has_verified_provenance"] = False

    assert stage_e_duplicate_winner_key(verified) < stage_e_duplicate_winner_key(unverified)


def test_earlier_manifest_slot_beats_later() -> None:
    from src.resources_compare import stage_e_duplicate_winner_key

    early = stage_e_base_row()
    early["manifest_slot"] = "slot-1"
    later = stage_e_base_row()
    later["manifest_slot"] = "slot-9"

    assert stage_e_duplicate_winner_key(early) < stage_e_duplicate_winner_key(later)


def test_lexical_run_id_breaks_final_tie() -> None:
    from src.resources_compare import stage_e_duplicate_winner_key

    a = stage_e_base_row()
    a["run_id"] = "run_a"
    b = stage_e_base_row()
    b["run_id"] = "run_b"

    assert stage_e_duplicate_winner_key(a) < stage_e_duplicate_winner_key(b)


def test_non_winners_route_to_duplicate_key_unavailable() -> None:
    from src.resources_compare import build_stage_e_quality_cost_tables

    winner = stage_e_base_row()
    winner["run_id"] = "run_a"
    winner["manifest_slot"] = "slot-1"

    loser = stage_e_base_row()
    loser["run_id"] = "run_b"
    loser["manifest_slot"] = "slot-2"

    out = build_stage_e_quality_cost_tables([loser, winner])
    assert len(out["quality_cost_master"]) == 1
    assert out["quality_cost_master"][0]["run_id"] == "run_a"
    assert len(out["quality_cost_unavailable"]) == 1
    assert out["quality_cost_unavailable"][0]["run_id"] == "run_b"
    assert out["quality_cost_unavailable"][0]["unavailable_class"] == "duplicate_key"
