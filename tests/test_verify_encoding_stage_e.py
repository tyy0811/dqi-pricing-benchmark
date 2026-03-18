"""Stage E paired-encoding validation tests."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


REQUIRED_COMPARATIVE_SUBSET = {
    "artifact_version",
    "mode",
    "bit_order",
    "n",
    "m",
    "B",
    "v",
    "term_weights",
    "diagnostics",
    "source_metadata",
}


def _valid_paired_payload() -> dict:
    common_meta = {
        "instance_id": "P5",
        "pricing_instance_path": "data/pricing_instances/instance_P5.json",
    }
    enc = {
        "artifact_version": "1.0",
        "mode": "xorsat",
        "bit_order": [f"x{i}" for i in range(10)],
        "n": 10,
        "m": 3,
        "B": [[1] * 10, [0] * 10, [1, 0] * 5],
        "v": [1, 0, 1],
        "term_weights": [1.0, 2.0, 3.0],
        "diagnostics": {},
        "source_metadata": dict(common_meta),
    }
    return {
        "paired_artifact_version": "1.0",
        "instance_id": "P5",
        "pairing_kind": "pricing_wht_vs_ilp_exact",
        "source_metadata": dict(common_meta),
        "encodings": {
            "wht_truncated": dict(enc),
            "ilp_derived": dict(enc),
        },
    }


def test_validation_applies_only_when_expects_flag_true() -> None:
    from src.verify_encoding import validate_paired_encoding_row

    row = {"instance_id": "P5", "expects_paired_encoding_validation": False}
    ok, reason_code, reason = validate_paired_encoding_row(row, paired_artifact=None)
    assert ok is True
    assert reason_code is None
    assert reason is None


def test_validation_passes_for_joinable_paired_artifact() -> None:
    from src.verify_encoding import validate_paired_encoding_row

    row = {"instance_id": "P5", "expects_paired_encoding_validation": True}
    ok, reason_code, reason = validate_paired_encoding_row(row, _valid_paired_payload())
    assert ok is True
    assert reason_code is None
    assert reason is None


def test_validation_failure_returns_machine_reason_code() -> None:
    from src.verify_encoding import validate_paired_encoding_row

    row = {"instance_id": "P5", "expects_paired_encoding_validation": True}
    payload = _valid_paired_payload()
    payload["encodings"]["ilp_derived"]["n"] = 8

    ok, reason_code, reason = validate_paired_encoding_row(row, payload)
    assert ok is False
    assert reason_code == "encoding_validation_failed"
    assert isinstance(reason, str)
    assert "n" in reason
